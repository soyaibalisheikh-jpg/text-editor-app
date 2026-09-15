"""
Local Text Editor (Image/PDF) — Metadata-Safe
================================================
A Streamlit app that lets you:
  1. Upload a JPG/PNG image or a PDF.
  2. Detect text blocks (OCR for images via Tesseract, native text
     layer for PDFs via PyMuPDF).
  3. Click a detected text line (or pick it from a list) and replace it.
  4. Download the result with the ORIGINAL file's metadata preserved
     (EXIF/ICC for images, document Info dictionary for PDFs).

You do not need to understand this code to use the app — just run it
(see the setup instructions you were given) and use the interface.
"""

import io
import hashlib
from collections import Counter

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from pytesseract import Output
import fitz  # PyMuPDF
from streamlit_image_coordinates import streamlit_image_coordinates

# ----------------------------------------------------------------------
# OPTIONAL: On Windows, Tesseract is usually NOT on the system PATH.
# If you installed it with the default installer, uncomment the next
# two lines and make sure the path matches where it was installed.
# ----------------------------------------------------------------------
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================================================================
# CONSTANTS
# =========================================================================
MAX_DISPLAY_WIDTH = 900          # image is shown scaled down to this width
DEFAULT_OCR_CONFIDENCE = 40      # 0-100, ignore low-confidence OCR words
FORMAT_MAP = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}
FONT_CANDIDATES = [
    "C:\\Windows\\Fonts\\arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT_CANDIDATES_BOLD = [
    "C:\\Windows\\Fonts\\arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


# =========================================================================
# SMALL HELPERS
# =========================================================================
def get_font(size: int, bold: bool = False):
    """Try to load a real TTF font at the requested size (regular or
    bold); fall back to Pillow's built-in bitmap font if nothing is
    found (still works, just won't scale as nicely)."""
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def most_common_color(pixels):
    if not pixels:
        return (255, 255, 255)
    counts = Counter(pixels)
    return counts.most_common(1)[0][0]


def get_bg_color(image: Image.Image, box):
    """Sample a thin strip just outside the box to guess the background
    color, so we can 'erase' the old text convincingly."""
    x0, y0, x1, y1 = box
    w, h = image.size
    strip_top = max(0, y0 - 3)
    if strip_top < y0:
        crop = image.crop((max(0, x0), strip_top, min(w, x1), y0))
    else:
        crop = image.crop((max(0, x0), y1, min(w, x1), min(h, y1 + 3)))
    if crop.size[0] == 0 or crop.size[1] == 0:
        return (255, 255, 255)
    rgb_crop = crop.convert("RGB")
    return most_common_color(list(rgb_crop.getdata()))


def get_text_color(image: Image.Image, box):
    """Look inside the ORIGINAL (unedited) box for the darkest pixel and
    use its color as an estimate of the original text color."""
    x0, y0, x1, y1 = box
    crop = image.crop((x0, y0, x1, y1)).convert("RGB")
    if crop.size[0] == 0 or crop.size[1] == 0:
        return (0, 0, 0)
    gray = crop.convert("L")
    pixels = list(gray.getdata())
    rgb_pixels = list(crop.getdata())
    min_idx = min(range(len(pixels)), key=lambda i: pixels[i])
    return rgb_pixels[min_idx]


def file_identity(uploaded_file) -> str:
    return hashlib.md5(
        (uploaded_file.name + str(uploaded_file.size)).encode()
    ).hexdigest()


def reset_state():
    for key in list(st.session_state.keys()):
        if key not in ("_file_id",):
            del st.session_state[key]


# =========================================================================
# IMAGE MODE
# =========================================================================
def load_image(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(raw_bytes))
    img.load()
    exif_bytes = img.info.get("exif")
    icc_profile = img.info.get("icc_profile")
    dpi = img.info.get("dpi")
    working = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img.copy()
    st.session_state.img_working = working
    st.session_state.img_exif = exif_bytes
    st.session_state.img_icc = icc_profile
    st.session_state.img_dpi = dpi
    st.session_state.img_original_bytes = raw_bytes
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    st.session_state.img_format = FORMAT_MAP.get(ext, "PNG")
    st.session_state.img_filename = uploaded_file.name


def detect_image_text(image: Image.Image, min_conf: int):
    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
    except pytesseract.TesseractNotFoundError:
        st.error(
            "Tesseract OCR engine was not found on this computer. "
            "Install it (see setup instructions) and, on Windows, set "
            "`pytesseract.pytesseract.tesseract_cmd` near the top of app.py."
        )
        return []

    groups = {}
    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        try:
            conf = int(float(data["conf"][i]))
        except (ValueError, TypeError):
            conf = -1
        if not word or conf < min_conf:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append(i)

    boxes = []
    gid = 0
    for _, idxs in groups.items():
        # A single OCR "line" often contains a form LABEL and its VALUE
        # side by side (e.g. "Source Account Name    ATAUR RAHMAN"). If we
        # treated the whole line as one editable box, editing the value
        # would also overwrite/erase the label. So we split the line into
        # separate boxes wherever there's an unusually large horizontal
        # gap between consecutive words (bigger than a normal word-space).
        idxs = sorted(idxs, key=lambda i: data["left"][i])
        heights = [data["height"][i] for i in idxs]
        avg_h = sum(heights) / len(heights) if heights else 20
        gap_threshold = max(25, avg_h * 1.8)

        subgroups = [[idxs[0]]]
        for prev_i, curr_i in zip(idxs, idxs[1:]):
            prev_right = data["left"][prev_i] + data["width"][prev_i]
            gap = data["left"][curr_i] - prev_right
            if gap > gap_threshold:
                subgroups.append([curr_i])
            else:
                subgroups[-1].append(curr_i)

        for sub in subgroups:
            words = [data["text"][i] for i in sub]
            text = " ".join(words)
            lefts = [data["left"][i] for i in sub]
            tops = [data["top"][i] for i in sub]
            rights = [data["left"][i] + data["width"][i] for i in sub]
            bottoms = [data["top"][i] + data["height"][i] for i in sub]
            x0, y0, x1, y1 = min(lefts), min(tops), max(rights), max(bottoms)
            boxes.append(
                {"id": gid, "text": text, "left": x0, "top": y0,
                 "width": x1 - x0, "height": y1 - y0}
            )
            gid += 1
    return boxes


def apply_image_edit(image: Image.Image, box, new_text: str, font_size: int, bold: bool = False):
    x0, y0 = box["left"], box["top"]
    x1, y1 = x0 + box["width"], y0 + box["height"]
    bg_color = get_bg_color(image, (x0, y0, x1, y1))
    text_color = get_text_color(image, (x0, y0, x1, y1))
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], fill=bg_color)
    font = get_font(font_size, bold=bold)
    draw.text((x0, y0), new_text, fill=text_color, font=font)
    return image


def save_image_with_metadata() -> bytes:
    image = st.session_state.img_working
    fmt = st.session_state.img_format
    save_kwargs = {}
    if fmt == "JPEG" and image.mode != "RGB":
        image = image.convert("RGB")
    if st.session_state.img_exif:
        save_kwargs["exif"] = st.session_state.img_exif
    if st.session_state.img_icc:
        save_kwargs["icc_profile"] = st.session_state.img_icc
    if st.session_state.img_dpi:
        save_kwargs["dpi"] = st.session_state.img_dpi
    if fmt == "JPEG":
        save_kwargs["quality"] = 95
        save_kwargs["subsampling"] = 0
    buf = io.BytesIO()
    image.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


# =========================================================================
# PDF MODE
# =========================================================================
def load_pdf(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    st.session_state.pdf_doc = doc
    st.session_state.pdf_metadata = dict(doc.metadata)  # exact copy to restore later
    st.session_state.pdf_page_num = 0
    st.session_state.pdf_filename = uploaded_file.name


def render_pdf_page(page, zoom=1.7):
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img, zoom


def detect_pdf_text(page, zoom):
    """Native text layer (works for digitally-created PDFs). Groups spans
    into lines, which is the natural 'click a text area' unit."""
    d = page.get_text("dict")
    boxes = []
    line_id = 0
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s["text"].strip()]
            if not spans:
                continue
            # Same idea as the image OCR path: don't merge a form LABEL
            # and its VALUE into one box just because they're on the same
            # line. Split wherever the horizontal gap between spans is
            # much bigger than normal letter/word spacing.
            spans = sorted(spans, key=lambda s: s["bbox"][0])
            avg_size = sum(s.get("size", 11) for s in spans) / len(spans)
            gap_threshold = max(12, avg_size * 1.5)

            subgroups = [[spans[0]]]
            for prev_s, curr_s in zip(spans, spans[1:]):
                gap = curr_s["bbox"][0] - prev_s["bbox"][2]
                if gap > gap_threshold:
                    subgroups.append([curr_s])
                else:
                    subgroups[-1].append(curr_s)

            for sub in subgroups:
                text = "".join(s["text"] for s in sub).strip()
                if not text:
                    continue
                x0 = min(s["bbox"][0] for s in sub)
                y0 = min(s["bbox"][1] for s in sub)
                x1 = max(s["bbox"][2] for s in sub)
                y1 = max(s["bbox"][3] for s in sub)
                font_size = sub[0].get("size", 11)
                font_name = sub[0].get("font", "")
                font_flags = sub[0].get("flags", 0)
                color_int = sub[0].get("color", 0)
                r = ((color_int >> 16) & 255) / 255
                g = ((color_int >> 8) & 255) / 255
                b = (color_int & 255) / 255
                boxes.append({
                    "id": line_id,
                    "text": text,
                    "pdf_rect": fitz.Rect(x0, y0, x1, y1),
                    "font_size": font_size,
                    "font_name": font_name,
                    "font_flags": font_flags,
                    "color": (r, g, b),
                    "left": x0 * zoom, "top": y0 * zoom,
                    "width": (x1 - x0) * zoom, "height": (y1 - y0) * zoom,
                })
                line_id += 1
    return boxes


def get_exact_pdf_font(page, font_name):
    """Try to pull the ORIGINAL embedded font (the exact typeface used in
    the document) so replacement text can reuse it instead of a generic
    substitute. Returns (fontname_to_use, font_bytes_or_None)."""
    try:
        doc = page.parent
        for f in page.get_fonts(full=True):
            xref, ext, ftype, basefont, name, encoding = f[:6]
            if basefont == font_name or name == font_name:
                extracted = doc.extract_font(xref)
                buffer = extracted[-1] if extracted else None
                if buffer:
                    return f"embedded-{xref}", buffer
    except Exception:
        pass
    return None, None


def fallback_base14_font(font_flags: int) -> str:
    """No embedded font available (e.g. it uses one of the 14 standard
    PDF fonts, which aren't embedded as files) -- pick the closest
    standard font, matching bold/italic from the original span's flags."""
    bold = bool(font_flags & 2 ** 4)
    italic = bool(font_flags & 2 ** 1)
    if bold and italic:
        return "hebi"
    if bold:
        return "hebo"
    if italic:
        return "heit"
    return "helv"


def sample_pdf_bg_color(page, rect: fitz.Rect):
    try:
        pix = page.get_pixmap(clip=rect, dpi=72, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        colors = list(img.getdata())
        r, g, b = most_common_color(colors)
        return (r / 255, g / 255, b / 255)
    except Exception:
        return (1, 1, 1)


def apply_pdf_edit(page, box, new_text: str):
    rect = box["pdf_rect"]
    bg_color = sample_pdf_bg_color(page, rect)

    # Reuse the ORIGINAL font (same typeface, not just same size/color)
    # whenever the document has it embedded; otherwise fall back to the
    # closest standard font, still matching bold/italic.
    fontname, font_bytes = get_exact_pdf_font(page, box.get("font_name", ""))
    if font_bytes:
        page.insert_font(fontname=fontname, fontbuffer=font_bytes)
    else:
        fontname = fallback_base14_font(box.get("font_flags", 0))

    page.add_redact_annot(rect, fill=bg_color)
    page.apply_redactions()
    page.insert_text(
        (rect.x0, rect.y1 - 1),
        new_text,
        fontsize=box["font_size"],
        fontname=fontname,
        color=box["color"],
    )


def save_pdf_with_metadata() -> bytes:
    doc = st.session_state.pdf_doc
    doc.set_metadata(st.session_state.pdf_metadata)  # force-restore exact metadata
    return doc.tobytes()


# =========================================================================
# SHARED: overlay boxes + click mapping
# =========================================================================
def draw_overlay(base_image: Image.Image, boxes, display_width):
    scale = min(1.0, display_width / base_image.width)
    disp_w = int(base_image.width * scale)
    disp_h = int(base_image.height * scale)
    disp_img = base_image.resize((disp_w, disp_h)).convert("RGB")
    draw = ImageDraw.Draw(disp_img)
    for b in boxes:
        x0, y0 = b["left"] * scale, b["top"] * scale
        x1, y1 = x0 + b["width"] * scale, y0 + b["height"] * scale
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        draw.text((x0, max(0, y0 - 12)), str(b["id"]), fill=(255, 0, 0))
    return disp_img, scale


def find_box_at_point(boxes, x, y):
    for b in boxes:
        if b["left"] <= x <= b["left"] + b["width"] and b["top"] <= y <= b["top"] + b["height"]:
            return b
    return None


# =========================================================================
# ACCESS CONTROL
# =========================================================================
def check_password():
    """A simple password gate. The password itself is NOT in this code --
    it lives in Streamlit Cloud's 'Secrets' settings for this app, which
    only the app OWNER (you) can see or edit (Manage app -> Settings ->
    Secrets). Anyone you share the link with must enter it to get in, but
    they have no way to view or change it. Changing or deleting it there
    instantly locks out everyone who doesn't already have a page open."""
    try:
        correct_password = st.secrets["APP_PASSWORD"]
    except Exception:
        st.title("🔒 App password not set up yet")
        st.warning(
            "The app owner needs to add an `APP_PASSWORD` secret first: "
            "on share.streamlit.io, open this app -> Settings -> Secrets, "
            "and add a line like:\n\nAPP_PASSWORD = \"your-chosen-password\""
        )
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.title("🔒 Locked")
    entered = st.text_input("Password", type="password")
    if st.button("Unlock"):
        if entered == correct_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


# =========================================================================
# MAIN APP
# =========================================================================
def main():
    st.set_page_config(page_title="Local Text Editor", layout="wide")
    check_password()
    st.title("🖼️ Local Text Editor — Image / PDF (Metadata-Safe)")
    st.caption(
        "Everything runs on your own computer. Original EXIF (images) and "
        "document metadata (PDFs) are preserved exactly on download."
    )

    uploaded_file = st.file_uploader(
        "Upload a JPG, PNG, or PDF file", type=["jpg", "jpeg", "png", "pdf"]
    )
    if uploaded_file is None:
        st.info("Upload a file to get started.")
        return

    fid = file_identity(uploaded_file)
    if st.session_state.get("_file_id") != fid:
        reset_state()
        st.session_state._file_id = fid
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        st.session_state.mode = "pdf" if ext == "pdf" else "image"
        if st.session_state.mode == "image":
            load_image(uploaded_file)
        else:
            load_pdf(uploaded_file)

    col_controls, col_canvas = st.columns([1, 2])

    # ---------------- IMAGE MODE ----------------
    if st.session_state.mode == "image":
        image = st.session_state.img_working

        with col_controls:
            st.subheader("Detected text")
            min_conf = st.slider("OCR confidence threshold", 0, 100, DEFAULT_OCR_CONFIDENCE)
            if st.button("🔍 Detect text (OCR)"):
                with st.spinner("Running OCR..."):
                    st.session_state.boxes = detect_image_text(image, min_conf)

            boxes = st.session_state.get("boxes", [])
            if boxes:
                choice = st.selectbox(
                    "Or pick a detected line:",
                    options=[b["id"] for b in boxes],
                    format_func=lambda i: f"{i}: {next(b['text'] for b in boxes if b['id']==i)[:40]}",
                )
                if st.button("Load selected line for editing"):
                    st.session_state.selected_id = choice

            sel_id = st.session_state.get("selected_id")
            if sel_id is not None and boxes:
                sel_box = next((b for b in boxes if b["id"] == sel_id), None)
                if sel_box:
                    st.markdown(f"**Editing box #{sel_id}**")
                    new_text = st.text_input("Replacement text", value=sel_box["text"])
                    default_size = max(10, int(sel_box["height"] * 0.85))
                    size_col, bold_col = st.columns([2, 1])
                    with size_col:
                        chosen_size = st.slider(
                            "Font size (make it bigger/smaller to match the original)",
                            min_value=6, max_value=200, value=default_size, key=f"size_{sel_id}",
                        )
                    with bold_col:
                        chosen_bold = st.checkbox("Bold", key=f"bold_{sel_id}")
                    if st.button("✅ Apply edit"):
                        st.session_state.img_working = apply_image_edit(
                            st.session_state.img_working, sel_box, new_text,
                            font_size=chosen_size, bold=chosen_bold,
                        )
                        st.session_state.selected_id = None
                        st.session_state.boxes = []  # boxes are stale after an edit
                        st.rerun()

            st.divider()
            if st.button("🔄 Reset to original"):
                raw = st.session_state.img_original_bytes
                st.session_state.img_working = Image.open(io.BytesIO(raw)).convert("RGB")
                st.session_state.boxes = []
                st.session_state.selected_id = None
                st.rerun()

            st.divider()
            data = save_image_with_metadata()
            st.download_button(
                "⬇️ Download edited image (metadata preserved)",
                data=data,
                file_name="edited_" + st.session_state.img_filename,
                mime=f"image/{st.session_state.img_format.lower()}",
            )

        with col_canvas:
            boxes = st.session_state.get("boxes", [])
            disp_img, scale = draw_overlay(image, boxes, MAX_DISPLAY_WIDTH)
            st.caption("Click a red box below to select that line for editing.")
            coords = streamlit_image_coordinates(disp_img, key="img_click")
            if coords and boxes:
                real_x, real_y = coords["x"] / scale, coords["y"] / scale
                hit = find_box_at_point(boxes, real_x, real_y)
                if hit:
                    st.session_state.selected_id = hit["id"]

    # ---------------- PDF MODE ----------------
    else:
        doc = st.session_state.pdf_doc
        with col_controls:
            if doc.page_count > 1:
                st.session_state.pdf_page_num = st.number_input(
                    "Page", min_value=1, max_value=doc.page_count, value=1
                ) - 1
            page = doc[st.session_state.pdf_page_num]

            st.subheader("Detected text")
            boxes = detect_pdf_text(page, zoom=1.7)
            if not boxes:
                st.warning(
                    "No selectable text layer found on this page "
                    "(likely a scanned PDF). This app currently edits "
                    "native PDF text only."
                )
            else:
                choice = st.selectbox(
                    "Pick a detected line:",
                    options=[b["id"] for b in boxes],
                    format_func=lambda i: f"{i}: {next(b['text'] for b in boxes if b['id']==i)[:40]}",
                )
                if st.button("Load selected line for editing"):
                    st.session_state.selected_id = choice

            sel_id = st.session_state.get("selected_id")
            if sel_id is not None and boxes:
                sel_box = next((b for b in boxes if b["id"] == sel_id), None)
                if sel_box:
                    st.markdown(f"**Editing box #{sel_id}**")
                    new_text = st.text_input("Replacement text", value=sel_box["text"])
                    if st.button("✅ Apply edit"):
                        apply_pdf_edit(page, sel_box, new_text)
                        st.session_state.selected_id = None
                        st.rerun()

            st.divider()
            if st.button("🔄 Reset to original"):
                raw = uploaded_file.getvalue()
                st.session_state.pdf_doc = fitz.open(stream=raw, filetype="pdf")
                st.session_state.selected_id = None
                st.rerun()

            st.divider()
            data = save_pdf_with_metadata()
            st.download_button(
                "⬇️ Download edited PDF (metadata preserved)",
                data=data,
                file_name="edited_" + st.session_state.pdf_filename,
                mime="application/pdf",
            )

        with col_canvas:
            disp_img, scale = render_pdf_page(page, zoom=1.7)
            overlay_img, _ = draw_overlay(disp_img, boxes, MAX_DISPLAY_WIDTH)
            st.caption("Click a red box below to select that line for editing.")
            coords = streamlit_image_coordinates(overlay_img, key="pdf_click")
            if coords and boxes:
                disp_scale = min(1.0, MAX_DISPLAY_WIDTH / disp_img.width)
                real_x, real_y = coords["x"] / disp_scale, coords["y"] / disp_scale
                hit = find_box_at_point(boxes, real_x, real_y)
                if hit:
                    st.session_state.selected_id = hit["id"]


if __name__ == "__main__":
    main()
