from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from src.infer import run_classic_inference
from src.utils import bgr_to_rgb, image_to_png_bytes, read_uploaded_image, readable_filename, save_image
from src.visualize import make_side_by_side, overlay_mask_and_boxes


st.set_page_config(
    page_title="Metal Surface Scratch Detector",
    page_icon="🔍",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 0.15rem;
}
.subtitle {
    color: #9aa4b2;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}
.metric-card {
    background: #161b22;
    border: 1px solid #263040;
    border-radius: 16px;
    padding: 1rem;
}
.status-ok {
    background: rgba(35, 134, 54, 0.18);
    border: 1px solid rgba(35, 134, 54, 0.55);
    border-radius: 14px;
    padding: 0.85rem 1rem;
}
.status-alert {
    background: rgba(248, 81, 73, 0.14);
    border: 1px solid rgba(248, 81, 73, 0.55);
    border-radius: 14px;
    padding: 0.85rem 1rem;
}
.footer-note {
    color: #8b949e;
    font-size: 0.84rem;
    margin-top: 1.6rem;
}
hr {
    border: none;
    border-top: 1px solid #263040;
    margin: 1rem 0 1.25rem 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="main-title">Metal Surface Scratch Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a metal-surface image and inspect possible scratches with a ready-to-run OpenCV baseline.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    max_dim = st.slider("Max image dimension", min_value=640, max_value=1800, value=1408, step=32)
    blur_kernel = st.select_slider("Blur kernel", options=[1, 3, 5, 7], value=3)
    clip_limit = st.slider("CLAHE clip limit", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    threshold_bias = st.slider("Threshold sensitivity", min_value=0.7, max_value=2.2, value=1.15, step=0.05)
    min_area = st.slider("Minimum candidate area", min_value=5, max_value=120, value=18, step=1)
    show_debug = st.checkbox("Show debug maps", value=True)

uploaded_file = st.file_uploader(
    "Upload one image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("Upload a metal surface image to start the inspection.")
    st.markdown(
        """
        **Recommended images**
        - close-up surface photos
        - good lighting
        - minimal motion blur
        - visible scratch direction
        """
    )
    st.markdown('<div class="footer-note">By Alan Masoud</div>', unsafe_allow_html=True)
    st.stop()

try:
    image_bgr = read_uploaded_image(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded image: {exc}")
    st.stop()

with st.spinner("Analyzing image..."):
    output = run_classic_inference(
        image_bgr=image_bgr,
        max_dim=max_dim,
        blur_kernel=blur_kernel,
        clip_limit=clip_limit,
        threshold_bias=threshold_bias,
        min_area=min_area,
    )

result = output.result
annotated_bgr = overlay_mask_and_boxes(output.resized_image_bgr, result)
comparison_bgr = make_side_by_side(output.resized_image_bgr, annotated_bgr)

if result.defect_detected:
    st.markdown(
        f'<div class="status-alert"><strong>Status:</strong> {result.label} &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Confidence:</strong> {result.confidence:.1%}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="status-ok"><strong>Status:</strong> {result.label} &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Confidence:</strong> {result.confidence:.1%}</div>',
        unsafe_allow_html=True,
    )

st.write("")

col1, col2, col3 = st.columns([1, 1, 0.9])

with col1:
    st.subheader("Original")
    st.image(bgr_to_rgb(output.resized_image_bgr), use_container_width=True)

with col2:
    st.subheader("Annotated result")
    st.image(bgr_to_rgb(annotated_bgr), use_container_width=True)

with col3:
    st.subheader("Inspection summary")
    st.metric("Confidence", f"{result.confidence:.1%}")
    st.metric("Candidates", f"{len(result.boxes)}")
    st.metric("Mask pixels", f"{result.metadata['mask_pixels']:,}")
    st.caption(f"Threshold: {result.metadata['threshold']:.2f}")
    st.caption(f"Score: {result.score:.4f}")

    filename = readable_filename(uploaded_file.name, suffix="_annotated.png")
    png_bytes = image_to_png_bytes(annotated_bgr)
    st.download_button(
        label="Download final result",
        data=png_bytes,
        file_name=filename,
        mime="image/png",
        use_container_width=True,
    )

    save_now = st.button("Save result inside outputs/predictions", use_container_width=True)
    if save_now:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("outputs") / "predictions" / f"{Path(filename).stem}_{timestamp}.png"
        save_image(out_path, annotated_bgr)
        st.success(f"Saved to: {out_path}")

st.markdown("---")
st.subheader("Before / after")
st.image(bgr_to_rgb(comparison_bgr), use_container_width=True)

if show_debug:
    st.markdown("---")
    st.subheader("Debug maps")
    dbg1, dbg2, dbg3 = st.columns(3)

    with dbg1:
        st.image(result.debug_images["enhanced_gray"], caption="Enhanced grayscale", use_container_width=True, clamp=True)
        st.image(result.debug_images["binary"], caption="Binary threshold", use_container_width=True, clamp=True)

    with dbg2:
        st.image(result.debug_images["blackhat"], caption="Directional black-hat", use_container_width=True, clamp=True)
        st.image(result.debug_images["cleaned_mask"], caption="Cleaned mask", use_container_width=True, clamp=True)

    with dbg3:
        st.image(result.debug_images["gradient"], caption="Gradient energy", use_container_width=True, clamp=True)
        st.image(result.debug_images["fusion"], caption="Fusion map", use_container_width=True, clamp=True)

st.markdown('<div class="footer-note">Built as a complete starter project for GitHub. By Alan Masoud</div>', unsafe_allow_html=True)
