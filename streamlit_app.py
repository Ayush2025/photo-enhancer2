import io
import numpy as np
from PIL import Image
import streamlit as st
from enhancer.enhancer import Enhancer

# --- Must be first Streamlit call ---
st.set_page_config(
    page_title="FRIDAY AI Photo Enhancer",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- App Header ---
st.title("👩‍🎨 FRIDAY AI Photo Enhancer")
st.write("Enhance portraits and backgrounds in your images with state-of-the-art AI models.")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
method_label = st.sidebar.selectbox(
    "Enhancement Method",
    ("Portrait Retouch", "Advanced Restoration")
)
method = "gfpgan" if method_label == "Portrait Retouch" else "RestoreFormer"
bg_enhance = st.sidebar.checkbox("Background Enhancement", value=True)
upscale = st.sidebar.radio("Upscale Factor", [1, 2, 4], index=1)
st.sidebar.markdown("---")
st.sidebar.write("Made by **Ayush**")

# --- File Uploader ---
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.info("Please upload a JPG/PNG image to get started.")
    st.stop()

# --- Load Image ---
try:
    original_image = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error(f"Could not read image file: {e}")
    st.stop()

# --- Show Original ---
st.subheader("Original")
st.image(original_image, use_column_width=True)

# --- Initialize Enhancer ---
try:
    enhancer = Enhancer(
        method=method,
        background_enhancement=bg_enhance,
        upscale=upscale
    )
except Exception as e:
    st.error(f"Failed to initialize enhancer: {e}")
    st.stop()

# --- Perform Enhancement ---
with st.spinner("🚀 Enhancing image..."):
    try:
        result_np = enhancer.enhance(np.array(original_image))
        enhanced_image = Image.fromarray(result_np)
    except Exception as e:
        st.error(f"Enhancement error: {e}")
        st.stop()

# --- Display Side by Side ---
st.subheader("Comparison")
col1, col2 = st.columns(2)
with col1:
    st.image(original_image, caption="Original")
with col2:
    st.image(enhanced_image, caption="Enhanced")

# --- Download Button ---
buf = io.BytesIO()
enhanced_image.save(buf, format="PNG")
st.download_button(
    label="⬇️ Download Enhanced Image",
    data=buf.getvalue(),
    file_name="friday_enhanced.png",
    mime="image/png",
    key="download_enhanced"
)

# --- Method Description ---
if method == "gfpgan":
    st.markdown("**Portrait Retouch** smooths skin and sharpens facial features while retaining a natural look.")
else:
    st.markdown("**Advanced Restoration** recovers fine details, removes artifacts, and restores clarity.")

# --- Footer ---
st.markdown("---")
st.write("Built with ❤ using Streamlit and GFPGAN/RealESRGAN")
