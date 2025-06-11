import os
import sys
# Ensure enhancer package can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import numpy as np
import streamlit as st
from PIL import Image

# --- Diagnostics: confirm imports ---
# Must set page config before any other Streamlit call
st.set_page_config(
    page_title="FRIDAY AI Photo Enhancer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Attempt to import Enhancer
try:
    from enhancer.enhancer import Enhancer
    st.write("✅ Enhancer module loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to import Enhancer: {e}")
    st.stop()

# --- App Header ---
st.title("👩‍🎨 FRIDAY AI Photo Enhancer")
st.write("Enhance portraits and backgrounds using AI-driven models.")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
method_label = st.sidebar.selectbox(
    "Enhancement Method",
    ["Portrait Retouch", "Advanced Restoration"]
)
method = "gfpgan" if method_label == "Portrait Retouch" else "RestoreFormer"
bg_enhance = st.sidebar.checkbox("Background Enhancement", value=True)
upscale = st.sidebar.selectbox("Upscale Factor", [1, 2, 4], index=1)

st.sidebar.markdown("---")
st.sidebar.write("Built by **Ayush**")

# --- File Uploader ---
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to get started.")
    st.stop()

# --- Load and display original image ---
try:
    original = Image.open(uploaded).convert("RGB")
    st.subheader("Original Image")
    st.image(original, use_column_width=True)
except Exception as e:
    st.error(f"Error loading image: {e}")
    st.stop()

# --- Initialize Enhancer ---
try:
    enhancer = Enhancer(
        method=method,
        background_enhancement=bg_enhance,
        upscale=upscale
    )
    st.write(f"✅ Using method: {method_label}")
except Exception as e:
    st.error(f"Failed to initialize Enhancer: {e}")
    st.stop()

# --- Perform enhancement ---
with st.spinner("Enhancing image, please wait..."):
    try:
        enhanced_np = enhancer.enhance(np.array(original))
        enhanced = Image.fromarray(enhanced_np)
    except Exception as e:
        st.error(f"Enhancement error: {e}")
        st.stop()

# --- Show comparison ---
st.subheader("Comparison")
col1, col2 = st.columns(2)
with col1:
    st.image(original, caption="Original", use_column_width=True)
with col2:
    st.image(enhanced, caption="Enhanced", use_column_width=True)

# --- Download button ---
buffer = io.BytesIO()
enhanced.save(buffer, format="PNG")
st.download_button(
    label="Download Enhanced Image",
    data=buffer.getvalue(),
    file_name="friday_enhanced.png",
    mime="image/png",
    key="download_img"
)

# --- Method details ---
if method == "gfpgan":
    st.info("Portrait Retouch: Smooth skin, sharpen features while keeping a natural look.")
else:
    st.info("Advanced Restoration: Recover fine details and remove artifacts.")

# --- Footer ---
st.markdown("---")
st.write("Powered by Streamlit, GFPGAN & RealESRGAN")
