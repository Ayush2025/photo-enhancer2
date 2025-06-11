import io
import os
import sys
import numpy as np
import streamlit as st
from PIL import Image

# Ensure enhancer package is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from enhancer.enhancer import Enhancer

# --- Page Configuration ---
st.set_page_config(
    page_title="FRIDAY",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global CSS & Animations ---
st.markdown(
    """
    <style>
      body {
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #2c3e50, #4ca1af);
        color: #fff;
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
      }
      @keyframes gradientBG {
        0% {background-position:0% 50%;}
        50% {background-position:100% 50%;}
        100% {background-position:0% 50%;}
      }
      .title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #ff6ec4, #7873f5, #4ade80, #facc15);
        -webkit-background-clip: text;
        color: transparent;
      }
      .sidebar .sidebar-content {
        background-color: #1b1b2f;
      }
      .enhance-btn button {
        animation: pulse 2s ease infinite;
      }
      @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
      }
    </style>
    """, unsafe_allow_html=True
)

# --- App Header ---
st.markdown("<div class='title'>FRIDAY</div>", unsafe_allow_html=True)
st.write("Enhance portraits and backgrounds with next-gen AI.")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
method_label = st.sidebar.radio(
    "Enhancement Method",
    ["Portrait Retouch", "Advanced Restoration"]
)
method = "gfpgan" if method_label == "Portrait Retouch" else "RestoreFormer"
bg_enhance = st.sidebar.checkbox("Background Enhancement", True)
upscale = st.sidebar.selectbox("Upscale Factor", [1, 2, 4], index=1)
width = st.sidebar.slider("Display Width", 100, 800, 400)
st.sidebar.markdown("---")
st.sidebar.write("Built by **Ayush** 💡")

# --- File Uploader ---
uploaded = st.file_uploader("📂 Upload an image", type=["png","jpg","jpeg"])
if not uploaded:
    st.info("Please upload an image to continue.")
    st.stop()

# --- Display Original ---
original = Image.open(uploaded).convert("RGB")
st.subheader("Original Image")
st.image(original, width=width, caption="Your upload")

# --- Animated Enhance Button ---
st.markdown("<div class='enhance-btn'>", unsafe_allow_html=True)
pressed = st.button("✨ Enhance Image", key="enhance_btn")
st.markdown("</div>", unsafe_allow_html=True)

if pressed:
    with st.spinner("Processing image..."):
        try:
            enhancer = Enhancer(
                method=method,
                background_enhancement=bg_enhance,
                upscale=upscale
            )
            enhanced_np = enhancer.enhance(np.array(original))
            enhanced = Image.fromarray(enhanced_np)
        except Exception as err:
            st.error(f"Enhancement failed: {err}")
            st.stop()

    # --- Show Comparison ---
    st.subheader("Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original", width=width)
    with col2:
        st.image(enhanced, caption="Enhanced", width=width)

    # --- Download Button ---
    buf = io.BytesIO()
    enhanced.save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download Enhanced",
        data=buf.getvalue(),
        file_name="friday_enhanced.png",
        mime="image/png",
        key="download"
    )

    # --- Method Details ---
    if method == "gfpgan":
        st.markdown("**Portrait Retouch**: Smooths skin and sharpens facial features while preserving natural look.")
    else:
        st.markdown("**Advanced Restoration**: Recovers fine details, removes artifacts, and restores clarity.")

else:
    st.info("Press the Enhance button above to begin!")
