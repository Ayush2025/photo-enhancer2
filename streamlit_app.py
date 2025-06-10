import os


import io
import streamlit as st
from PIL import Image
import numpy as np
from enhancer.enhancer import Enhancer

# --- Page config ---
st.set_page_config(
    page_title="FRIDAY - AI Photo Enhancer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Hide default menu + footer ---
st.markdown("""
<style>
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Custom CSS (dark sidebar, animations, colors, animated background) ---
st.markdown("""
<style>
  /* Animated page background */
  body {
    background: linear-gradient(135deg,
      #ff9a9e 0%, #fad0c4 25%, #fad0c4 25%, #fbc2eb 50%, #a6c1ee 75%, #84fab0 100%
    );
    background-size: 600% 600%;
    animation: gradientBG 20s ease infinite;
  }
  @keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  /* Dark sidebar */
  [data-testid="stSidebar"] {
    background-color: #1e1e2e !important;
    color: #ffffff;
  }

  /* Animated rainbow title */
  @keyframes titleGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  .title-anim {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 0.5rem;
    background: linear-gradient(45deg,
      #ff6ec4, #7873f5, #4ade80, #facc15, #fb7185
    );
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: titleGradient 8s ease infinite;
  }

  /* Uploader box */
  .stFileUploader > div {
    border: 2px dashed #ff6ec4 !important;
    border-radius: 0.75rem;
    padding: 1rem !important;
  }

  /* Image cards */
  .image-card {
    padding: 1rem;
    border-radius: 0.75rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
  }
  .image-card-left { background: #FFFBCC; }
  .image-card-right { background: #CCF0FF; }

  /* Buttons */
  .enhance-btn button,
  .download-btn button {
    color: #fff;
    padding: 0.6rem 1.2rem;
    border: none;
    border-radius: 0.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
  }
  .enhance-btn button {
    background-color: #f39c12;
  }
  .download-btn button {
    background-color: #27ae60;
  }
  .enhance-btn button:hover,
  .download-btn button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
  }
</style>
""", unsafe_allow_html=True)

# --- Animated title ---
st.markdown("<div class='title-anim'>FRIDAY</div>", unsafe_allow_html=True)
st.markdown("### Next-gen AI Photo Enhancer")
st.divider()

# --- File uploader ---
uploaded = st.file_uploader("📂 Upload an image", type=['png','jpg','jpeg'])

# --- Sidebar controls ---
st.sidebar.header("App Settings:")
method = st.sidebar.selectbox("Method", ["gfpgan", "RestoreFormer", "codeformer"])
bg_enhance = st.sidebar.selectbox("Background enhancement", ["True","False"])
bg_enhance = True if bg_enhance=="True" else False
upscale = st.sidebar.selectbox("Upscale factor", [2,4])
width = st.sidebar.slider("Display width", 100, 600, 300)
st.sidebar.markdown("---")
st.sidebar.markdown("Made by **Ayush**")

# --- Main logic ---
if uploaded:
    img_np = np.array(Image.open(uploaded))
    enhancer = Enhancer(
        method=method,
        background_enhancement=bg_enhance,
        upscale=upscale
    )

    with st.spinner("✨ Enhancing—please wait..."):
        out_np = enhancer.enhance(img_np)
    out_img = Image.fromarray(out_np)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='image-card image-card-left'>", unsafe_allow_html=True)
        st.subheader("Original")
        st.image(uploaded, width=width)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='image-card image-card-right'>", unsafe_allow_html=True)
        st.subheader("Enhanced")
        st.image(out_img, width=width)
        st.markdown("</div>", unsafe_allow_html=True)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    data = buf.getvalue()
    st.markdown("<div class='download-btn'>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download Enhanced",
        data=data,
        file_name="FRIDAY_enhanced.png",
        mime="image/png"
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='enhance-btn'>", unsafe_allow_html=True)
    st.button("Enhance", help="Upload an image first", disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)
