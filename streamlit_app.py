import io
import sys
import os
import numpy as np
import streamlit as st
from PIL import Image
from enhancer.enhancer import Enhancer

# --- Wrap entire app to catch unexpected errors ---
try:
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

    # --- Custom CSS and animations ---
    st.markdown("""
    <style>
      body {
        background: radial-gradient(circle at 50% 0%, #1e1e2e, #12121b);
        color: #fff;
        animation: bgPulse 15s ease infinite;
      }
      @keyframes bgPulse {
        0%,100% { background-size: 100% 100%; }
        50% { background-size: 120% 120%; }
      }
      .title-anim {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        background: linear-gradient(90deg, #ff6ec4, #7873f5, #4ade80, #facc15, #fb7185);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        animation: titleGradient 5s ease infinite;
      }
      @keyframes titleGradient {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
      }
    </style>
    """, unsafe_allow_html=True)

    # --- Animated title ---
    st.markdown("<div class='title-anim'>FRIDAY</div>", unsafe_allow_html=True)
    st.markdown("### Next-gen AI Photo Enhancer")
    st.divider()

    # --- File uploader ---
    uploaded = st.file_uploader("📂 Upload an image", type=['png', 'jpg', 'jpeg'])

    # --- Sidebar controls ---
    st.sidebar.header("App Settings:")
    method = st.sidebar.selectbox(
        "Enhancement Method",
        [
            ("Portrait Retouch", "gfpgan"),
            ("Advanced Restoration", "RestoreFormer"),
        ],
        format_func=lambda x: x[0]
    )
    bg_enhance = st.sidebar.checkbox("Background Enhancement", value=True)
    upscale = st.sidebar.radio("Upscale Factor", [2, 4], index=0)
    width = st.sidebar.slider("Display Width", 100, 800, 400)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made by **Ayush** 💡")

    # --- Main logic ---
    if uploaded:
        img_np = np.array(Image.open(uploaded))
        enhance_key = method[1]

        try:
            enhancer = Enhancer(
                method=enhance_key,
                background_enhancement=bg_enhance,
                upscale=upscale
            )
        except Exception as e:
            st.error(f"Initialization error in Enhancer: {e}")
            st.stop()

        with st.spinner("✨ Enhancing—please wait..."):
            try:
                out_np = enhancer.enhance(img_np)
            except Exception as e:
                st.error(f"Enhancement error: {e}")
                st.stop()

        out_img = Image.fromarray(out_np)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original 🖼️")
            st.image(uploaded, width=width)
        with col2:
            st.subheader("Enhanced 🚀")
            st.image(out_img, width=width)

        # Download button with unique key
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=buf.getvalue(),
            file_name="FRIDAY_enhanced.png",
            mime="image/png",
            key="download_btn"
        )

        # Method descriptions
        if enhance_key == 'gfpgan':
            st.info("**Portrait Retouch**: Smooths skin, sharpens facial features, keeps natural look.")
        else:
            st.info("**Advanced Restoration**: Recovers fine details, removes artifacts, restores old photos.")
    else:
        st.info("Upload an image to get started!")

except Exception as main_e:
    st.error(f"Unexpected application error: {main_e}")
    st.stop()
