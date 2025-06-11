import io
import os
import sys
import numpy as np
import streamlit as st
from PIL import Image

# Ensure enhancer package path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from enhancer.enhancer import Enhancer

# --- Page config (must be first) ---
st.set_page_config(
    page_title="FRIDAY AI Photo Enhancer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS & Animations ---
st.markdown('''
<style>
/* Full-page animated gradient background */
body {
  background: linear-gradient(120deg, #1e1e2e, #12121b, #1e1e2e);
  background-size: 400% 400%;
  animation: gradientBG 20s ease infinite;
  color: #fff;
}
@keyframes gradientBG {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* Sidebar styling */
[data-testid="stSidebar"] > div:first-child {
  background-color: #1b1b2f !important;
}

/* Title animation */
.title-anim {
  font-size: 4rem;
  font-weight: 700;
  text-align: center;
  background: linear-gradient(90deg, #ff6ec4, #7873f5, #4ade80, #facc15);
  background-clip: text;
  -webkit-background-clip: text;
  color: transparent;
  animation: titleGradient 7s ease infinite;
}
@keyframes titleGradient {
  0% { background-position: 0% 50%; }
  100% { background-position: 100% 50%; }
}

/* Image card */
.image-card {
  padding: 1rem;
  border-radius: 0.75rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
  background: rgba(255,255,255,0.05);
}

/* Buttons */
button {
  border-radius: 0.5rem !important;
}
</style>
''', unsafe_allow_html=True)

# --- App Header ---
st.markdown("<div class='title-anim'>FRIDAY</div>", unsafe_allow_html=True)
st.markdown("## Next-generation AI Photo Enhancer")

# --- Sidebar Controls ---
st.sidebar.title("Settings")
method = st.sidebar.selectbox(
    "Enhancement Method",
    {"Portrait Retouch": "gfpgan", "Advanced Restoration": "RestoreFormer"}
)
bg_enhance = st.sidebar.checkbox("Background Enhancement", value=True)
upscale = st.sidebar.radio("Upscale Factor", [1, 2, 4], index=1)
st.sidebar.markdown("---")
st.sidebar.write("Built by **Ayush** 💡")

# --- File Uploader ---
uploaded = st.file_uploader("📂 Upload an image", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.warning("Please upload a JPG/PNG image to continue.")
    st.stop()

# --- Preview Original ---
original = Image.open(uploaded).convert("RGB")
st.subheader("Original")
st.image(original, use_column_width=True, caption="Your upload")

# --- Enhance Button ---
if st.button("✨ Enhance Image"):
    # Instantiate Enhancer once, cached for performance
    @st.cache_resource
    def get_enhancer(method, bg, up):
        return Enhancer(method=method, background_enhancement=bg, upscale=up)

    enhancer = get_enhancer(method, bg_enhance, upscale)

    with st.spinner("Processing image..."):
        try:
            result_np = enhancer.enhance(np.array(original))
            enhanced = Image.fromarray(result_np)
        except Exception as e:
            st.error(f"Enhancement failed: {e}")
            st.stop()

    # --- Show Results ---
    st.subheader("Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.image(original, caption="Original", use_column_width=True)
    with c2:
        st.image(enhanced, caption="Enhanced", use_column_width=True)

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

    # --- Method Info ---
    if method == "gfpgan":
        st.info("**Portrait Retouch**: Smooths skin and sharpens facial features, preserving natural appearance.")
    else:
        st.info("**Advanced Restoration**: Recovers details, removes artifacts, and restores clarity.")
```
