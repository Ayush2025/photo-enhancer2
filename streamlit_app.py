import io
import sys
import os
import numpy as np
import streamlit as st
from PIL import Image
from enhancer.enhancer import Enhancer

# --- Page config (MUST be first Streamlit call) ---
st.set_page_config(
    page_title="FRIDAY - AI Photo Enhancer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Diagnostic Check ---
st.write("✅ FRIDAY App Initialized")
