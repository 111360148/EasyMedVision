#Reset
import app
import os
import streamlit as st  
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from predictor import init_state,reset_state
import matplotlib.pyplot as plt
from PIL import Image
uploaded_file, left, top, predictor, inference_state, frame_names = app.get_app_variables()
st.write("開始reset...")
inference_state = predictor.init_state(video_path=uploaded_file)
predictor.reset_state(inference_state)