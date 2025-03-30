import streamlit as st
import subprocess

st.title("EasyMedVision")

mode = st.radio(
    "Select Mode", 
    ["Label", "Train-DeepLabV3","Train-YoloV11", "Inference", "DICOM to image converter"], 
    help=(
        "- **Label**: Use SAM2 for object annotation.\n"  
        "- **Train-DeepLabV3**: Train a model using DeepLabV3.\n" 
        "- **Train-YoloV11**: Train a model using YoloV11.\n"
        "- **Inference**: Select a model and perform analysis.\n"  
        "- **DICOM to image converter**: Convert DICOM medical images to standard image formats.\n"
    )
)


if st.button("Confirm"):
    if mode == "Label":
        subprocess.Popen(["streamlit", "run", "label_mode.py"])
    elif mode == "Train-DeepLabV3":
        subprocess.Popen(["streamlit", "run", "train_deeplab_mode.py"])
    elif mode == "Train-YoloV11":
        subprocess.Popen(["streamlit", "run", "train_yolo_mode.py"])
    elif mode == "Inference":
        subprocess.Popen(["streamlit", "run", "inference_mode.py"])
    elif mode == "DICOM to image converter":
        subprocess.Popen(["streamlit", "run", "dicom_mode.py"])
