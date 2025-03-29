import streamlit as st
import os
import zipfile
import shutil
import subprocess


st.title('EasyMedVision: Inference Mode')

# **使用者輸入分析名稱**
analysis_name = st.text_input("Enter analysis name")
trained_analysis_name = st.text_input("Enter trained analysis name")

if not analysis_name or not trained_analysis_name:
    st.warning("Please enter both an analysis name and a trained analysis name to proceed.")
    st.stop()

# **根據 trained_analysis_name 獲取對應的模型路徑**
base_path = os.getcwd()
analysis_path = os.path.join(base_path, analysis_name)
images_path = os.path.join(analysis_path, "images")
masks_path = os.path.join(analysis_path, "masks")
model_path = os.path.join(base_path, trained_analysis_name, "outputs", "best_model.pth")

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

# **上傳壓縮檔案（zip）**
uploaded_file = st.file_uploader("Upload a .zip file", type=["zip"])

if uploaded_file:
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    
    temp_zip_path = os.path.join(base_path, "temp_file.zip")
    with open(temp_zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(images_path)
    
    os.remove(temp_zip_path)
    st.success("File extracted successfully!")
    
    # **確保 masks_path 存在**
    os.makedirs(masks_path, exist_ok=True)
    
    # **執行 Inference 指令**
    test_script_path = os.path.join(base_path, "DeepLabV3", "src", "inference_image.py")
    test_command = f'python "{test_script_path}" --input "{images_path}" --output "{masks_path}" --model "{model_path}"'
    subprocess.run(test_command, shell=True, cwd=os.path.join(base_path, "DeepLabV3", "src"))
    st.write("Run the following command in your terminal:")
    st.code(test_command)