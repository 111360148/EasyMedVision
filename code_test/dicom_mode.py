import streamlit as st
import pydicom
import numpy as np
from PIL import Image
import io
import zipfile
import os

# Streamlit 介面
st.title("DICOM to Image Converter")

# 上傳 DICOM 檔案
uploaded_file = st.file_uploader("Upload a DICOM file", type=["dcm"])

if uploaded_file:
    # 讀取 DICOM 檔案
    dicom_data = pydicom.dcmread(uploaded_file)
    image_array = dicom_data.pixel_array  # 影像數據 (可能是 3D)

    # **檢查影像維度**
    if len(image_array.shape) == 3:  # 3D 影像（多張切片）
        num_slices = image_array.shape[0]  # 總切片數
    elif len(image_array.shape) == 2:  # 單張影像
        num_slices = 1
        image_array = np.expand_dims(image_array, axis=0)  # 轉成 3D 以統一處理
    else:
        st.error(f"Unexpected image shape: {image_array.shape}")
        st.stop()

    st.write(f"Total slices: {num_slices}")

    # **轉換所有切片為 PNG**
    images = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i in range(num_slices):
            slice_array = image_array[i]

            # **正規化並轉換為 8-bit 圖片**
            slice_array = (slice_array - np.min(slice_array)) / (np.max(slice_array) - np.min(slice_array))
            slice_array = (slice_array * 255).astype(np.uint8)

            # 轉換為 PIL 影像
            image = Image.fromarray(slice_array)

            # 儲存到記憶體中
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_data = img_byte_arr.getvalue()

            # 存入 zip 檔案
            zf.writestr(f"slice_{i}.png", img_data)

            # 只顯示第一張預覽
            if i == 0:
                images.append(image)

    # **顯示第一張切片作為預覽**
    st.image(images[0], caption="Preview: First Slice", use_container_width=False, width=300)


    if st.download_button(
        label="Download All Slices as ZIP",
        data=zip_buffer,
        file_name="dicom_slices.zip",
        mime="application/zip"
    ):
        st.success(f"All {num_slices} slices download completed.")