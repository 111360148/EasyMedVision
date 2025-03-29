import streamlit as st
import subprocess
import os
import shutil
import random
import numpy as np
import cv2
import zipfile
import io

st.title('EasyMedVision: Training Mode')
base_path = os.getcwd()
target_base_dir = os.path.join(base_path, "DeepLabV3", "input")
train_frames_dir = os.path.join(target_base_dir, "train_images")
train_masks_dir = os.path.join(target_base_dir, "train_masks")
valid_frames_dir = os.path.join(target_base_dir, "valid_images")
valid_masks_dir = os.path.join(target_base_dir, "valid_masks")
test_frames_dir = os.path.join(target_base_dir, "test_images")
test_masks_dir = os.path.join(target_base_dir, "test_masks")

# 初始化 session_state
if "training_status" not in st.session_state:
    st.session_state.training_status = "Not started"
if "training_complete" not in st.session_state:
    st.session_state.training_complete = False
if "testing_status" not in st.session_state:
    st.session_state.testing_status = "Not started"
if "testing_complete" not in st.session_state:
    st.session_state.testing_complete = False
if "analysis_names" not in st.session_state:
    st.session_state.analysis_names = [""]  # 預設一個輸入框
    
for folder in [train_frames_dir, train_masks_dir, valid_frames_dir, valid_masks_dir, test_frames_dir, test_masks_dir]:
    os.makedirs(folder, exist_ok=True)  # 確保資料夾存在

    # 刪除資料夾內的所有檔案
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # 刪除檔案或符號連結
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 刪除資料夾及其內容
        except Exception as e:
            print(f"❌ 無法刪除 {file_path}: {e}")

# **➕ 按鈕來新增輸入框**
if st.button("➕ Add Another Analysis"):
    st.session_state.analysis_names.append("")

# **動態輸入框**
analysis_names = []
for i in range(len(st.session_state.analysis_names)):
    analysis_name = st.text_input(f"Enter analysis name {i+1}", st.session_state.analysis_names[i])
    st.session_state.analysis_names[i] = analysis_name
    if analysis_name.strip():
        analysis_names.append(analysis_name)

# **顯示當前訓練狀態**
st.write(f"### Current Status: {st.session_state.training_status}")

# **檢查是否有輸入有效的分析名稱**
if analysis_names:
    total_images = 0
    train_count = 0
    valid_count = 0
    test_count = 0

    for analysis_name in analysis_names:
        output_dir = os.path.join(base_path, analysis_name)

        if not os.path.exists(output_dir):
            st.error(f"The folder \"{analysis_name}\" does not exist! Please check your input.")
            continue

        frames_dir = os.path.join(output_dir, "frames_select")
        masks_dir = os.path.join(output_dir, "masks")

        if os.path.exists(frames_dir) and os.path.exists(masks_dir):
            frame_files = sorted(os.listdir(frames_dir))
            mask_files = sorted(os.listdir(masks_dir))

            if len(frame_files) != len(mask_files):
                st.error(f"Mismatch between 'frames' and 'masks' count in \"{analysis_name}\"! Please check the dataset.")
                continue

            data = list(zip(frame_files, mask_files))
            random.shuffle(data)

            split_train_idx = int(0.7 * len(data))
            split_valid_idx = int(0.9 * len(data))

            train_data = data[:split_train_idx]
            valid_data = data[split_train_idx:split_valid_idx]
            test_data = data[split_valid_idx:]

            total_images += len(data)
            train_count += len(train_data)
            valid_count += len(valid_data)
            test_count += len(test_data)

            for frame, mask in train_data:
                new_frame_name = f"{analysis_name}_{frame}"
                new_mask_name = f"{analysis_name}_{mask}"
                shutil.copy(os.path.join(frames_dir, frame), os.path.join(train_frames_dir, new_frame_name))
                shutil.copy(os.path.join(masks_dir, mask), os.path.join(train_masks_dir, new_mask_name))

            for frame, mask in valid_data:
                new_frame_name = f"{analysis_name}_{frame}"
                new_mask_name = f"{analysis_name}_{mask}"
                shutil.copy(os.path.join(frames_dir, frame), os.path.join(valid_frames_dir, new_frame_name))
                shutil.copy(os.path.join(masks_dir, mask), os.path.join(valid_masks_dir, new_mask_name))

            for frame, mask in test_data:
                new_frame_name = f"{analysis_name}_{frame}"
                new_mask_name = f"{analysis_name}_{mask}"
                shutil.copy(os.path.join(frames_dir, frame), os.path.join(test_frames_dir, new_frame_name))
                shutil.copy(os.path.join(masks_dir, mask), os.path.join(test_masks_dir, new_mask_name))

    st.success(f"Dataset successfully split! 🎯\n"
               f"Total images: {total_images}\n"
               f"Train: {train_count} images (70%)\n"
               f"Validation: {valid_count} images (20%)\n"
               f"Test: {test_count} images (10%)")

    # **設定訓練參數**
    st.sidebar.header("Training Parameters")
    epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=10000, value=30)
    batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=64, value=4)
    lr = st.sidebar.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-3, format="%.6f")

    if st.sidebar.button("Train Model"):
        st.session_state.training_status = "Training..."
        st.session_state.training_complete = False
        st.rerun()

    if st.session_state.training_status == "Training...":
        with st.spinner("Training in progress..."):
            train_script_path = os.path.join(base_path, "DeepLabV3", "src", "train.py")
            command = f'python "{train_script_path}" --epochs {epochs} --batch {batch_size} --lr {lr}'


            
            # 執行子行程，捕獲 stdout 和 stderr
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=os.path.join(base_path, "DeepLabV3", "src"), text=True, bufsize=1)

            progress_file = os.path.join(base_path, "DeepLabV3", "outputs", "train_progress.txt")

            progress_bar = st.progress(0)  # 進度條
            log_container = st.empty()  # 用來顯示終端機輸出
            current_epoch = 0

            # **讀取 stdout 並即時顯示**
            log_output = []
            for line in iter(process.stdout.readline, ''):  
                log_output.append(line.strip())  # 儲存輸出
                log_container.text_area("Training Log", "\n".join(log_output), height=300)  # 更新 UI

                # **非阻塞方式讀取當前 epoch**
                if os.path.exists(progress_file):
                    with open(progress_file, "r") as f:
                        try:
                            current_epoch = int(f.read().strip())
                        except ValueError:
                            pass
                
                # 更新進度條
                progress_bar.progress(min(current_epoch / epochs, 1.0))

                # **當 current_epoch 達到 epochs，提前結束迴圈**
                if current_epoch >= epochs:
                    break

            process.wait()  # 等待 subprocess 完成
            if process.returncode != 0:
                error_message = process.stderr.read()  # 讀取錯誤輸出
                st.error(f"Training failed with error:\n\n{error_message}")
                st.session_state.training_status = "Training Failed"
                st.session_state.training_complete = False
                st.stop()
            progress_bar.progress(1.0)  # 訓練完成時填滿進度條

            st.session_state.training_status = "Training Completed"
            st.session_state.training_complete = True
            st.rerun()

    if st.session_state.training_complete:

        with st.spinner("Testing in progress..."):
            test_script_path = os.path.join(base_path, "DeepLabV3", "src", "inference_image.py")
            outputs = os.path.join(base_path, "DeepLabV3", "src", "outputs", "inference_masks")
            model_path = os.path.join(base_path, "DeepLabV3", "src", "outputs", "best_model.pth")
            test_command = f'python "{test_script_path}" --input "../input/test_images" --output "{outputs}" --model "{model_path}"'
            subprocess.run(test_command, shell=True, cwd=os.path.join(base_path, "DeepLabV3", "src"))

        iou_scores = []
        inference_masks_dir = os.path.join(base_path, "DeepLabV3", "src", "outputs","inference_masks")

        for mask_name in os.listdir(test_masks_dir):
            test_mask_path = os.path.join(test_masks_dir, mask_name)
            infer_mask_path = os.path.join(inference_masks_dir, mask_name)

            if os.path.exists(infer_mask_path):
                test_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
                infer_mask = cv2.imread(infer_mask_path, cv2.IMREAD_GRAYSCALE)
                infer_mask = cv2.resize(infer_mask, (test_mask.shape[1], test_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                intersection = np.logical_and(test_mask, infer_mask).sum()
                union = np.logical_or(test_mask, infer_mask).sum()
                iou = intersection / union if union > 0 else 0
                iou_scores.append(iou)

        avg_iou = np.mean(iou_scores) if iou_scores else 0
        st.write(f"### **Mean IoU Accuracy: {avg_iou:.4f}** 🎯")

        
        target_output_dir = os.path.join(base_path, analysis_names[0], "outputs")
        trained_model_dir = os.path.join(base_path, "DeepLabV3", "src", "outputs")

        accuracy_img_path = os.path.join(trained_model_dir, "accuracy.png")
        loss_img_path = os.path.join(trained_model_dir, "loss.png")

        if os.path.exists(accuracy_img_path) and os.path.exists(loss_img_path):
            st.image([accuracy_img_path, loss_img_path], caption=["Accuracy", "Loss"])
        else:
            st.warning("accuracy.png or loss.png not found in outputs folder.")

        if st.button("Save Model"):
            if os.path.exists(trained_model_dir):
                if os.path.exists(target_output_dir):
                    shutil.rmtree(target_output_dir)
                shutil.move(trained_model_dir, target_output_dir)
                st.success(f"Model and outputs folder saved to {target_output_dir}")
            else:
                st.error("Training outputs directory not found!")

        if os.path.exists(target_output_dir):
            # 壓縮 target_output_dir 資料夾到記憶體中的 zip 檔
            zip_filename = "model_outputs.zip"
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for root, dirs, files in os.walk(target_output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 利用相對路徑存入 zip
                        arcname = os.path.relpath(file_path, target_output_dir)
                        zip_file.write(file_path, arcname)
            zip_buffer.seek(0)
            st.download_button(
                label="Download Entire Folder",
                data=zip_buffer,
                file_name=zip_filename,
                mime="application/zip"
            )
        else:
            st.warning("Target output folder not found for download.")

else:
    st.error("Please enter at least one analysis name.")