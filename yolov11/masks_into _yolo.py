import os
import cv2
import numpy as np

# 設定資料夾
mask_dir = r"C:\yolov11-main_emv\dataset\emv\labels\valid_masks"   # Mask 存放資料夾
output_dir = r"C:\yolov11-main_emv\dataset\emv\labels\vaild_yolo"  # 轉換後的 YOLO txt 標註存放資料夾
os.makedirs(output_dir, exist_ok=True)

# 解析所有 PNG Mask
for mask_file in os.listdir(mask_dir):
    if mask_file.endswith(".png"):
        # 讀取 Mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 獲取圖片尺寸
        h, w = mask.shape[:2]

        # 找到輪廓 (contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 建立對應的 YOLO txt 檔案
        txt_filename = os.path.splitext(mask_file)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, "w") as f:
            for contour in contours:
                # 轉換座標為 YOLO 格式（歸一化）
                points = contour.reshape(-1, 2)  # 展平成 (N,2) 陣列
                norm_points = [(x/w, y/h) for x, y in points]
                
                # 過濾掉太小的物件
                if len(norm_points) > 3:
                    line = "0 " + " ".join([f"{x:.6f} {y:.6f}" for x, y in norm_points])
                    f.write(line + "\n")

print("PNG Mask 已成功轉換為 YOLO 格式！")
