import torch
import argparse
import cv2
import os
import numpy as np
from PIL import Image
from utils import get_segment_labels, draw_segmentation_map
from config import ALL_CLASSES
from model import prepare_model

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Path to input directory')
parser.add_argument('-o', '--output', default='./outputs/inference_masks', help='Path to output directory')
parser.add_argument('-m', '--model', help='Path to trained model')
args = parser.parse_args()

# Create output directory if not exist.
os.makedirs(args.output, exist_ok=True)

# Set computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model.
model = prepare_model(len(ALL_CLASSES))
ckpt = torch.load(args.model)
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

# Process images in the input directory.
all_image_paths = os.listdir(args.input)
for i, image_name in enumerate(all_image_paths):
    print(f"Processing Image {i+1}/{len(all_image_paths)}: {image_name}")

    # Read the image and記錄原始尺寸
    image_path = os.path.join(args.input, image_name)
    image = Image.open(image_path)
    original_size = image.size  # (width, height)

    # Resize very large images (if width > 1024) to avoid OOM on GPUs.
    if image.size[0] > 1024:
        image = image.resize((800, 800))

    # Perform segmentation.
    outputs = get_segment_labels(image, model, device)
    outputs = outputs['out']  # Get the segmented output
    mask = draw_segmentation_map(outputs)  # Get segmentation mask (NumPy array)

    # Convert mask to PIL image (ensure it's grayscale)
    mask_pil = Image.fromarray(mask.astype(np.uint8))

    # 將 mask 調整回原始圖片的尺寸，使用 nearest neighbor 插值以保持標籤一致
    if mask_pil.size != original_size:
        mask_pil = mask_pil.resize(original_size, resample=Image.NEAREST)

    # Save mask as PNG to retain quality
    output_mask_path = os.path.join(args.output, image_name.replace('.jpg', '.png'))
    mask_pil.save(output_mask_path)

    print(f"Saved mask to {output_mask_path}")