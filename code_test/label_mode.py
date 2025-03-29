import io
import math 
import streamlit as st   
from PIL import Image
import tempfile
import cv2
import numpy as np
import os
import zipfile
import pydicom
import shutil
import json
import torch
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut
from streamlit_drawable_canvas import st_canvas
import sys
import os

sys.path.append(os.path.abspath("C:/Users/Lin LiTung/Desktop/sam2"))

from sam2.build_sam import build_sam2_video_predictor

def process_image(image):
    rgb_image = np.array(image.convert("RGB"))
    return rgb_image
total_frames = 0
def process_video(video_file_path, frame_interval=None, num_frames_to_extract=None):
    processed_frames = []
    video = cv2.VideoCapture(video_file_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_interval:
        frame_indices = range(0, total_frames, frame_interval)
    elif num_frames_to_extract:
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
    else:
        return []

    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        if i in frame_indices:
            rgb_frame = frame[..., ::-1]
            processed_frames.append(rgb_frame)

    video.release()
    return processed_frames

def save_image(image, output_dir, file_name):
    image_dir = os.path.join(output_dir, "image") 
    os.makedirs(image_dir, exist_ok=True) 
    
    image_path = os.path.join(image_dir, file_name)
    Image.fromarray(image).convert("RGB").save(image_path, format='JPEG', quality=95)
    
    return image_path

def save_frame_images(frames, video_dir):
    frames_dir = os.path.join(video_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    image_paths = []

    for i, frame in enumerate(frames):
        image_path = os.path.join(frames_dir, f"{i+1}.jpg")
        Image.fromarray(frame).convert("RGB").save(image_path, format='JPEG', quality=95)
        image_paths.append(image_path)
    st.session_state.frames_dir = frames_dir
    zip_file_path = os.path.join(tempfile.gettempdir(), f"{os.path.basename(video_dir)}.zip")
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for image_path in image_paths:
            zipf.write(image_path, os.path.basename(image_path)) 

    return zip_file_path, image_paths  

def get_app_variables():
    uploaded_file = st.session_state.get('uploaded_file', None)
    left = st.session_state.get('left', 0)
    top = st.session_state.get('top', 0)
    predictor = st.session_state.get('predictor', None)
    inference_state = st.session_state.get('inference_state', False)
    frame_names = st.session_state.get('frame_names', [])

    return uploaded_file, left, top, predictor, inference_state, frame_names

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_polygon(polygon, ax, color='red'):
    """在 Matplotlib 圖上顯示多邊形"""
    from matplotlib.patches import Polygon

    poly_patch = Polygon(polygon, edgecolor=color, facecolor='none', linewidth=2)
    ax.add_patch(poly_patch)
    ax.scatter(*zip(*polygon), color=color, marker='o', s=10)  # 標出頂點

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    
    return segmentation

def create_mask(image_size, polygon):
    mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)  # 确保是二维数组
    if polygon:
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask

st.title('EasyMedVision:Label Mode')
video_dir = "."
file_extensions_to_delete = [".jpg", ".jpeg", ".JPG", ".JPEG"]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

if 'page' not in st.session_state:
    st.session_state.page = 'upload'

if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

if st.session_state.page == 'upload':
    analysis_name = st.text_input("Enter analysis name")
    base_path = os.getcwd()
    output_dir = os.path.join(base_path, analysis_name)

    os.makedirs(output_dir, exist_ok=True)

    st.session_state.base_path = base_path
    st.session_state.analysis_name = analysis_name

    st.sidebar.header("Parameter Settings")

    selection_method = st.sidebar.radio(
        "Select one parameter to set",
        ['Frame interval', 'Number of frames to extract'],
        help="- **Frame interval**：Set the interval for extracting frames\n"
             "- **Number of frames to extract**：Set the total number of frames to extract from the video"
    )

    if selection_method == 'Frame interval':
        frame_interval = st.sidebar.number_input(
            "Frame interval", min_value=1, max_value=100, value=10,
            help="Set the interval for extracting frames"
        )
        num_frames = None
    else:
        num_frames = st.sidebar.number_input(
            "Number of frames to extract", min_value=1, max_value=100, value=10,
            help="Set the total number of frames to extract from the video"
        )
        frame_interval = None

    uploaded_file = st.file_uploader("Upload an image, video, or zip file", type=["jpg", "jpeg", "png", "mp4", "mov", "zip"])

    if uploaded_file is not None:
        file_type = uploaded_file.type
        st.write(f"File type: {file_type}")

        if file_type in ["image/jpeg", "image/png", "image/jpg"]:
            image = Image.open(uploaded_file)
            processed_image = process_image(image)  

            processed_pil_image = Image.fromarray(processed_image)
            image_dir = os.path.join(output_dir, "image")
            os.makedirs(image_dir, exist_ok=True)
            processed_image_path = os.path.join(image_dir, "1.jpg")

            processed_pil_image.save(processed_image_path, format='JPEG', quality=95)
            st.image(processed_pil_image, caption="Processed Image", use_container_width=False, width=400)

            if st.button("Confirm"):
                st.session_state.page = 'process'
                st.rerun()

        elif file_type in ["video/mp4", "video/quicktime"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
                temp_video_file.write(uploaded_file.read())
                temp_video_path = temp_video_file.name

            if frame_interval is not None:
                processed_frames = process_video(temp_video_path, frame_interval, None)
            elif num_frames is not None:
                processed_frames = process_video(temp_video_path, None, num_frames)
            else:
                st.error("Please set either 'Frame interval' or 'Number of frames to extract'.")
                processed_frames = []

            if processed_frames:
                zip_file_path, _ = save_frame_images(processed_frames, output_dir)

                st.write("Processed Video Frames:")
                cols = st.columns(4)
                for i, frame in enumerate(processed_frames):
                    with cols[i % 4]:
                        st.image(frame, caption=f'Frame {i+1}', use_container_width=True)

                if st.button("Confirm"):
                    st.session_state.page = 'process'
                    st.rerun()

        elif file_type in ["application/x-zip-compressed", "application/zip"]:
            zip_extract_dir = os.path.join(output_dir, "frames")
            os.makedirs(zip_extract_dir, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
                zip_ref.extractall(zip_extract_dir)

            extracted_images = []
            extracted_videos = []

            for root, _, files in os.walk(zip_extract_dir):
                for file in sorted(files):  
                    file_path = os.path.join(root, file)
                    file_ext = file.lower().split('.')[-1]

                    if file_ext in ["jpg", "jpeg", "png"]:
                        extracted_images.append(file_path)
                    elif file_ext in ["mp4", "mov"]:
                        extracted_videos.append(file_path)

            if extracted_images:
                st.write("Extracted Images Preview:")
                cols = st.columns(4)
                for i, img_path in enumerate(extracted_images[:12]): 
                    with cols[i % 4]:
                        image = Image.open(img_path)
                        st.image(image, caption=f'Image {i+1}', use_container_width=True)

            if extracted_videos:
                st.write("Extracted Videos Preview:")
                for i, video_path in enumerate(extracted_videos[:2]): 
                    st.video(video_path)

            if st.button("Confirm"):
                st.session_state.page = 'process'
                st.session_state.uploaded_zip_path = zip_extract_dir
                st.rerun()

        # 保存基本的檔案資訊
        st.session_state.uploaded_file = uploaded_file
        st.session_state.analysis_name = analysis_name
        st.session_state.frame_interval = frame_interval
        st.session_state.num_frames = num_frames

elif st.session_state.page == 'process':

    if hasattr(st.session_state, 'uploaded_zip_path'):
        zip_dir = st.session_state.uploaded_zip_path
        image_files = sorted(
            [f for f in os.listdir(zip_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

        if len(image_files) == 0:
            st.error("No images found in the ZIP file.")
        else:
            st.write(f"Found {len(image_files)} images in ZIP.")
            
            # 直接選擇 ZIP 裡的第一張圖片
            selected_image_path = os.path.join(zip_dir, image_files[0])
            image = Image.open(selected_image_path)
            processed_image = process_image(image)

            st.session_state.processed_image = processed_image
            st.session_state.image_size = image.size
            st.image(processed_image, caption=f"Processed Image ({image_files[0]})", use_container_width=True)

            analysis_region = st.radio("Choose region selection method", ['Click to select', 'Drag to select', 'Click and drag', 'Manual Label'])
            if st.button('Start Selection'):
                if analysis_region == 'Click to select':
                    st.session_state.page = 'select_click'
                elif analysis_region == 'Drag to select':
                    st.session_state.page = 'select_drag'
                elif analysis_region == 'Click and drag':
                    st.session_state.page = 'select_both'
                elif analysis_region == 'Manual Label':
                    st.session_state.page = 'select_manual'
                st.rerun()

    elif hasattr(st.session_state, 'uploaded_file'):

        if st.session_state.uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            image = Image.open(st.session_state.uploaded_file)
            processed_image = process_image(image)
            st.session_state.processed_image = processed_image
            image_size = image.size
            st.session_state.image_size = image_size

            st.image(processed_image, caption="Processed Image", use_container_width=True)

            analysis_region = st.radio("Choose region selection method", ['Click to select', 'Drag to select', 'Click and drag', 'Manual Label'])
            if st.button('Start Selection'):
                if analysis_region == 'Click to select':
                    st.session_state.page = 'select_click'
                elif analysis_region == 'Drag to select':
                    st.session_state.page = 'select_drag'
                elif analysis_region == 'Click and drag':
                    st.session_state.page = 'select_both'
                elif analysis_region == 'Manual Label':
                    st.session_state.page = 'select_manual'
                st.rerun()

        elif st.session_state.uploaded_file.type in ["video/mp4", "video/quicktime"]:
            video_file_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")
            with open(video_file_path, "wb") as f:
                f.write(st.session_state.uploaded_file.getbuffer())

            video = cv2.VideoCapture(video_file_path)
            success, frame = video.read()
            if success:
                first_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.session_state.first_frame = first_frame
                st.image(first_frame, caption="First Frame of Video", use_container_width=True)

            analysis_region = st.radio("Choose region selection method", ['Click to select', 'Drag to select', 'Click and drag', 'Manual Label'])
            if st.button('Start Selection'):
                if analysis_region == 'Click to select':
                    st.session_state.page = 'select_click'
                elif analysis_region == 'Drag to select':
                    st.session_state.page = 'select_drag'
                elif analysis_region == 'Click and drag':
                    st.session_state.page = 'select_both'
                elif analysis_region == 'Manual Label':
                    st.session_state.page = 'select_manual'
                st.rerun()

        else:
            st.write("Unsupported file type!")
    else:
        st.write("No uploaded file found!")


elif st.session_state.page == 'select_click':
    uploaded_file = st.session_state.uploaded_file 
    file_type = st.session_state.uploaded_file.type
    left = 0
    top = 0

    if "predictor" not in st.session_state:
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        st.session_state.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        
        if file_type in ["image/jpeg", "image/png", "image/jpg"]:
            video_path = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "image")
            st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_path)
        else:
            video_path = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames")
            st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_path)

    predictor = st.session_state.predictor
    inference_state = st.session_state.inference_state

    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        if "processed_image" in st.session_state:
            image = st.session_state.processed_image
            image_size = st.session_state.image_size
            
            scale_factor = math.ceil(image_size[0] / 700)
            w = image_size[0] / scale_factor  
            h = image_size[1] / scale_factor  

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_color="rgba(0, 255, 0, 1)",
                stroke_width=2,
                width=w,
                height=h,
                drawing_mode="circle",
                background_image=Image.fromarray(image),
                key="canvas",
            )

            if canvas_result.json_data:
                selected_points = canvas_result.json_data["objects"]
                for point in selected_points:
                    if point["type"] == "circle":
                        left = point["left"] * scale_factor
                        top = point["top"] * scale_factor
                        st.write("Selected Point Coordinates:", (left, top))

            if st.button("Strat analyze"):

                frame_names = [
                    p for p in os.listdir(video_dir)
                    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
                ]
                frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

                ann_frame_idx = 0
                ann_obj_id = 1
                points = np.array([[left, top]], dtype=np.float32)
                labels = np.array([1], np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )

                frame_idx = 0
                image_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "image")

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.imshow(Image.open(os.path.join(image_dir, '1.jpg')))
                ax.set_title(f"frame {ann_frame_idx}")
                show_points(points, labels, ax)
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])

                st.pyplot(fig)
                st.success(f"All results are saved")
                st.session_state.image_mask = (out_mask_logits[0] > 0.0).cpu().numpy()

            if "image_mask" in st.session_state:
                mask_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "mask")
                os.makedirs(mask_dir, exist_ok=True) 

                mask_path = os.path.join(mask_dir, "mask.png")

                if not os.path.exists(mask_path):
                    mask_img = Image.fromarray((st.session_state.image_mask.squeeze() * 255).astype('uint8'))
                    mask_img.save(mask_path)

            if st.button("Reset"):
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                inference_state = predictor.init_state(video_path=video_dir)
                predictor.reset_state(inference_state)      
    
    elif file_type in ["video/mp4", "video/quicktime"]:
    
        analysis_name = st.session_state.analysis_name  # 請確認這個變數是否正確
        frames_dir = os.path.join(video_dir, analysis_name, "frames")
        os.makedirs(frames_dir, exist_ok=True)  # 確保 frames 目錄存在
        
        frame_names = sorted(
            [p for p in os.listdir(frames_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
            key=lambda p: int(os.path.splitext(p)[0])
        )
        
        total_frames = len(frame_names)  # 影片總影格數

        if st.session_state.frame_idx < total_frames:
            frame_path = os.path.join(frames_dir, frame_names[st.session_state.frame_idx])
            image = Image.open(frame_path)
            st.session_state.first_frame = image

            image_size = image.size
            scale_factor = math.ceil(image_size[0] / 700)
            w, h = image_size[0] / scale_factor, image_size[1] / scale_factor

            if st.button("Next Frame"):
                if st.session_state.frame_idx < total_frames - 1:
                    st.session_state.frame_idx += 1  # 移動到下一幀
                    st.rerun()
                else:
                    st.warning("No more frames available.")


            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_color="rgba(0, 255, 0, 1)",
                stroke_width=2,
                width=w,
                height=h,
                drawing_mode="point",
                background_image=image,
                key="canvas",
            )

            if canvas_result.json_data:
                selected_points = canvas_result.json_data["objects"]
                for point in selected_points:
                    if point["type"] == "circle":
                        left = point["left"] * scale_factor
                        top = point["top"] * scale_factor
                        st.write("Selected Point Coordinates:", (left, top))

            if "analyze_done" not in st.session_state:
                st.session_state.analyze_done = False  # 初始化變數
            
            if st.button("Start analyze"):
                ann_frame_idx = st.session_state.frame_idx  # 記錄當前幀索引
                ann_obj_id = 1
                points = np.array([[left, top]], dtype=np.float32)
                labels = np.array([1], np.int32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.imshow(image)
                ax.set_title(f"frame {ann_frame_idx}")
                show_points(points, labels, ax)
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])

                st.pyplot(fig)
                st.session_state.predictor = predictor
                st.session_state.inference_state = inference_state
                st.session_state.analyze_done = True  # 設定分析完成標誌

            # 只有當分析完成後，才顯示 Confirm 按鈕
            if st.session_state.analyze_done:
                if st.button("Confirm"):
                    st.session_state.page = 'click_all'
                    st.rerun()

            if st.button("Reset"):
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                inference_state = predictor.init_state(video_path=uploaded_file)
                predictor.reset_state(inference_state)
        else:
            st.warning("No more frames available.")

elif st.session_state.page == 'select_drag':
    uploaded_file = st.session_state.uploaded_file 
    file_type = st.session_state.uploaded_file.type
    scaled_top_left_x = 0
    scaled_top_left_y = 0
    scaled_bottom_right_x = 0
    scaled_bottom_right_y = 0

    if "predictor" not in st.session_state:
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        st.session_state.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        
        if file_type in ["image/jpeg", "image/png", "image/jpg"]:
            video_path = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "image")
            st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_path)
        else:
            video_path = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames")
            st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_path)

    predictor = st.session_state.predictor
    inference_state = st.session_state.inference_state

    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        if "processed_image" in st.session_state:
            image = st.session_state.processed_image
            image_size = st.session_state.image_size
            a = math.ceil(st.session_state.image_size[0] / 700)
            w=st.session_state.image_size[0]/a  
            h=st.session_state.image_size[1]/a  

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_color="rgba(0, 255, 0, 1)",
                stroke_width=2,
                width=w,
                height=h,
                drawing_mode="rect",
                background_image=Image.fromarray(image),
                key="canvas",
            )

            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                for obj in objects:
                    if obj["type"] == "rect":
                        top_left_x = obj["left"]
                        top_left_y = obj["top"]
                        bottom_right_x = obj["left"] + obj["width"]
                        bottom_right_y = obj["top"] + obj["height"]

                        scaled_top_left_x = int(top_left_x * a)
                        scaled_top_left_y = int(top_left_y * a)
                        scaled_bottom_right_x = int(bottom_right_x * a)
                        scaled_bottom_right_y = int(bottom_right_y * a)

                        st.write(f"Rectangle top-left corner: ({scaled_top_left_x}, {scaled_top_left_y}), "
                                f"bottom-right corner: ({scaled_bottom_right_x}, {scaled_bottom_right_y})")
            
            if st.button("Strat analyze"):
                frame_names = [
                    p for p in os.listdir(video_dir)
                    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
                ]
                frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

                ann_frame_idx = 0
                ann_obj_id = 4
                box = np.array([scaled_top_left_x, scaled_top_left_y, scaled_bottom_right_x, scaled_bottom_right_y], dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=box,
                )
    
                frame_idx = 0
                image_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "image")

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.imshow(Image.open(os.path.join(image_dir, '1.jpg')))
                ax.set_title(f"frame {ann_frame_idx}")
                show_box(box, plt.gca())
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

                st.pyplot(fig)
                st.success(f"All results are saved")
                st.session_state.image_mask = (out_mask_logits[0] > 0.0).cpu().numpy()

            if "image_mask" in st.session_state:
                mask_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "mask")
                os.makedirs(mask_dir, exist_ok=True)

                mask_path = os.path.join(mask_dir, "mask.png")

                if not os.path.exists(mask_path):
                    mask_img = Image.fromarray((st.session_state.image_mask.squeeze() * 255).astype('uint8'))
                    mask_img.save(mask_path)

            if st.button("Reset"):
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                inference_state = predictor.init_state(video_path=video_dir)
                predictor.reset_state(inference_state)


    elif file_type in ["video/mp4", "video/quicktime"]:
        frames_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames")

        # 確保 frames_dir 存在
        if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
            st.error(f"Frames directory not found: {frames_dir}")
        else:
            frame_names = sorted(
                [p for p in os.listdir(frames_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
                key=lambda p: int(os.path.splitext(p)[0])
            )

            total_frames = len(frame_names)

            # 初始化影格索引
            if "frame_idx" not in st.session_state:
                st.session_state.frame_idx = 0

            # 檢查當前影格是否存在
            if st.session_state.frame_idx < total_frames:
                frame_path = os.path.join(frames_dir, frame_names[st.session_state.frame_idx])
                image = Image.open(frame_path)
                st.session_state.first_frame = image

                image_size = image.size
                scale_factor = max(image_size[0] / 700, 1)
                w = int(image_size[0] / scale_factor)
                h = int(image_size[1] / scale_factor)

                a = scale_factor

                if st.button("Next Frame"):
                    if st.session_state.frame_idx < total_frames - 1:
                        st.session_state.frame_idx += 1  # 移動到下一幀
                        st.rerun()
                    else:
                        st.warning("No more frames available.")

                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_color="rgba(0, 255, 0, 1)",
                    stroke_width=2,
                    width=w,
                    height=h,
                    drawing_mode="rect",
                    background_image=image,
                    key="canvas",
                )

                if canvas_result.json_data:
                    objects = canvas_result.json_data["objects"]
                    for obj in objects:
                        if obj["type"] == "rect":
                            top_left_x = obj["left"]
                            top_left_y = obj["top"]
                            bottom_right_x = obj["left"] + obj["width"]
                            bottom_right_y = obj["top"] + obj["height"]

                            scaled_top_left_x = int(top_left_x * a)
                            scaled_top_left_y = int(top_left_y * a)
                            scaled_bottom_right_x = int(bottom_right_x * a)
                            scaled_bottom_right_y = int(bottom_right_y * a)

                            st.write(f"Rectangle top-left corner: ({scaled_top_left_x}, {scaled_top_left_y}), "
                                    f"bottom-right corner: ({scaled_bottom_right_x}, {scaled_bottom_right_y})")

                if "analyze_done" not in st.session_state:
                    st.session_state.analyze_done = False  # 初始化變數

                # **開始分析按鈕**
                if st.button("Start analyze"):
                    ann_frame_idx = st.session_state.frame_idx  # 取得當前影格索引
                    ann_obj_id = 4  # 設定物件 ID

                    box = np.array([scaled_top_left_x, scaled_top_left_y, scaled_bottom_right_x, scaled_bottom_right_y], dtype=np.float32)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=box,
                    )

                    # 顯示分析結果
                    fig, ax = plt.subplots(figsize=(9, 6))
                    ax.imshow(image)
                    ax.set_title(f"frame {ann_frame_idx}")
                    show_box(box, ax)
                    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])

                    st.pyplot(fig)
                    st.session_state.predictor = predictor
                    st.session_state.inference_state = inference_state
                    st.session_state.analyze_done = True  # 設定分析完成標誌
                    st.session_state.total_frames = total_frames 

                # 只有當分析完成後，才顯示 Confirm 按鈕
                if st.session_state.analyze_done:
                    if st.button("Confirm"):
                        st.session_state.page = 'click_all'
                        st.rerun()

                # **重置按鈕**
                if st.button("Reset"):
                    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                    inference_state = predictor.init_state(video_path=video_dir)
                    predictor.reset_state(inference_state)

            else:
                st.warning("No more frames available.")

elif st.session_state.page == 'select_both':
    uploaded_file = st.session_state.uploaded_file 
    file_type = st.session_state.uploaded_file.type
    scaled_top_left_x = 0
    scaled_top_left_y = 0
    scaled_bottom_right_x = 0
    scaled_bottom_right_y = 0
    scaled_cx = 0
    scaled_cx = 0

    if "predictor" not in st.session_state:
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        st.session_state.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        
        if file_type in ["image/jpeg", "image/png", "image/jpg"]:
            video_path = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "image")
            st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_path)
        else:
            video_path = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames")
            st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_path)

    predictor = st.session_state.predictor
    inference_state = st.session_state.inference_state

    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        if "processed_image" in st.session_state:
            image = st.session_state.processed_image
            image_size = st.session_state.image_size
            a = math.ceil(st.session_state.image_size[0] / 700)
            w=st.session_state.image_size[0]/a  
            h=st.session_state.image_size[1]/a  

            selected_tool = st.radio(
                "",
                options=["Rectangle", "Circle"],
                index=0,
                horizontal=True,
            )

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_color="rgba(0, 255, 0, 1)",
                stroke_width=2,
                width=w,
                height=h,
                drawing_mode="rect" if selected_tool == "Rectangle" else "circle",
                background_image=Image.fromarray(image),
                key="canvas",
            )

            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                for obj in objects:
                    if obj["type"] == "rect":
                        top_left_x = obj["left"]
                        top_left_y = obj["top"]
                        bottom_right_x = obj["left"] + obj["width"]
                        bottom_right_y = obj["top"] + obj["height"]

                        scaled_top_left_x = int(top_left_x * a)
                        scaled_top_left_y = int(top_left_y * a)
                        scaled_bottom_right_x = int(bottom_right_x * a)
                        scaled_bottom_right_y = int(bottom_right_y * a)

                        st.write(f"Rectangle top-left corner: ({scaled_top_left_x}, {scaled_top_left_y}), "
                                f"bottom-right corner: ({scaled_bottom_right_x}, {scaled_bottom_right_y})")
                    
                    elif obj["type"] == "circle":            
                        cx = obj["left"] + obj["radius"]
                        cy = obj["top"] + obj["radius"]

                        radius = obj["radius"]
                        scaled_cx = int(cx * a)
                        scaled_cy = int(cy * a)
                        scaled_radius = int(radius * a)

                        st.write(f"Circle: Center ({scaled_cx}, {scaled_cy})")
            
            if st.button("Strat analyze"):
                frame_names = [
                    p for p in os.listdir(video_dir)
                    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
                ]
                frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

                ann_frame_idx = 0  
                ann_obj_id = 4  

                points = np.array([[scaled_cx, scaled_cy]], dtype=np.float32)
                labels = np.array([1], np.int32)
            
                box = np.array([scaled_top_left_x, scaled_top_left_y, scaled_bottom_right_x, scaled_bottom_right_y], dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                    box=box,
                )
    
                frame_idx = 0
                image_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "image")

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.imshow(Image.open(os.path.join(image_dir, '1.jpg')))
                ax.set_title(f"frame {ann_frame_idx}")
                show_box(box, plt.gca())
                show_points(points, labels, plt.gca())
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

                st.pyplot(fig)
                st.success(f"All results are saved")
                st.session_state.image_mask = (out_mask_logits[0] > 0.0).cpu().numpy()

            if "image_mask" in st.session_state:
                mask_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "mask")
                os.makedirs(mask_dir, exist_ok=True)  

                mask_path = os.path.join(mask_dir, "mask.png")

                if not os.path.exists(mask_path):
                    mask_img = Image.fromarray((st.session_state.image_mask.squeeze() * 255).astype('uint8'))
                    mask_img.save(mask_path)

            if st.button("Reset"):
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                inference_state = predictor.init_state(video_path=video_dir)
                predictor.reset_state(inference_state)


    elif file_type in ["video/mp4", "video/quicktime"]:
        frames_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames")

        if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
            st.error(f"Frames directory not found: {frames_dir}")
        else:
            frame_names = sorted(
                [p for p in os.listdir(frames_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
                key=lambda p: int(os.path.splitext(p)[0])
            )

            total_frames = len(frame_names)

            if "frame_idx" not in st.session_state:
                st.session_state.frame_idx = 0

            if st.session_state.frame_idx < total_frames:
                frame_path = os.path.join(frames_dir, frame_names[st.session_state.frame_idx])
                image = Image.open(frame_path)
                st.session_state.first_frame = image

                image_size = image.size
                scale_factor = max(image_size[0] / 700, 1)
                w = int(image_size[0] / scale_factor)
                h = int(image_size[1] / scale_factor)
                if st.button("Next Frame"):
                    if st.session_state.frame_idx < total_frames - 1:
                        st.session_state.frame_idx += 1  # 移動到下一幀
                        st.rerun()
                    else:
                        st.warning("No more frames available.")
                
                selected_tool = st.radio(
                    "",
                    options=["Rectangle", "Circle"],
                    index=0,
                    horizontal=True,
                )

                a = scale_factor
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_color="rgba(0, 255, 0, 1)",
                    stroke_width=2,
                    width=w,
                    height=h,
                    drawing_mode="rect" if selected_tool == "Rectangle" else "point",
                    background_image=image,
                    key="canvas",
                )

                if canvas_result.json_data:
                    objects = canvas_result.json_data["objects"]
                    for obj in objects:
                        if obj["type"] == "rect":
                            top_left_x = obj["left"]
                            top_left_y = obj["top"]
                            bottom_right_x = obj["left"] + obj["width"]
                            bottom_right_y = obj["top"] + obj["height"]

                            scaled_top_left_x = int(top_left_x * a)
                            scaled_top_left_y = int(top_left_y * a)
                            scaled_bottom_right_x = int(bottom_right_x * a)
                            scaled_bottom_right_y = int(bottom_right_y * a)

                            st.write(f"Rectangle top-left corner: ({scaled_top_left_x}, {scaled_top_left_y}), "
                                    f"bottom-right corner: ({scaled_bottom_right_x}, {scaled_bottom_right_y})")

                        elif obj["type"] == "circle":            
                            cx = obj["left"] + obj["radius"]
                            cy = obj["top"] + obj["radius"]

                            radius = obj["radius"]
                            scaled_cx = int(cx * a)
                            scaled_cy = int(cy * a)
                            scaled_radius = int(radius * a)

                            st.write(f"Circle: Center ({scaled_cx}, {scaled_cy})")

                if "analyze_done" not in st.session_state:
                    st.session_state.analyze_done = False  # 初始化變數           

                if st.button("Start analyze"):
                    ann_frame_idx = st.session_state.frame_idx
                    ann_obj_id = 4

                    box = np.array([scaled_top_left_x, scaled_top_left_y, scaled_bottom_right_x, scaled_bottom_right_y], dtype=np.float32)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=box,
                    )

                    fig, ax = plt.subplots(figsize=(9, 6))
                    ax.imshow(image)
                    ax.set_title(f"frame {ann_frame_idx}")
                    show_box(box, ax)
                    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])

                    st.pyplot(fig)
                    st.session_state.predictor = predictor
                    st.session_state.inference_state = inference_state
                    st.session_state.analyze_done = True  # 設定分析完成標誌

                # 只有當分析完成後，才顯示 Confirm 按鈕
                if st.session_state.analyze_done:
                    if st.button("Confirm"):
                        st.session_state.page = 'click_all'
                        st.rerun()

                if st.button("Reset"):
                    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                    inference_state = predictor.init_state(video_path=video_dir)
                    predictor.reset_state(inference_state)

            else:
                st.warning("No more frames available.")

elif st.session_state.page == 'select_manual':
    uploaded_file = st.session_state.uploaded_file 
    file_type = st.session_state.uploaded_file.type
    left = 0
    top = 0

    if "predictor" not in st.session_state:
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        st.session_state.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        
        if file_type in ["image/jpeg", "image/png", "image/jpg"]:
            video_path = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "image")
            st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_path)

        elif file_type == "application/zip":
            frames_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames")
            st.session_state.inference_state = st.session_state.predictor.init_state(video_path=frames_dir)
        else:
            video_path = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames")
            st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_path)

    predictor = st.session_state.predictor
    inference_state = st.session_state.inference_state

    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        if "processed_image" in st.session_state:
            image = st.session_state.processed_image
            image_size = st.session_state.image_size
            a = math.ceil(st.session_state.image_size[0] / 700)
            w = st.session_state.image_size[0] / a  
            h = st.session_state.image_size[1] / a  

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_color="rgba(0, 255, 0, 1)",
                stroke_width=2,
                width=w,
                height=h,
                drawing_mode="polygon",
                background_image=Image.fromarray(image),
                key="canvas",
            )

            h, w = int(h), int(w)

            if st.button("Generate Mask"):
                if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
                    objects = canvas_result.json_data["objects"]
                    scaled_polygon = []

                    for obj in objects:
                        if obj["type"] == "path" and "path" in obj:
                            path_data = obj["path"]
                            points = []

                            for item in path_data:
                                if isinstance(item, list) and len(item) == 3:
                                    x, y = int(item[1]), int(item[2])
                                    points.append((x, y))

                            if len(points) >= 3:
                                if points[-1] != points[0]: 
                                    points.append(points[0])
                                scaled_polygon = points

                    if scaled_polygon:
                        mask = create_mask((h, w), scaled_polygon) 
                        st.session_state.image_mask = mask  

                        st.subheader("Generated Mask")
                        st.image(mask, caption="Mask Preview", use_container_width=True)
                    else:
                        st.warning("No polygon found. Please draw a polygon before generating the mask.")

            if "image_mask" in st.session_state:
                if st.button("Save Mask"):
                    mask = st.session_state.image_mask  

                    mask_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "mask")
                    os.makedirs(mask_dir, exist_ok=True)
                    mask_path = os.path.join(mask_dir, "mask.png")
                    mask = Image.fromarray(mask)
                    mask.save(mask_path)

                    st.success(f"Mask saved to {mask_path}")


    elif file_type in ["video/mp4", "video/quicktime"]:
        frames_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames")

        if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
            st.error(f"Frames directory not found: {frames_dir}")
        else:
            frame_names = sorted(
                [p for p in os.listdir(frames_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
                key=lambda p: int(os.path.splitext(p)[0])
            )
            total_frames = len(frame_names)

            if "frame_idx" not in st.session_state:
                st.session_state.frame_idx = 0
            if "masks" not in st.session_state:
                st.session_state.masks = {}

            frame_idx = st.session_state.frame_idx
            if frame_idx < total_frames:
                frame_path = os.path.join(frames_dir, frame_names[frame_idx])
                image = Image.open(frame_path)
                image_size = image.size
                scale_factor = max(image_size[0] / 700, 1)
                w, h = int(image_size[0] / scale_factor), int(image_size[1] / scale_factor)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Next Frame"):
                        if frame_idx < total_frames - 1:
                            st.session_state.frame_idx += 1
                            st.rerun()
                        else:
                            st.warning("No more frames available.")

                with col2:
                    if st.button("Reset"):
                        st.session_state.frame_idx = 0
                        st.session_state.masks = {}
                        st.rerun()

                
                
                st.subheader(f"Frame {frame_idx+1}/{total_frames}")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_color="rgba(0, 255, 0, 1)",
                    stroke_width=2,
                    width=w,
                    height=h,
                    drawing_mode="polygon",
                    background_image=image,  
                    key=f"canvas_{frame_idx}",
                )

                polygon_points = []
                if canvas_result.json_data:
                    objects = canvas_result.json_data.get("objects", [])
                    for obj in objects:
                        if obj.get("type") == "path" and "path" in obj:
                            polygon_points = obj["path"]
                            break

                mask = None
                if polygon_points:

                    scaled_polygon = [
                        [int(pt[1] * scale_factor), int(pt[2] * scale_factor)]
                        for pt in polygon_points if isinstance(pt, list) and len(pt) >= 3
                    ]
                    mask = create_mask((image_size[1], image_size[0]), scaled_polygon)
                    st.session_state.masks[frame_idx] = mask

                    mask_resized = Image.fromarray(mask).resize(image_size, Image.NEAREST)
                    st.image(mask_resized, caption=f"Mask for Frame {frame_idx+1}", use_container_width =True)

                if st.button("Show All Masks"):
                    st.session_state.show_masks = True
                    st.rerun() 

            if st.session_state.get("show_masks", False) and st.session_state.masks:
                st.subheader("All Generated Masks")
                mask_images = list(st.session_state.masks.values())
                num_masks = len(mask_images)
                num_cols = 4
                num_rows = -(-num_masks // num_cols)  

                for i in range(num_rows):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        idx = i * num_cols + j
                        if idx < num_masks:
                            with cols[j]:
                                original_size = Image.open(os.path.join(frames_dir, frame_names[idx])).size
                                resized_mask = Image.fromarray(mask_images[idx]).resize(original_size)
                                st.image(resized_mask, caption=f"Frame {idx+1}", use_container_width =False)

                if st.button("Save All Masks"):
                    mask_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "mask")
                    os.makedirs(mask_dir, exist_ok=True)

                    saved_paths = []

                    for idx, mask in st.session_state.masks.items():
                        mask_path = os.path.join(mask_dir, f"{idx+1}.png")
                        mask_image = Image.fromarray(mask)
                        mask_image.save(mask_path)
                        saved_paths.append(mask_path)

                    st.success(f"All masks saved to: {mask_dir}")

    elif file_type in ["application/x-zip-compressed", "application/zip"]:

        frames_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames")
        
        image_files = sorted(
            [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        total_images = len(image_files)

        if total_images == 0:
            st.error("No images found in the extracted folder.")
        else:
            if "image_idx" not in st.session_state:
                st.session_state.image_idx = 0
            if "zip_masks" not in st.session_state:
                st.session_state.zip_masks = {}

            image_idx = st.session_state.image_idx
            if image_idx < total_images:
                image_path = os.path.join(frames_dir, image_files[image_idx])
                image = Image.open(image_path)
                image_size = image.size
                scale_factor = max(image_size[0] / 700, 1)
                w, h = int(image_size[0] / scale_factor), int(image_size[1] / scale_factor)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Next Image"):
                        if image_idx < total_images - 1:
                            st.session_state.image_idx += 1
                            st.rerun()
                        else:
                            st.warning("No more images available.")

                with col2:
                    if st.button("Reset"):
                        st.session_state.image_idx = 0
                        st.session_state.zip_masks = {}
                        st.rerun()

                st.subheader(f"Image {image_idx + 1}/{total_images}")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.3)",
                    stroke_color="rgba(0, 255, 0, 1)",
                    stroke_width=2,
                    width=w,
                    height=h,
                    drawing_mode="polygon",
                    background_image=image,
                    key=f"canvas_zip_{image_idx}",
                )

                polygon_points = []
                if canvas_result.json_data:
                    objects = canvas_result.json_data.get("objects", [])
                    for obj in objects:
                        if obj.get("type") == "path" and "path" in obj:
                            polygon_points = obj["path"]
                            break

                mask = None
                if polygon_points:
                    scaled_polygon = [
                        [int(pt[1] * scale_factor), int(pt[2] * scale_factor)]
                        for pt in polygon_points if isinstance(pt, list) and len(pt) >= 3
                    ]
                    mask = create_mask((image_size[1], image_size[0]), scaled_polygon)
                    st.session_state.zip_masks[image_idx] = mask

                    mask_resized = Image.fromarray(mask).resize(image_size, Image.NEAREST)
                    st.image(mask_resized, caption=f"Mask for Image {image_idx + 1}", use_container_width=True)

                if st.button("Show All Masks"):
                    st.session_state.show_zip_masks = True
                    st.rerun()

            if st.session_state.get("show_zip_masks", False) and st.session_state.zip_masks:
                st.subheader("All Generated Masks")
                mask_images = list(st.session_state.zip_masks.values())
                num_masks = len(mask_images)
                num_cols = 4
                num_rows = -(-num_masks // num_cols) 

                for i in range(num_rows):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        idx = i * num_cols + j
                        if idx < num_masks:
                            with cols[j]:
                                original_size = Image.open(os.path.join(frames_dir, image_files[idx])).size
                                resized_mask = Image.fromarray(mask_images[idx]).resize(original_size)
                                st.image(resized_mask, caption=f"Image {idx + 1}", use_container_width=False)

                if st.button("Save All Masks"):
                    mask_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "masks")
                    os.makedirs(mask_dir, exist_ok=True)

                    saved_paths = []
                    for idx, mask in st.session_state.zip_masks.items():
                        mask_path = os.path.join(mask_dir, f"{idx + 1}.png")
                        mask_image = Image.fromarray(mask)
                        mask_image.save(mask_path)
                        saved_paths.append(mask_path)

                    st.success(f"All masks saved to: {mask_dir}")

elif st.session_state.page == 'click_all':  
    predictor = st.session_state.predictor
    inference_state = st.session_state.inference_state

    if "video_results" not in st.session_state:
        st.session_state.video_results = []
        st.session_state.video_segments = {}

        progress_text = st.empty()  

        total_frames = inference_state.get("num_frames", 0)  # 影片總影格數
        skipped_frames = st.session_state.frame_idx  # 已跳過的影格數
        remaining_frames = total_frames - skipped_frames  # 剩餘影格數

        if remaining_frames <= 0:
            st.error("No frames left to process.")
        else:
            processed_count = 0  

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                processed_count += 1
                progress_text.text(f"Processing frame {processed_count}/{remaining_frames}") 

                st.session_state.video_results.append((out_frame_idx, out_obj_ids, out_mask_logits))

            progress_text.text("Processing completed!")  


    if "selected_masks" not in st.session_state:
        st.session_state.selected_masks = {}

    if not st.session_state.video_segments:
        st.session_state.video_segments = {
            out_frame_idx: {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            for out_frame_idx, out_obj_ids, out_mask_logits in st.session_state.video_results
        }

    frames_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames")
    mask_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "masks")
    frames_select_dir = os.path.join(st.session_state.base_path, st.session_state.analysis_name, "frames_select")

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(frames_select_dir, exist_ok=True)

    if st.button("Select All & Download"):
        st.session_state.selected_masks = {f: True for f in st.session_state.video_segments}

        selected_mask_paths = []
        selected_frame_paths = []
        new_index = 1  

        for out_frame_idx, segment_data in sorted(st.session_state.video_segments.items()):
            original_image_path = os.path.join(frames_dir, f"{out_frame_idx+1}.jpg")
            new_image_path = os.path.join(frames_select_dir, f"{new_index}.jpg")
            shutil.copy(original_image_path, new_image_path)
            selected_frame_paths.append(new_image_path)

            for out_obj_id, out_mask in segment_data.items():
                mask_img = Image.fromarray((out_mask.squeeze() * 255).astype('uint8'))  
                mask_path = os.path.join(mask_dir, f"{new_index}.png")
                mask_img.save(mask_path)
                selected_mask_paths.append(mask_path)

            new_index += 1  

        st.success(f"Saved {len(selected_mask_paths)} masks and {len(selected_frame_paths)} images.")
        st.stop()  

    with st.form("mask_selection_form"):
        cols = st.columns(4)
        updated_masks = {}

        for i, (out_frame_idx, segment_data) in enumerate(sorted(st.session_state.video_segments.items())):
            with cols[i % 4]:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.set_title(f"Frame {out_frame_idx}")

                image_path = os.path.join(frames_dir, f'{out_frame_idx+1}.jpg')
                ax.imshow(Image.open(image_path))

                for out_obj_id, out_mask in segment_data.items():
                    show_mask(out_mask, ax, obj_id=out_obj_id)
                ax.axis("off")
                st.pyplot(fig)

                selected = st.checkbox(f"Select Frame {out_frame_idx}", 
                                       value=st.session_state.selected_masks.get(out_frame_idx, False))
                updated_masks[out_frame_idx] = selected

        save_button = st.form_submit_button("Save Selected")

        if save_button:
            st.session_state.selected_masks.update(updated_masks)

    if st.session_state.selected_masks:
        selected_mask_paths = []
        selected_frame_paths = []
        new_index = 1  

        for out_frame_idx, segment_data in sorted(st.session_state.video_segments.items()):
            if st.session_state.selected_masks.get(out_frame_idx):  
                original_image_path = os.path.join(frames_dir, f"{out_frame_idx+1}.jpg")
                new_image_path = os.path.join(frames_select_dir, f"{new_index}.jpg")
                shutil.copy(original_image_path, new_image_path)
                selected_frame_paths.append(new_image_path)

                for out_obj_id, out_mask in segment_data.items():
                    mask_img = Image.fromarray((out_mask.squeeze() * 255).astype('uint8'))  
                    mask_path = os.path.join(mask_dir, f"{new_index}.png")
                    mask_img.save(mask_path)
                    selected_mask_paths.append(mask_path)

                new_index += 1  

        st.success(f"Saved {len(selected_mask_paths)} masks and {len(selected_frame_paths)} images.")
    else:
        st.error("No masks selected for saving.")
