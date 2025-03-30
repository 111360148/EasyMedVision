import math 
import streamlit as st   
from PIL import Image
import tempfile
import cv2
import numpy as np
import os
import zipfile
import pydicom
import json
import torch
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut
from streamlit_drawable_canvas import st_canvas
import sys
import os

sys.path.append(os.path.abspath("C:/Users/Kaee/sam2"))

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

def process_dicom(dicom_file):
    dicom_data = pydicom.dcmread(dicom_file)
    image = dicom_data.pixel_array
    image = apply_voi_lut(image, dicom_data)
    image = (image / image.max() * 255).astype(np.uint8)
    if dicom_data.PhotometricInterpretation == "MONOCHROME1":
        image = cv2.bitwise_not(image)
    return image

def save_image(image, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, file_name)
    Image.fromarray(image).convert("RGB").save(image_path, format='JPEG', quality=95)
    return image_path

def save_frame_images(frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []

    for i, frame in enumerate(frames):
        image_path = os.path.join(output_dir, f"{i+1}.jpg")
        Image.fromarray(frame).convert("RGB").save(image_path, format='JPEG', quality=95)
        image_paths.append(image_path)

    zip_file_path = os.path.join(tempfile.gettempdir(), f"{os.path.basename(output_dir)}.zip")
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

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    
    return segmentation

def save_coco_json(output_path, video_segments, video_dir):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}]
    }
    annotation_id = 0

    image = st.session_state.first_frame
    w,h = image.size
    

    for frame_idx, segment_data in video_segments.items():
        file_name = f"{frame_idx + 1}.jpg"
        image_info = {
            "id": frame_idx + 1,
            "file_name": file_name,
            "width": w,
            "height": h
        }
        coco_data["images"].append(image_info)

        for obj_id, mask in segment_data.items():
            mask = np.array(mask)
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = np.squeeze(mask)

            if mask.shape != (h, w):
                print(f"錯誤：mask shape = {mask.shape}，仍然不對")
                continue

            segmentation = mask_to_polygon(mask)

            if not segmentation:
                print(f"Frame {frame_idx + 1} Object {obj_id}: segmentation 無效，未加入標註")
                continue

            annotation = {
                "id": int(annotation_id),
                "image_id": int(frame_idx + 1),
                "category_id": 1,
                "segmentation": segmentation,
                "area": int(np.sum(mask))
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=4)

st.title('EasyMedVision')
video_dir = "."
file_extensions_to_delete = [".jpg", ".jpeg", ".JPG", ".JPEG"]

if 'page' not in st.session_state:
    st.session_state.page = 'upload'

if st.session_state.page == 'upload':
    analysis_name = st.text_input("Enter analysis name")
    output_dir = os.getcwd()

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions_to_delete):
                file_path = os.path.join(root, file)
                os.remove(file_path)

    st.sidebar.header("Parameter Settings")
    selection_method = st.sidebar.radio("Select one parameter to set", ['Frame interval', 'Number of frames to extract'])

    if selection_method == 'Frame interval':
        frame_interval = st.sidebar.number_input("Frame interval", min_value=1, max_value=100, value=10)
        num_frames = None
    else:
        num_frames = st.sidebar.number_input("Number of frames to extract", min_value=1, max_value=100, value=10)
        frame_interval = None

    uploaded_file = st.file_uploader("Upload an image, video, or DICOM", type=["jpg", "jpeg", "png", "mp4", "mov", "dcm"])

    if uploaded_file is not None:
        file_type = uploaded_file.type
        st.write(f"File type: {file_type}")

        if file_type in ["image/jpeg", "image/png", "image/jpg"]:
            image = Image.open(uploaded_file)
            processed_image = process_image(image)

            processed_image_path = save_image(processed_image, output_dir, "1.jpg")
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

        elif file_type == "application/dicom":
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as temp_dicom_file:
                temp_dicom_file.write(uploaded_file.read())
                temp_dicom_path = temp_dicom_file.name
            
            processed_image = process_image(Image.open(temp_dicom_path))  # 假設您用相同的處理方式處理DICOM圖像

            processed_image_path = save_image(processed_image, output_dir, "dicom_image.jpg")
            
            st.image(processed_image, caption='Processed DICOM Image', use_container_width=True)

            if st.button("Confirm"):
                st.session_state.page = 'process'
                st.rerun()

        st.session_state.uploaded_file = uploaded_file
        st.session_state.analysis_name = analysis_name
        st.session_state.frame_interval = frame_interval
        st.session_state.num_frames = num_frames       

elif st.session_state.page == 'process':
    
    file_type = st.session_state.uploaded_file.type

    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        image = Image.open(st.session_state.uploaded_file)
        processed_image = process_image(image)
        st.session_state.processed_image = processed_image
        image_size = image.size
        st.session_state.image_size = image_size
        
        st.image(processed_image, caption="Processed Image", use_container_width=True)
        
        analysis_region = st.radio("Choose region selection method", ['Click to select', 'Drag to select', 'Click and drag'])
        if st.button('Start Selection'):
            if analysis_region == 'Click to select':
                st.session_state.page = 'select_click'
            elif analysis_region == 'Drag to select':
                st.session_state.page = 'select_drag'
            elif analysis_region == 'Click and drag':
                st.session_state.page = 'select_both'
            st.rerun()

    elif file_type in ["video/mp4", "video/quicktime"]:
        video_file_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")
        with open(video_file_path, "wb") as f:
            f.write(st.session_state.uploaded_file.getbuffer())
        
        video = cv2.VideoCapture(video_file_path)
        success, frame = video.read()
        if success:
            first_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.session_state.first_frame = first_frame
            st.image(first_frame, caption="First Frame of Video", use_container_width=True)

        analysis_region = st.radio("Choose region selection method", ['Click to select', 'Drag to select', 'Click and drag'])
        if st.button('Start Selection'):
            if analysis_region == 'Click to select':
                st.session_state.page = 'select_click'
            elif analysis_region == 'Drag to select':
                st.session_state.page = 'select_drag'
            elif analysis_region == 'Click and drag':
                st.session_state.page = 'select_both'
            st.rerun()

    elif file_type == "application/dicom":
        dicom_image = process_dicom(st.session_state.uploaded_file)
        st.session_state.dicom_image = dicom_image
        st.image(dicom_image, caption="Processed DICOM Image", use_container_width=True)

        analysis_region = st.radio("Choose region selection method", ['Click to select', 'Drag to select', 'Click and drag'])
        if st.button('Start Selection'):
            if analysis_region == 'Click to select':
                st.session_state.page = 'select_click'
            elif analysis_region == 'Drag to select':
                st.session_state.page = 'select_drag'
            elif analysis_region == 'Click and drag':
                st.session_state.page = 'select_both'
            st.rerun()

    else:
        st.write("Unsupported file type!")  

elif st.session_state.page == 'select_click':
    uploaded_file = st.session_state.uploaded_file 
    file_type = st.session_state.uploaded_file.type
    left = 0
    top = 0

    if "predictor" not in st.session_state:
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        st.session_state.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cpu"))
        st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_dir)

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

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.imshow(Image.open(os.path.join(video_dir, '1.jpg')))
                ax.set_title(f"frame {ann_frame_idx}")
                show_points(points, labels, ax)
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])

                st.pyplot(fig) 

            if st.button("Reset"):
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cpu"))
                inference_state = predictor.init_state(video_path=video_dir)
                predictor.reset_state(inference_state)

    elif file_type in ["video/mp4", "video/quicktime"]:
        
        if "first_frame" in st.session_state:
            image = st.session_state.first_frame
            image_size = image.size

            scale_factor = math.ceil(image_size[0] / 700)
            w = image_size[0] / scale_factor
            h = image_size[1] / scale_factor

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

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.imshow(Image.open(os.path.join(video_dir, '1.jpg')))
                ax.set_title(f"frame {ann_frame_idx}")
                show_points(points, labels, ax)
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), ax, obj_id=out_obj_ids[0])

                st.pyplot(fig)
                st.session_state.predictor = predictor
                st.session_state.inference_state = inference_state

                
            if st.button("Confirm"):
                st.session_state.page = 'click_all'
                st.rerun()

            if st.button("Reset"):
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cpu"))
                inference_state = predictor.init_state(video_path=uploaded_file)
                predictor.reset_state(inference_state)

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
        st.session_state.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cpu"))
        st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_dir)

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

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.imshow(Image.open(os.path.join(video_dir, '1.jpg')))
                ax.set_title(f"frame {ann_frame_idx}")
                show_box(box, plt.gca())
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

                st.pyplot(fig)

            if st.button("Reset"):
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cpu"))
                inference_state = predictor.init_state(video_path=video_dir)
                predictor.reset_state(inference_state)


    elif file_type in ["video/mp4", "video/quicktime"]:
        if "first_frame" in st.session_state:
            image = st.session_state.first_frame
            image_size = image.size

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            image_size = image.size
            scale_factor = max(image_size[0] / 700, 1)
            w = int(image_size[0] / scale_factor)
            h = int(image_size[1] / scale_factor)

            a = scale_factor
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

                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

                # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
                box = np.array([scaled_top_left_x, scaled_top_left_y, scaled_bottom_right_x, scaled_bottom_right_y], dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=box,
                )
    
                frame_idx = 0

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.imshow(Image.open(os.path.join(video_dir, '1.jpg')))
                ax.set_title(f"frame {ann_frame_idx}")
                show_box(box, plt.gca())
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

                st.pyplot(fig)
                st.session_state.predictor = predictor
                st.session_state.inference_state = inference_state

                
            if st.button("Confirm"):
                st.session_state.page = 'click_all'
                st.rerun()


            if st.button("Reset"):
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cpu"))
                inference_state = predictor.init_state(video_path=video_dir)
                predictor.reset_state(inference_state)

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
        st.session_state.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cpu"))
        st.session_state.inference_state = st.session_state.predictor.init_state(video_path=video_dir)

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

                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

                points = np.array([[scaled_cx, scaled_cy]], dtype=np.float32)
                labels = np.array([1], np.int32)
                # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
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

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.imshow(Image.open(os.path.join(video_dir, '1.jpg')))
                ax.set_title(f"frame {ann_frame_idx}")
                show_box(box, plt.gca())
                show_points(points, labels, plt.gca())
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

                st.pyplot(fig)

            if st.button("Reset"):
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cpu"))
                inference_state = predictor.init_state(video_path=video_dir)
                predictor.reset_state(inference_state)


    elif file_type in ["video/mp4", "video/quicktime"]:
        if "first_frame" in st.session_state:
            image = st.session_state.first_frame
            image_size = image.size

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            
            image_size = image.size
            scale_factor = max(image_size[0] / 700, 1)
            w = int(image_size[0] / scale_factor)
            h = int(image_size[1] / scale_factor)

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

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.imshow(Image.open(os.path.join(video_dir, '1.jpg')))
                ax.set_title(f"frame {ann_frame_idx}")
                show_box(box, plt.gca())
                show_points(points, labels, plt.gca())
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

                st.pyplot(fig)
                st.session_state.predictor = predictor
                st.session_state.inference_state = inference_state

                
            if st.button("Confirm"):
                st.session_state.page = 'click_all'
                st.rerun()

            if st.button("Reset"):
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=torch.device("cpu"))
                inference_state = predictor.init_state(video_path=video_dir)
                predictor.reset_state(inference_state)

elif st.session_state.page == 'click_all':

    predictor = st.session_state.predictor
    inference_state = st.session_state.inference_state
    if "video_results" not in st.session_state:
        st.session_state.video_results = list(predictor.propagate_in_video(inference_state))
        st.session_state.video_segments = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in st.session_state.video_results:
        st.session_state.video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    cols = st.columns(4)
    for i, (out_frame_idx, segment_data) in enumerate(sorted(st.session_state.video_segments.items())):
        with cols[i % 4]:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set_title(f"Frame {out_frame_idx}")
            ax.imshow(Image.open(os.path.join(video_dir, f'{out_frame_idx+1}.jpg')))
            for out_obj_id, out_mask in segment_data.items():
                show_mask(out_mask, ax, obj_id=out_obj_id)
            ax.axis("off")
            st.pyplot(fig)

    st.success("所有影像已處理完成！")

    if st.button("Save COCO JSON"):
        output_json_path = os.path.join(video_dir, "annotations.json")
        save_coco_json(output_json_path, st.session_state.video_segments, video_dir)
        st.success("已下載 COCO JSON")
