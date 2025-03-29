import argparse
from ultralytics import YOLO

def train_yolo(epochs, batch, workers):
    # 使用 Segmentation 版的 YOLOv11 預訓練模型
    model = YOLO("yolo11n-seg.pt")

    # 訓練 YOLO Segmentation 模型
    train_results = model.train(
        data="emv-seg.yaml",
        epochs=epochs,
        imgsz=512,
        batch=batch,
        device="0",
        workers=workers  # 使用主線程加載數據
    )

    # 驗證模型
    metrics = model.val(split="val", save_json=True)

    # 使用訓練好的最佳模型進行推論
    results = model('yolov11/dataset/emv/images/test')
    results[0].show()

    # 導出 ONNX 模型
    onnx_path = model.export(format="onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11 Segmentation Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="Number of data loading workers")

    args = parser.parse_args()
    print(f"Starting training with epochs={args.epochs}, batch={args.batch}, workers={args.workers}")
    train_yolo(args.epochs, args.batch, args.workers)