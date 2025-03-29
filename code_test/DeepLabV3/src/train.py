import torch
import torch.nn as nn
import os
import argparse

from datasets import get_images, get_dataset, get_data_loaders
from engine import train, validate
from model import prepare_model
from config import ALL_CLASSES, LABEL_COLORS_LIST
from utils import save_model, SaveBestModel, save_plots

parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs',
    default=20,
    help='number of epochs to train for',
    type=int
)
parser.add_argument(
    '--lr',
    default=0.0001,
    help='learning rate for optimizer',
    type=float
)
parser.add_argument(
    '--batch',
    default=4,
    help='batch size for data loader',
    type=int
)
args = parser.parse_args()



if __name__ == '__main__':
    # Create a directory with the model name for outputs.

    out_dir = os.path.join('.\outputs')
    out_dir_valid_preds = os.path.join('..', 'outputs', 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss() 

    # 取得當前程式所在的目錄 (train.py所在的目錄)
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 根據程式所在目錄計算 'input' 目錄的相對路徑
    root_path = os.path.join(base_path,'..', 'input')

    train_images, train_masks, valid_images, valid_masks = get_images(root_path)

    classes_to_train = ALL_CLASSES

    train_dataset, valid_dataset = get_dataset(
        train_images, 
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        classes_to_train,
        LABEL_COLORS_LIST,
        img_size=224
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, valid_dataset, batch_size=args.batch
    )

    batch_size = args.batch
    while batch_size > 0:
        try:
            train_dataloader, valid_dataloader = get_data_loaders(
                train_dataset, valid_dataset, batch_size=batch_size,
            )
            print(f"Using batch size: {batch_size}")
            break  # 成功初始化 dataloader 則跳出迴圈
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM detected! Reducing batch size from {batch_size} to {batch_size // 2}")
                batch_size //= 2  # 減少 batch size 再試一次
                torch.cuda.empty_cache()  # 釋放記憶體
            else:
                raise e  # 其他錯誤則直接拋出

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()

    EPOCHS = args.epochs
    train_loss, train_pix_acc = [], []
    valid_loss, valid_pix_acc = [], []

    progress_file = os.path.join('..', 'outputs', "train_progress.txt")

    # 訓練開始前，先清空進度文件
    with open(progress_file, "w") as f:
        f.write("0\n")  # 初始 epoch 設為 0

    for epoch in range (EPOCHS):
        print(f"EPOCH: {epoch + 1}")


        train_epoch_loss, train_epoch_pixacc = train(
            model,
            train_dataset,
            train_dataloader,
            device,
            optimizer,
            criterion,
            classes_to_train
        )
        valid_epoch_loss, valid_epoch_pixacc = validate(
            model,
            valid_dataset,
            valid_dataloader,
            device,
            criterion,
            classes_to_train,
            LABEL_COLORS_LIST,
            epoch,
            ALL_CLASSES,
            save_dir=out_dir_valid_preds
        )

        with open(progress_file, "w") as f:
            f.write(str(epoch + 1) + "\n")  # 每次寫入最新的 epoch 數

        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc.cpu())
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc.cpu())

        save_best_model(
            valid_epoch_loss, epoch, model, out_dir
        )

        print(f"Train Epoch Loss: {train_epoch_loss:.4f}, Train Epoch PixAcc: {train_epoch_pixacc:.4f}")
        print(f"Valid Epoch Loss: {valid_epoch_loss:.4f}, Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}")

        # 釋放 GPU 記憶體，防止累積導致 OOM
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    save_model(EPOCHS, model, optimizer, criterion, out_dir)
    # Save the loss and accuracy plots.
    save_plots(
        train_pix_acc, valid_pix_acc, train_loss, valid_loss, out_dir
    )
    print('TRAINING COMPLETE')