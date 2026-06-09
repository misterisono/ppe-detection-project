import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.dataset.custom_dataset import PPEDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_rcnn():
    print(f"[INFO] Запуск обучения Faster R-CNN. CUDA: {torch.cuda.is_available()}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_img_dir = r"C:\Practicum\data\processed\images\train"
    train_lbl_dir = r"C:\Practicum\data\processed\labels\train"

    dataset = PPEDataset(train_img_dir, train_lbl_dir)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)


    model = get_model(num_classes=4)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 15
    loss_history = []

    print("[INFO] Начинаем обучение")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, targets in progress_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())

        avg_loss = epoch_loss / len(data_loader)
        print(f"Эпоха {epoch + 1} завершена. Средний Loss: {avg_loss:.4f}")
        loss_history.append({"epoch": epoch + 1, "train_loss": avg_loss})

    save_dir = r"C:\Practicum\src\models"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'faster_rcnn_PPE.pth'))

    logs_dir = r"C:\Practicum\results\logs\faster_rcnn"
    os.makedirs(logs_dir, exist_ok=True)
    pd.DataFrame(loss_history).to_csv(os.path.join(logs_dir, "results.csv"), index=False)

    print("[SUCCESS] Модель обучена")


if __name__ == '__main__':
    train_rcnn()