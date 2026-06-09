import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class PPEDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms

        self.image_files = []
        for img_name in os.listdir(images_dir):
            if img_name.endswith('.jpg'):
                lbl_name = img_name.replace('.jpg', '.txt')
                if os.path.exists(os.path.join(labels_dir, lbl_name)):
                    self.image_files.append(img_name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        h, w, _ = image.shape

        lbl_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt'))
        boxes = []
        labels = []

        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])

                    xmin = (xc - bw / 2) * w
                    ymin = (yc - bh / 2) * h
                    xmax = (xc + bw / 2) * w
                    ymax = (yc + bh / 2) * h

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls_id + 1)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        image = torch.as_tensor(image).permute(2, 0, 1)

        return image, target


if __name__ == "__main__":
    train_img_dir = r"C:\Practicum\data\processed\images\train"
    train_lbl_dir = r"C:\Practicum\data\processed\labels\train"

    dataset = PPEDataset(train_img_dir, train_lbl_dir)
    img, target = dataset[0]

    print(f"Размер тензора картинки: {img.shape}")
    print(f"Тензор боксов:\n{target['boxes']}")
    print(f"Тензор меток классов:\n{target['labels']}")
    print("[SUCCESS] Кастомный Dataset на PyTorch работает!")