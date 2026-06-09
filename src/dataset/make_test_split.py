import os
import glob
import shutil
import random


def create_test_split():
    # Пути
    val_images_dir = r"C:\Practicum\data\processed\images\val"
    val_labels_dir = r"C:\Practicum\data\processed\labels\val"

    test_images_dir = r"C:\Practicum\data\processed\images\test"
    test_labels_dir = r"C:\Practicum\data\processed\labels\test"

    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    val_images = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    print(f"[INFO] Всего картинок в val: {len(val_images)}")

    split_size = int(len(val_images) * 0.5)
    test_images = random.sample(val_images, split_size)

    print(f"[INFO] Переносим {split_size} картинок в test...")

    for img_path in test_images:
        base_name = os.path.basename(img_path)
        lbl_name = base_name.replace('.jpg', '.txt')

        lbl_path = os.path.join(val_labels_dir, lbl_name)

        shutil.move(img_path, os.path.join(test_images_dir, base_name))

        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(test_labels_dir, lbl_name))

    print("[SUCCESS] Разбиение завершено.")


if __name__ == '__main__':
    create_test_split()