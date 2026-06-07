import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm


def balance_and_reduce_dataset():
    # ================= УКАЖИ СВОИ ПУТИ ЗДЕСЬ =================
    raw_images_dir = r"C:\Practicum\data\raw\images\train"
    raw_labels_dir = r"C:\Practicum\data\raw\labels\train"

    # Куда сохраняем новый крутой датасет
    processed_images_dir = r"C:\Practicum\data\processed\images\train"
    processed_labels_dir = r"C:\Practicum\data\processed\labels\train"

    TARGET_PER_CLASS = 2500  # Сколько примерно объектов каждого класса хотим
    # =========================================================

    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(processed_labels_dir, exist_ok=True)

    print("[INFO] Анализируем исходный датасет...")

    # Словарь: класс -> список txt файлов, где он встречается
    class_to_files = defaultdict(list)
    all_label_files = [f for f in os.listdir(raw_labels_dir) if f.endswith('.txt')]

    for label_file in tqdm(all_label_files, desc="Чтение разметки"):
        filepath = os.path.join(raw_labels_dir, label_file)
        with open(filepath, 'r') as f:
            lines = f.readlines()

        classes_in_file = set()
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                classes_in_file.add(class_id)

        for c in classes_in_file:
            class_to_files[c].append(label_file)

    print("\n[INFO] Статистика до балансировки (в скольких файлах есть класс):")
    for c, files in class_to_files.items():
        print(f"Класс {c}: {len(files)} картинок")

    # Собираем уникальные файлы для копирования
    selected_files = set()
    for c, files in class_to_files.items():
        random.shuffle(files)
        # Берем нужное количество (или все, если их меньше TARGET_PER_CLASS)
        selected = files[:TARGET_PER_CLASS]
        selected_files.update(selected)

    print(f"\n[INFO] Отобрано уникальных файлов для нового датасета: {len(selected_files)}")
    print("[INFO] Копируем файлы в data/processed/ ...")

    for label_file in tqdm(selected_files, desc="Копирование"):
        # Имена файлов
        image_file = label_file.replace('.txt', '.jpg')  # Убедись, что у тебя .jpg, а не .png

        src_label = os.path.join(raw_labels_dir, label_file)
        dst_label = os.path.join(processed_labels_dir, label_file)

        src_image = os.path.join(raw_images_dir, image_file)
        dst_image = os.path.join(processed_images_dir, image_file)

        # Копируем только если есть и картинка, и разметка
        if os.path.exists(src_image) and os.path.exists(src_label):
            shutil.copy(src_image, dst_image)
            shutil.copy(src_label, dst_label)

    print("\n[SUCCESS] Новый сбалансированный датасет готов!")
    print(f"Путь: {processed_images_dir}")


if __name__ == "__main__":
    balance_and_reduce_dataset()