import os
import shutil
from pathlib import Path


def verify_and_copy_dataset():
    project_root = Path(__file__).resolve().parent.parent

    base_raw_dir = project_root / "data" / "raw"
    base_proc_dir = project_root / "data" / "processed"

    print(f"[LOG] Корень проекта определен как: {project_root}")
    print(f"[LOG] Ищем исходные данные в: {base_raw_dir}")

    splits = ['train', 'val', 'test']

    classes_file = base_raw_dir / "labels" / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"\n Найдено классов ({len(classes)}): {classes}")
    else:
        print(f"\n ОШИБКА: Файл не найден по пути: {classes_file}")
        return

    for split in splits:
        img_src_dir = base_raw_dir / "images" / split
        lbl_src_dir = base_raw_dir / "labels" / split

        img_dst_dir = base_proc_dir / "images" / split
        lbl_dst_dir = base_proc_dir / "labels" / split

        if not img_src_dir.exists() or not lbl_src_dir.exists():
            print(f"\n ОШИБКА: Не найдены папки для выборки '{split}'")
            print(f"   Ожидалось: {img_src_dir} и {lbl_src_dir}")
            continue

        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}
        raw_images = {f.stem: f.suffix for f in img_src_dir.iterdir() if f.suffix in image_extensions}
        raw_labels = {f.stem for f in lbl_src_dir.iterdir() if f.suffix == '.txt'}

        valid_stems = raw_images.keys() & raw_labels

        print(f"\nВыборка [{split}]:")
        print(f"  Картинок на диске: {len(raw_images)}, Файлов разметки: {len(raw_labels)}")
        print(f"  Успешно валидировано пар: {len(valid_stems)}")

        copied_count = 0
        for stem in valid_stems:
            img_ext = raw_images[stem]
            shutil.copy(img_src_dir / f"{stem}{img_ext}", img_dst_dir / f"{stem}{img_ext}")
            shutil.copy(lbl_src_dir / f"{stem}.txt", lbl_dst_dir / f"{stem}.txt")
            copied_count += 1

        print(f"  Перенесено в processed/: {copied_count} пар.")


if __name__ == "__main__":
    verify_and_copy_dataset()