import os
import cv2
import random
from ultralytics import YOLO


def quick_throwaway_check():
    # Находим папку скрипта и поднимаемся на уровень выше (в корень проекта)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Путь к модели
    model_path = os.path.join(project_root, "results", "logs", "yolov8m_balanced-3", "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"[-] Ошибка: Веса не найдены по пути:\n{model_path}")
        return

    print(f"[+] Загружаем модель...")
    model = YOLO(model_path)

    # Путь к картинкам
    img_dir = os.path.join(project_root, "data", "raw", "images", "test")

    if not os.path.exists(img_dir):
        print(f"[-] Ошибка: Папка с картинками не найдена:\n{img_dir}")
        return

    # 1. Собираем ВООБЩЕ ВСЕ картинки из папки
    all_images = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not all_images:
        print("[-] Ошибка: В папке нет картинок!")
        return

    # 2. Выбираем 5 случайных (или меньше, если в папке их всего пара штук)
    num_to_test = min(5, len(all_images))
    images_to_test = random.sample(all_images, num_to_test)

    print(f"[+] Выбрано случайных картинок: {len(images_to_test)}")
    print("[+] Жми любую кнопку на клавиатуре (находясь в активном окне), чтобы переключить на следующую.")

    results = model.predict(
        source=images_to_test,
        stream=True,
        save=False,
        save_txt=False,
        conf=0.25
    )

    for i, result in enumerate(results):
        annotated_frame = result.plot()

        window_name = f"YOLOv8 Test - Random Image {i + 1}"
        cv2.imshow(window_name, annotated_frame)

        # Ждем нажатия
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("[SUCCESS] Тест окончен. Окна закрыты, диски чисты.")


if __name__ == "__main__":
    quick_throwaway_check()