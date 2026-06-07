import os
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline для детекции СИЗ")
    parser.add_argument('--model', type=str, required=True, help='Выберите модель (например: yolo)')
    args = parser.parse_args()

    if args.model.lower() == 'yolo':
        print("[INFO] Запуск детекции YOLO...")
        project_dir = os.path.abspath(os.path.dirname(__file__))

        model_path = os.path.join(project_dir, 'src', 'models', 'yolov8m_trained_balanced_30.pt')

        if not os.path.exists(model_path):
            print(f"[ERROR] Веса не найдены по пути: {model_path}")
            return

        model = YOLO(model_path)
        print("[SUCCESS] Модель загружена. Готово к работе!")
    else:
        print(f"[ERROR] Модель '{args.model}' не поддерживается.")


if __name__ == '__main__':
    main()