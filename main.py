import argparse
import sys
import src.training.train_yolo as train_yolo
import src.training.train_faster_rcnn as rcnn
import src.evaluation.evaluate as evaluate

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline для детекции СИЗ")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Доступные модели: yolov8n_baseline, yolov8m_main, yolov5s, detr, faster_rcnn'
    )
    args = parser.parse_args()
    model_name = args.model.lower()

    ultralytics_models = ['yolov8n_baseline', 'yolov8m_main', 'yolov5s', 'detr']

    if model_name in ultralytics_models:
        train_yolo.train_model(model_name)
    elif model_name == 'faster_rcnn':
        rcnn.train_rcnn()
    else:
        print(f"[ERROR] Модель '{args.model}' не поддерживается.")
        sys.exit(1)

    print(f"[INFO] Обучение завершено. Генерация metrics.json для {model_name}...")
    evaluate.generate_metrics_json(model_name)

if __name__ == '__main__':
    main()