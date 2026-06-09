import os
import glob
import random
import cv2
from ultralytics import YOLO


def get_model_path(model_choice, project_dir):
    paths = {
        "1": os.path.join(project_dir, 'results', 'logs', 'baseline_yolov8n', 'weights', 'best.pt'),
        "2": os.path.join(project_dir, 'src', 'models', 'yolov8m_trained_balanced_30.pt'),
        "3": os.path.join(project_dir, 'results', 'logs', 'yolov5s_balanced', 'weights', 'best.pt'),
        "4": os.path.join(project_dir, 'results', 'logs', 'rtdetr_balanced', 'weights', 'best.pt')
    }
    if model_choice == "2" and not os.path.exists(paths["2"]):
        paths["2"] = os.path.join(project_dir, 'results', 'logs', 'yolov8m_balanced-3', 'weights', 'best.pt')

    return paths.get(model_choice)


def run_cascade_logic(img_path, person_model, ppe_model):
    img = cv2.imread(img_path)
    person_results = person_model.predict(img_path, classes=[0], conf=0.25, verbose=False)[0]
    ppe_results = ppe_model.predict(img_path, conf=0.5, verbose=False)[0]

    persons = person_results.boxes.data.tolist()
    ppes = ppe_results.boxes.data.tolist()

    for p_box in persons:
        px1, py1, px2, py2, p_conf, p_cls = map(int, p_box[:6])
        has_helmet, has_vest = False, False

        for e_box in ppes:
            ex1, ey1, ex2, ey2 = map(int, e_box[:4])
            e_cls = int(e_box[5])
            cx, cy = (ex1 + ex2) // 2, (ey1 + ey2) // 2

            if px1 <= cx <= px2 and py1 <= cy <= py2:
                if e_cls == 0: has_helmet = True
                if e_cls == 1: has_vest = True

        if has_helmet and has_vest:
            color, text = (0, 255, 0), "SAFE"
        elif has_helmet or has_vest:
            color, text = (0, 165, 255), "WARNING"
        else:
            color, text = (0, 0, 255), "DANGER"

        cv2.rectangle(img, (px1, py1), (px2, py2), color, 3)
        cv2.putText(img, text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img


def main():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    test_images_dir = os.path.join(project_dir, 'data', 'processed', 'images', 'test')
    test_images = glob.glob(os.path.join(test_images_dir, "*.jpg"))

    if not test_images:
        test_images_dir = os.path.join(project_dir, 'data', 'processed', 'images', 'val')
        test_images = glob.glob(os.path.join(test_images_dir, "*.jpg"))

    print("\n" + "=" * 50)
    print(" 🛠 ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ МОДЕЛЕЙ (INFERENCE)")
    print("=" * 50)
    print("1. YOLOv8n (Baseline)")
    print("2. YOLOv8m (Main)")
    print("3. YOLOv5s")
    print("4. RT-DETR (Transformer)")
    print("5. CASCADE")
    print("0. Выход")

    choice = input("\nВыберите номер модели (1-5): ").strip()

    if choice == '0': return

    if choice == '5':
        print("[INFO] Загрузка каскадного пайплайна...")
        person_model = YOLO("yolov8m.pt")
        ppe_path = get_model_path("2", project_dir)
        if not ppe_path or not os.path.exists(ppe_path):
            print(f"[ERROR] Веса не найдены: {ppe_path}")
            return
        model = YOLO(ppe_path)
        is_cascade = True
    elif choice in ["1", "2", "3", "4"]:
        model_path = get_model_path(choice, project_dir)
        if not model_path or not os.path.exists(model_path):
            print(f"[ERROR] Веса не найдены: {model_path}")
            return
        print(f"[INFO] Загрузка модели из: {model_path}...")
        model = YOLO(model_path)
        is_cascade = False
    else:
        print("[ERROR] Неверный выбор.")
        return

    print("\n[УПРАВЛЕНИЕ]")
    print(" - [ENTER] (или Пробел) -> Следующее фото")
    print(" - [ESC] / [Q] / [КРЕСТИК] -> Выход")

    window_name = "Inference (Press ENTER for next, ESC to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    while True:
        img_path = random.choice(test_images)

        if is_cascade:
            annotated_frame = run_cascade_logic(img_path, person_model, model)
        else:
            results = model.predict(img_path, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()

        cv2.imshow(window_name, annotated_frame)

        while True:
            key = cv2.waitKey(50) & 0xFF

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[INFO] Работа завершена.")
                cv2.destroyAllWindows()
                return

            if key in [13, 32]:
                break
            elif key in [27, ord('q')]:
                print("\n[INFO] Работа завершена.")
                cv2.destroyAllWindows()
                return


if __name__ == '__main__':
    main()