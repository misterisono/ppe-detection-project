import os
import yaml
from ultralytics import YOLO, RTDETR


def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'default.yaml'))
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_paths():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_yaml = os.path.join(project_dir, 'configs', 'dataset.yaml')
    results_dir = os.path.join(project_dir, 'results', 'logs')
    return data_yaml, results_dir


def train_model(model_name):
    cfg = load_config()
    data_yaml, results_dir = get_paths()
    seed = cfg.get('seed', 42)

    if model_name not in cfg:
        print(f"[ERROR] Настройки для {model_name} не найдены в default.yaml!")
        return

    m_cfg = cfg[model_name]

    print(f"[INFO] Запуск обучения {model_name} (Seed: {seed})...")

    if model_name == 'yolov8n_baseline':
        model = YOLO("yolov8n.pt")
        model.train(data=data_yaml, epochs=m_cfg['epochs'], batch=m_cfg['batch_size'], imgsz=m_cfg['imgsz'],
                    project=results_dir, name="baseline_yolov8n", device=0, seed=seed, workers=4)

    elif model_name == 'yolov8m_main':
        model = YOLO("yolov8m.pt")
        model.train(data=data_yaml, epochs=m_cfg['epochs'], batch=m_cfg['batch_size'], imgsz=m_cfg['imgsz'],
                    project=results_dir, name="yolov8m_balanced", device=0, seed=seed, workers=4)

    elif model_name == 'yolov5s':
        model = YOLO("yolov5s.pt")
        model.train(data=data_yaml, epochs=m_cfg['epochs'], batch=m_cfg['batch_size'], imgsz=m_cfg['imgsz'],
                    project=results_dir, name="yolov5s_balanced", device=0, seed=seed, workers=4)

    elif model_name == 'detr':
        model = RTDETR("rtdetr-l.pt")
        model.train(data=data_yaml, epochs=m_cfg['epochs'], batch=m_cfg['batch_size'], imgsz=m_cfg['imgsz'],
                    project=results_dir, name="rtdetr_balanced", device=0, seed=seed, workers=4)