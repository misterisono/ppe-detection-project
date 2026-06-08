# Система детекции СИЗ на строительной площадке (PPE Detection)

Проект полного цикла в области компьютерного зрения для автоматического обнаружения средств индивидуальной защиты (каски, жилеты). Разработан в рамках проектного практикума.

## Структура проекта
- `configs/` — конфигурационные файлы экспериментов (YAML).
- `data/` — скрипты и данные (raw/processed). Добавлено в `.gitignore`.
- `notebooks/` — Jupyter ноутбуки для EDA и тестов.
- `results/` — графики, логи обучения и веса моделей.
- `src/` — исходный код (dataset, models, training, evaluation).

## Установка и запуск

1. Клонирование репозитория:
```bash
git clone https://github.com/misterisono/ppe-detection-project.git
cd ppe-detection-project
pip install -r requirements.txt
python main.py --model yolo