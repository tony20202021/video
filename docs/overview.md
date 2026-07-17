# Общее описание системы

Система реального времени для анализа видеопотоков с 4 IP-камер iCSee (8MP, Wi-Fi, RTSP). Выполняет поэтапное распознавание людей:
- первичное обнаружение,
- классификацию по группам
- и индивидуальную идентификацию жителей.

Работает на CPU без GPU. Результаты фиксируются в базе данных и доступны через Telegram-бот.

---

## Архитектура системы

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Камера 1    │  │  Камера 2    │  │  Камера 3    │  │  Камера 4    │
│ (та же сеть) │  │ (роутер 2)  │  │ (роутер 3)  │  │ (роутер 4)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       └────────────────VPN / port-forward──────────────────┘
                                  │ RTSP (субпоток)
                                  ▼
               ┌──────────────────────────────────────────┐
               │              Backend (FastAPI)            │
               │                                          │
               │  ┌────────────────────────────────────┐  │
               │  │  Stream Manager                    │  │
               │  │  - asyncio + OpenCV VideoCapture   │  │
               │  │  - 1 поток на камеру               │  │
               │  │  - анализ 1–5 кадров/сек           │  │
               │  └──────────────┬─────────────────────┘  │
               │                 │ кадр                    │
               │                 ▼                         │
               │  ┌────────────────────────────────────┐  │
               │  │  ML Orchestrator                   │  │
               │  │  - вызовы ML Service по REST       │  │
               │  │  - логика пайплайна                │  │
               │  └──────────────┬─────────────────────┘  │
               │                 │                         │
               └─────────────────┼─────────────────────────┘
                    ┌────────────┴──────────────┐
                    │ REST API                  │ REST API
                    ▼                           ▼
          ┌──────────────────┐       ┌─────────────────────┐
          │   ML Service     │       │      MongoDB         │
          │ (ONNX Runtime)   │       │  + /data/images/     │
          └──────────────────┘       └─────────────────────┘
                                               ▲
                                               │ REST API
                                               ▼
                                  ┌────────────────────────┐
                                  │   Telegram Bot         │
                                  │   (aiogram 3.0)        │
                                  └────────────────────────┘
```

---

## Архитектура (целевая)

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Telegram Bot   │  │   Web App       │  │  Android App    │
│ (aiogram 3.0)   │  │ (FastAPI+HTML)  │  │ (запланировано) │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         └──────────────────┬─┘                   │
                            │ HTTP REST            │
                            ▼                      │
              ┌─────────────────────────┐         │
              │   Backend (FastAPI)     │◄────────┘
              │   Сервис логики         │
              │   /events /persons      │
              │   /cameras /status      │
              └──────────┬──────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────────┐
    │ MongoDB  │  │  .output/│  │ Camera Agent │
    │ (сервер) │  │  (диск)  │  │ (скрипты 4,5)│
    └──────────┘  └──────────┘  └──────────────┘
```

## Текущее состояние

| Компонент | Статус | Примечание |
|-----------|--------|------------|
| Скрипты камер (1–5) | **Реализовано** | Поиск, зондирование, верификация, motion detection, YOLOv8n |
| YOLOv8n ONNX | **Реализовано** | `.models/detect/yolov8n.onnx` (~13 MB) |
| Telegram Bot | **Реализовано** | aiogram 3.0, обращается к backend API |
| Backend (FastAPI) | **В разработке** | `/status` работает, `/events` `/persons` `/cameras` — стабы |
| Web App | **В разработке** | Базовый HTML-интерфейс |
| Android App | **Запланировано** | |
| ML классификация группы | **Запланировано** | MobileNetV3-Small, модель не обучена |
| ML идентификация жителя | **Запланировано** | MobileFaceNet, модель не обучена |
| MongoDB (схема) | **Запланировано** | Схема описана в `database.md`, код не написан |

---

## Нефункциональные требования

| Параметр                | Значение                                      |
|-------------------------|-----------------------------------------------|
| Камеры                  | 4 (iCSee 8MP, RTSP)                           |
| GPU                     | Нет, только CPU                               |
| Задержка детекции       | < 50 мс на кадр (YOLOv8n ONNX на CPU)        |
| Частота анализа         | 1–5 FPS на камеру (настраиваемо)              |
| Формат моделей          | ONNX Runtime                                  |
| Хранение изображений    | Файловая система (`/data/images/`)            |
| API аутентификация      | API-ключ (header `X-API-Key`)                 |
| Язык                    | Python 3.11+                                  |
| Контейнеризация         | Docker + Docker Compose                       |

---

## Версия

Единый источник версии — файл `VERSION` в корне репозитория (формат `MAJOR.MINOR.PATCH`).

- В коде: `from common.version import get_version` (используется в FastAPI-приложениях: backend, transfer).
- Обновление: `python scripts/version.py bump` (минор +1), `--patch`, `--major`, либо `set X.Y.Z`.
- Правило проекта: на каждое изменение — `bump` (минор +1); мажор — только когда явно сказано.

```bash
python scripts/version.py            # показать
python scripts/version.py bump       # 0.1.0 → 0.2.0
python scripts/version.py bump --major   # 0.2.0 → 1.0.0
```

## Структура репозитория

```
video/
  src/
    common/
      utils/
        atomic.py         — атомарные записи: imwrite() и copy() через .tmp → rename
        cam_urls.py       — сбор CAM_<stem>_URL из окружения
        cam_crop.py       — обрезка кадра по долям x,y,w,h
        classes.py        — GROUP_CLASSES — единый список классов классификатора
        adaptive_rate.py  — AdaptiveRateLimiter, регулировка YOLO по CPU-нагрузке
        log_setup.py      — setup_logging() с единым форматом %(asctime)s  %(levelname)-8s
        motion_utils.py   — общие утилиты motion-цикла
        person_detector.py — обёртка YOLOv8n ONNX
    frontend/             — Telegram Bot (aiogram 3.0)
      bot.py              — диспетчер и роутер
      config.py           — конфигурация из .env
      api_client.py       — HTTP-клиент к backend API
      handlers/           — /start, /events, /persons, /cameras, /unclassified, /admin
      keyboards/          — inline-кнопки (пагинация, фильтры)
  scripts/
    cameras/
      1_scan_cameras.py       — поиск камер в сети по портам
      2_probe_channels.py     — перебор channel×stream (XM/iCSee)
      3_verify_cameras.py     — проверка RTSP + сохранение кадра
    pipeline/
      1_motion_diff.py        — motion diff, baseline/heartbeat кадры
      2_yolo_boxes_files.py   — YOLOv8n инференс → кропы людей (watch-режим)
      3_classify_groups.py    — Модель 1: классификация кропов по группам
      4_identify_residents.py — Модель 2: идентификация жителей
      5_track_direction.py    — определение направления движения
    train/
      dataset_groups.py       — управление датасетом: build/apply/check/add/status
      2_label_ui.py           — веб-разметчик кропов (Flask)
      3_train_groups.py       — обучение Модели 1
      5_train_residents.py    — обучение Модели 2
    transfer/
      server.py               — FastAPI-сервер приёма файлов (атомарная запись)
      client.py               — CLI-клиент (send / watch / health / runs)
  sh/                         — скрипты запуска (Windows .ps1 / Linux .sh по ОС)
    pipeline/                 — Linux-сервер, кроме 1_motion_diff (Windows-камера)
      1_motion_diff.ps1       — захват дифф-кадров (Windows)
      2_yolo_boxes_files.sh   — YOLO-детекция кропов (watch + delete-after)
      3_classify_groups.sh    — классификация групп (Модель 1)
      4_identify_residents.sh — идентификация жителей (Модель 2)
      5_track_direction.sh    — направление движения
    train/                    — Linux-сервер
      1_1_dataset_groups_check.sh    — watch: 2_yolo_boxes_files → new/ (дубли против датасета)
      1_2_dataset_groups_check_new.sh — watch: дедупликация внутри new/
      1_3_dataset_groups_inference_labels.sh — watch: обновление labels.json из inference/images/
      1_dataset_groups.sh     — ручное управление датасетом
      2_label_ui.sh           — запуск веб-разметчика
      3_dataset_build.sh      — сборка датасета v2 (мастер-скрипт)
      3_dataset_from_inference.sh — стратегии 1,2,3,5,8: из inference/images/YYYYMMDD/
      3_dataset_from_prev.sh  — стратегии 4,7: прогон модели на предыдущем датасете
      4_train_groups.sh       — обучение Модели 1
      5_train_residents.sh    — обучение Модели 2
    transfer/
      1_start_server.sh       — запуск Transfer-сервера (Linux-сервер)
      2_send.ps1              — запуск Transfer-клиента watch (Windows-камера)
    system/
      watchdog.ps1            — Windows: авто-перезапуск процессов пайплайна
                                (motion diff + transfer client); -Register для Task Scheduler
  tests/
    test_atomic.py              — атомарные записи (imwrite / copy)
    test_adaptive_rate.py       — AdaptiveRateLimiter
    test_transfer.py            — Transfer server API
    test_motion_utils.py        — утилиты motion-цикла
    test_access.py              — IP whitelist
    test_smoke.py               — загрузка YOLO, формат выхода
    test_ml_pipeline.py         — ML-пайплайн classify → identify
    (+ другие)
  .output/                — результаты скриптов (в .gitignore)
    pipeline/
      1_motion_diff/
        images/YYYYMMDD/  — дифф-кадры по датам (MSK)
        meta/YYYYMMDD/    — CSV, JSON, графики, run.log
      2_yolo_boxes_files/
        images/YYYYMMDD/  — кропы людей
        annotated/YYYYMMDD/ — аннотированные кадры с боксами
        meta/YYYYMMDD/    — crops.csv, run_params.json, run.log
      transfer/
        diff/YYYYMMDD/<cam>/  — принятые дифф-кадры
        meta/YYYYMMDD/<cam>/  — sidecar JSON
  .data/
    groups/
      v1/
        dataset/          — датасет Модели 1: 1_resident/, 2_delivery/, 3_utilities/, 4_guest/, skip/, unknown/
        new/              — очередь кропов для разметки (→ dataset/ после apply)
        inference/        — результаты 3_classify_groups: images/YYYYMMDD/<class>/, meta/YYYYMMDD/
  .models/
    detect/yolov8n.onnx   — детекция людей (~13 MB)
    classify/v1_1.onnx    — Модель 1 (активная: датасет v1, прогон 1, val_acc 79%)
    classify/v1_1.json    — манифест модели
    classify/backbone.pt  — backbone для инициализации Модели 2
    identify/v*.onnx      — Модель 2 (идентификатор жителей)
  docs/
    overview.md           — этот файл
    setup.md              — Python, conda, окружение, requirements.txt
    cameras.md            — камеры, сетевой доступ, скрипты
    ml.md                 — ML пайплайн, датасет, обучение
    transfer.md           — передача файлов клиент → сервер
    testing.md            — тестирование
  .env                    — RTSP URL, секреты, пути к моделям (в .gitignore)
```
