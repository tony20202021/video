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
| YOLOv8n ONNX | **Реализовано** | `models/yolov8n.onnx` (~13 MB) |
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

## Структура репозитория

```
video/
  src/
    common/
      utils/
        cam_urls.py       — сбор CAM_<stem>_URL из окружения
        cam_crop.py       — обрезка кадра по долям x,y,w,h
        motion_utils.py   — общие утилиты motion-цикла
        person_detector.py — обёртка YOLOv8n ONNX
    frontend/             — Telegram Bot (aiogram 3.0)
      bot.py              — диспетчер и роутер
      config.py           — конфигурация из .env
      api_client.py       — HTTP-клиент к backend API
      handlers/           — /start, /events, /persons, /cameras, /unclassified, /admin
      keyboards/          — inline-кнопки (пагинация, фильтры)
  models/
    yolov8n.onnx          — детекция людей (~13 MB, скачивается отдельно)
  tests/
  scripts/
    cameras/
      1_scan_cameras.py   — поиск камер в сети по портам
      2_probe_channels.py — перебор channel×stream (XM/iCSee)
      3_verify_cameras.py — проверка RTSP + сохранение кадра
      4_motion_diff_low.py — frame diff по субпотоку, сохранение LOW-кадра
      5_diff_yolo_boxes_low.py  — motion diff → YOLOv8n → кадры с людьми
    setup_models.py       — скачать и конвертировать ONNX-модели
    bot/run_bot.py        — запуск Telegram Bot
  sh/                     — shell скрипты (start_bot.sh, psiphon_proxy.ps1)
  .output/                — результаты скриптов (в .gitignore)
    1_scan_cameras/       — отчёты 1_scan_cameras.py
    2_probe_channels/     — кадры и отчёты 2_probe_channels.py
    3_cam_verify/         — кадры и отчёты 3_verify_cameras.py
    4_motion_diff_low/    — baseline и кадры движения 4_motion_diff_low.py
    5_diff_yolo_boxes_low/ — кадры с людьми (bbox) 5_diff_yolo_boxes_low.py
  docs/
    overview.md           — этот файл
    setup.md              — Python, conda, окружение, requirements.txt
    cameras.md            — камеры, сетевой доступ, скрипты
    ml.md                 — ML пайплайн и логика обработки видео
    database.md           — схема MongoDB
    services.md           — API сервисов, конфиг, экспорт
  .env                    — RTSP URL и секреты (в .gitignore)
  .config.md              — инфраструктура: серверы, камеры, сети
```
