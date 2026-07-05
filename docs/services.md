# Сервисы

## Backend (FastAPI)

> **Статус: запланировано.** Код не написан. Telegram Bot обращается к `BACKEND_URL` из `.env`, но без запущенного backend работать не будет.

**Технологии:** Python 3.11+, FastAPI, OpenCV, asyncio

**Ответственности:**
- Захват субпотока RTSP с каждой камеры (`cv2.VideoCapture` в отдельном потоке)
- Семплирование кадров (1–5 FPS, настраиваемо)
- Оркестрация ML-пайплайна (async HTTP вызовы в ML Service)
- Сохранение событий в MongoDB и кадров на диск
- REST API для Telegram Bot

**Эндпоинты:**
```
GET   /events                  — список событий (фильтр: камера, класс, дата, person_id)
GET   /events/{id}             — детали события + URL изображения
GET   /events/{id}/image       — бинарник изображения
GET   /persons                 — список жителей
GET   /persons/{id}            — профиль жителя + его изображения
GET   /unclassified            — неидентифицированные записи (для разметки)
POST  /persons                 — добавить жителя
POST  /persons/{id}/images     — добавить эталонное фото жителя
PATCH /unclassified/{id}       — назначить person_id вручную
POST  /training/export         — сформировать zip с обучающей выборкой
GET   /cameras                 — список камер и статус подключения
```

---

## ML Service (FastAPI + ONNX Runtime)

> **Статус: запланировано.** Описание интерфейса — спецификация для будущей реализации.

**Технологии:** Python 3.11+, FastAPI, ONNX Runtime, OpenCV, NumPy

Изображение приходит как `bytes` по multipart, декодируется в numpy array в памяти, подаётся в модель. На диск в режиме inference ничего не пишется.

### Inference

```
POST /inference/detect
  body: image (bytes, multipart)
  response: { "persons": [ { "bbox": [x1,y1,x2,y2], "confidence": 0.91 } ] }

POST /inference/classify
  body: image (bytes, multipart)   — crop одного человека
  response: { "class": "1_resident", "confidence": 0.87 }
  # class: 1_resident | 2_delivery | 3_utilities | 4_guest

POST /inference/identify
  body: image (bytes, multipart)   — crop одного человека
  response: { "person_id": "p_0042", "confidence": 0.83, "method": "face" }
  # person_id: null если не распознан; method: face | body
```

### Training

```
POST /training/run
  body: { "model": "detect|classify|identify",
          "images_dir": "/data/training/images",
          "labels_file": "/data/training/labels.json" }
  response: { "job_id": "abc123", "status": "started" }

GET  /training/status/{job_id}
GET  /models
POST /models/{model_type}/activate
```

### Формат labels.json

```json
{
  "version": "1.0",
  "task": "classify",
  "classes": ["1_resident", "2_delivery", "3_utilities", "4_guest"],
  "labels": [
    { "image": "img_001.jpg", "class": "2_delivery", "person_id": null },
    { "image": "img_002.jpg", "class": "1_resident", "person_id": "p_0042" }
  ]
}
```

---

## Telegram Bot (aiogram 3.0)

**Команды:**
```
/start          — приветствие
/events         — последние события (пагинация, inline-кнопки)
/persons        — список жителей
/unclassified   — записи для ручной разметки (только для админа)
/cameras        — статус камер (online/offline)
/export         — выгрузить обучающую выборку (только для админа)
```

**UX:**
- События — карточки: фото + подпись (дата, камера, класс, имя жителя)
- Навигация — inline-кнопки ← →, фильтры
- Разметка: бот показывает фото, предлагает выбрать person_id из списка или создать нового

---

## Конфигурация (config.yaml)

```yaml
thresholds:
  detection:
    min_confidence: 0.55      # детекция человека в кадре
  classification:
    min_confidence: 0.65      # классификация группы
  identification:
    min_confidence: 0.75      # идентификация жителя

sampling:
  fps: 2                      # анализировать 2 кадра в секунду на камеру
  substream: true             # субпоток для анализа, основной для сохранения

temporal:
  detection_window_size: 5    # M — размер скользящего окна
  detection_min_hits: 3       # N — минимум срабатываний для фиксации события
  event_end_silence_frames: 10

models:
  detect:   .models/detect/yolov8n.onnx
  classify: .models/classify/v1_1.onnx
  identify: .models/identify/v1.onnx
```

---

## Экспорт обучающей выборки

```
POST /training/export
  body: { "model": "classify", "include_unclassified": true }

→ training_export_20260417_153000.zip
    images/
      img_001.jpg
      ...
    labels.json
```
