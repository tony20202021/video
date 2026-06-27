# ML-пайплайн: классификация и идентификация

## Модели

Все модели в формате **ONNX** — единый рантайм, не требует PyTorch/TF в production.

| Задача | Модель | Размер | Скорость на CPU | Статус |
|--------|--------|--------|-----------------|--------|
| Детекция людей | YOLOv8n | ~13 MB | ~20–40 мс/кадр | **Реализовано** `models/yolov8n.onnx` |
| Классификация группы | MobileNetV3-Small | ~10 MB | ~5–10 мс/crop | Код готов, нужно обучить |
| Идентификация жителя | MobileFaceNet | ~4 MB | ~5 мс/лицо | Код готов, нужна предобученная модель |

---

## Пайплайн (один кадр)

```
Кадр с HI-потока (2304×2592)
      │
      ▼
[ frame diff на LOW → движение? ]
      │ нет → пропустить
      │ да
      ▼
┌─────────────────────────────────────┐
│  YOLOv8n: детекция людей            │
│  → список bounding boxes            │
└───────────────┬─────────────────────┘
                │
         для каждого bbox:
         crop + resize
                │
                ▼
┌─────────────────────────────────────┐
│  MobileNetV3-Small: группа          │
│  → resident / courier / delivery /  │
│    utilities / other / unknown      │
└───────────────┬─────────────────────┘
                │
        group == "resident"?
                │
                ▼
┌─────────────────────────────────────┐
│  MobileFaceNet: идентификация       │
│  embedding → cosine similarity      │
│  с эмбеддингами жителей из БД       │
└───────────────┬─────────────────────┘
                │
          sim ≥ threshold?
          ├─ да → person_id
          └─ нет → unclassified_persons
```

---

## Код

```
src/ml/
  classify.py     — GroupClassifier (MobileNetV3-Small ONNX wrapper)
  identify.py     — PersonIdentifier (MobileFaceNet + cosine similarity)
  pipeline.py     — MLPipeline: оркестрация classify → identify
```

### Использование

```python
from ml.pipeline import MLPipeline, MLConfig
from pathlib import Path

pipeline = MLPipeline(MLConfig(
    classify_model=Path("models/classify/v1.onnx"),
    identify_model=Path("models/identify/v1.onnx"),
))

# Загружаем эмбеддинги жителей из БД
pipeline.load_person_embeddings({"p_001": [[0.1, 0.2, ...]]})

# Обрабатываем кадр
results = pipeline.run(bgr_frame, yolo_detections)
for r in results:
    print(r.group_class, r.person_id, r.identify_conf)
```

### Без обученных моделей

`GroupClassifier` и `PersonIdentifier` работают без моделей — возвращают `("unknown", 0.0)`.
Это позволяет запускать систему до обучения.

---

## Система разметки

### Workflow

```
1. Агент обнаруживает человека → кроп + событие в MongoDB
   (group_class=None, person_id=None → unclassified_persons)

2. Админ в Telegram: /unclassified
   → видит фото, выбирает group_class и/или person_id

3. PATCH /unclassified/{id}   → reviewed=True, assigned_person_id

4. POST /training/export      → скачать ZIP (images/ + labels.json)

5. python scripts/train/train_classifier.py --data export.zip

6. Обученная модель → models/classify/v2.onnx
```

### API разметки

```
GET  /unclassified                     — список неразмеченных
PATCH /unclassified/{id}               — назначить person_id + group_class

POST /persons                          — создать жителя
POST /persons/{id}/images              — добавить эталонное фото (→ вычислить эмбеддинг)
GET  /persons                          — список жителей
```

### Веб-интерфейс разметки (label_ui)

`scripts/train/4_label_ui.py` — локальный Flask-сервер для ручной разметки кропов.  
Открывает браузер автоматически. Метки сохраняются в `labels.json` при каждом клике.

```bash
# Разметить кропы из последнего прогона 5_diff_yolo_boxes_low
python scripts/train/4_label_ui.py

# Указать конкретный каталог и порт
python scripts/train/4_label_ui.py --input .output/cameras/5_diff_yolo_boxes_low/run_XXX/crops --port 5050

# Сохранить метки в явный файл
python scripts/train/4_label_ui.py --labels .output/train/4_label_ui/labels.json
```

**Горячие клавиши в браузере:**

| Клавиша | Действие |
|---------|----------|
| `1` | resident |
| `2` | courier |
| `3` | delivery |
| `4` | utilities |
| `5` | other |
| `→` | следующий |
| `←` | предыдущий |
| `U` | пропустить |

Классы соответствуют `CLASSES` в `src/ml/classify.py`.

Результат: `labels.json` → передать в `1_export_data.py` или сохранить для дообучения.

---

## Обучение классификатора

### 1. Собрать данные

```bash
# Экспортировать размеченные данные из MongoDB
python scripts/train/export_data.py --task classify

# Или через API
GET /training/export?model=classify   → скачивает ZIP
```

### 2. Обучить модель

```bash
pip install torch torchvision

python scripts/train/train_classifier.py \
    --data .output/training/export_classify_*.zip \
    --epochs 30
```

Результат: `models/classify/v1.onnx`

### 3. Проверить метрики

```
models/classify/v1.onnx   — модель для production
.output/training/run/training_results.json  — val_acc, history
```

### 4. Активировать

Обновить `config.yaml`:
```yaml
models:
  classify: models/classify/v1.onnx
```

---

## Идентификация жителей

### Добавить жителя

```bash
# Через API (bot отправляет фото)
POST /persons              body: {person_id: "p_0001", name: "Иванов И.И.", apartment_id: "42"}
POST /persons/p_0001/images  body: {image_b64: "<base64>"}
```

При добавлении фото эмбеддинг вычисляется автоматически (если модель доступна) и сохраняется в MongoDB.

### Обновить эмбеддинги

После смены модели (`identify/v2.onnx`) пересчитать все эмбеддинги:
```bash
python scripts/train/export_data.py --task identify
```

---

## Структура моделей

```
models/
  yolov8n.onnx          — детекция людей (готово)
  classify/
    v1.onnx             — классификатор группы (создаётся обучением)
    v2.onnx             — следующая версия после дообучения
  identify/
    v1.onnx             — MobileFaceNet (скачать: scripts/setup_models.py --task identify)
```

---

## Параметры качества (config.yaml)

```yaml
thresholds:
  detection:
    min_confidence: 0.35   # YOLO — порог детекции человека
  classification:
    min_confidence: 0.65   # MobileNetV3 — ниже → "uncertain"
  identification:
    min_confidence: 0.75   # MobileFaceNet — ниже → unclassified_persons
```

---

## Тесты

```bash
# Unit-тесты ML и utils
pytest tests/test_ml_pipeline.py tests/test_motion_utils.py tests/test_osd_time.py -v

# Интеграционные тесты на реальных данных (требует tests/data/)
pytest tests/test_integration.py -v

# Все тесты
pytest tests/ -q
```

---

## Временны́е метки кадра (OSD)

Камера iCSee записывает время в правый верхний угол кадра (`YYYY-MM-DD HH:MM:SS`).
Модуль `src/common/utils/osd_time.py` извлекает это время без OCR-библиотек:

```python
from common.utils.osd_time import extract_osd_time
cam_dt = extract_osd_time(hi_frame)   # datetime | None
```

Шаблоны цифр хранятся в `models/osd_templates.npz`.
Сборка шаблонов из нового кадра с известным временем:

```python
from common.utils.osd_time import build_templates_from_frame
build_templates_from_frame(frame, "2026-05-31 14:49:31")
```

Включение в `5_diff_yolo_boxes_low.py`:
```
python scripts/cameras/5_diff_yolo_boxes_low.py --cam-ts
```
Имя файла: `{stem}_{cam_YYYYMMDD_HHMMSS}_{pc_метка}_{suffix}.jpg`

### Точное время через RTCP NTP

Модуль `src/common/utils/rtcp_time.py` извлекает точное время из сетевых пакетов RTSP/TCP.  
Метод: RTCP Sender Report (PT=200) содержит 64-bit NTP timestamp — точность ±0.1 сек.

```python
from common.utils.rtcp_time import RtcpTimingReader

reader = RtcpTimingReader(rtsp_url)
calib = reader.get_calibration(timeout=5.0)  # ждём первый RTCP SR
if calib:
    # calib.ntp_unix  — Unix timestamp из NTP (wall clock камеры)
    # calib.rtp_ts    — соответствующий RTP timestamp
    # calib.clock_rate — частота часов (обычно 90000)
    cam_time = calib.rtp_to_datetime(rtp_ts)
```

Отличие от OSD: OSD даёт точность 1 сек (по экранным цифрам), RTCP — субсекундную.  
OSD работает без подключения к сети (из уже захваченного кадра), RTCP требует отдельного RTSP соединения.

---

## Бенчмарки

```bash
# FPS пропускная способность по компонентам
python scripts/bench/fps_bench.py --duration 30

# Сравнение детекции LOW vs HI разрешения
python scripts/bench/compare_lohi.py --pairs 20

# Рассинхрон LOW/HI потоков + camera-PC offset
python scripts/bench/sync_offset.py --duration 60 --cam-ts
```

---

## Тестовые данные

```bash
# Собрать представительную выборку из .output/
python scripts/utils/collect_test_frames.py
```

Структура `tests/data/`:
- `raw_pairs/` — HI-кадры при срабатывании порога движения
- `no_person/` — heartbeat кадры (пустая сцена)
- `with_person/` — кадры с обнаруженными людьми
- `artifacts/` — маленькие/нетипичные кадры
