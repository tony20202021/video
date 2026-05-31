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
pytest tests/test_ml_pipeline.py -v
# 18 тестов: classify, identify, pipeline, cosine similarity, preprocessing
```
