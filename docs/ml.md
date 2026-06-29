# ML-пайплайн: классификация и идентификация

## Модели

Все модели в формате **ONNX** — единый рантайм, не требует PyTorch/TF в production.

| Задача | Модель | Размер | Скорость на CPU | Статус |
|--------|--------|--------|-----------------|--------|
| Детекция людей | YOLOv8n | ~13 MB | ~20–40 мс/кадр | **Готово** `.models/detect/yolov8n.onnx` |
| Классификация группы (Модель 1) | MobileNetV3-Small | ~10 MB | ~5–10 мс/crop | v0 — болванка, нужно обучить |
| Идентификация жителя (Модель 2) | MobileNetV3-Small | ~10 MB | ~5–10 мс/crop | Нужно обучить |

---

## Почему две модели с одной архитектурой

Задачи принципиально разные по типу и частоте обновления:

| | Модель 1 (группы) | Модель 2 (жители) |
|---|---|---|
| Классы | resident / courier / delivery / utilities / other | конкретные жители + unknown_resident |
| Множество классов | закрытое, меняется редко | открытое (новый житель = новый класс) |
| Датасет | все группы, данных относительно много | только жители, данных мало |
| Переобучение | редко (новый тип посетителя) | раз в 2–4 недели по мере накопления данных |

Одна модель на обе задачи не подходит: периодическое переобучение Модели 2 (новые жители,
новые outfit) сдвигало бы backbone и деградировало бы Модель 1.

Та же архитектура для обеих потому что:
- вход одинаковый (кроп сверху, тот же ракурс и освещение)
- датасет жителей маленький → более тяжёлый backbone даст переобучение, а не точность
- позволяет использовать трансфер цепочкой (см. ниже)

---

## Почему не cosine similarity по эмбеддингам для жителей

Очевидный подход — взять предобученные эмбеддинги (ReID, FaceNet) и сравнивать их
косинусным расстоянием. Он не подходит по нескольким причинам:

- **Угол камеры сверху** — стандартные ReID-модели обучены на виде сбоку/спереди,
  у них другое распределение признаков; лицо часто не видно вообще
- **Разная одежда** — appearance-based эмбеддинги кодируют цвет куртки как основной
  признак; тот же человек в другой одежде будет дальше по косинусному расстоянию,
  чем другой человек в похожей одежде
- **Нет контрольных фото** — для enrollment нужны фотографии в контролируемых
  условиях; у нас только кропы из тех же камер с тем же углом

Классификатор, обученный на накопленных кропах из тех же камер, учит признаки
специфичные для нашего ракурса (форма силуэта сверху, ширина плеч, форма головы)
и с каждым циклом переобучения становится лучше, включая новые outfit.

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
│  Модель 1 — MobileNetV3-Small       │
│  → resident / courier / delivery /  │
│    utilities / other                │
│  conf < 0.65 → uncertain            │
└───────────────┬─────────────────────┘
                │
        group == "resident"?
                │ да
                ▼
┌─────────────────────────────────────┐
│  Модель 2 — MobileNetV3-Small       │
│  → Иванов(0.91) / Петров(0.43) /   │
│    ... / unknown_resident           │
│  conf < 0.70 → unknown_resident     │
└───────────────┬─────────────────────┘
                │
          conf ≥ 0.70?
          ├─ да → person_id (авто-метка)
          └─ нет → очередь на ручную разметку
```

---

## Код

```
src/ml/
  classify.py     — GroupClassifier (Модель 1: MobileNetV3-Small ONNX wrapper)
  identify.py     — PersonIdentifier (Модель 2: MobileNetV3-Small ONNX wrapper)
  pipeline.py     — MLPipeline: оркестрация classify → identify
```

### Использование

```python
from ml.pipeline import MLPipeline, MLConfig
from pathlib import Path

pipeline = MLPipeline(MLConfig(
    classify_model=Path(".models/classify/v1.onnx"),
    identify_model=Path(".models/identify/v1.onnx"),
))

# Обрабатываем кадр
results = pipeline.run(bgr_frame, yolo_detections)
for r in results:
    # r.group_class   — "resident" / "courier" / ...
    # r.group_conf    — уверенность Модели 1
    # r.person_id     — "person_01" или "" если unknown_resident
    # r.identify_conf — уверенность Модели 2 (0 если group != "resident")
    print(r.group_class, r.person_id, r.identify_conf)
```

### Без обученных моделей

`GroupClassifier` и `PersonIdentifier` работают без моделей — возвращают `("unknown", 0.0)`.
Это позволяет запускать скрипты до обучения.

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

6. Обученная модель → .models/classify/v2.onnx
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

python scripts/train/2_train_groups.py \
    --data .output/training/export_classify_*.zip \
    --epochs 30
```

Результат: `.models/classify/v1.onnx` и `.models/classify/backbone.pt` (для Модели 2)

### 3. Проверить метрики

```
.models/classify/v1.onnx       — модель для production
.models/classify/backbone.pt   — backbone для инициализации Модели 2
.output/train/2_train_groups/run/training_results.json  — val_acc, history
```

### 4. Активировать

Обновить `config.yaml`:
```yaml
models:
  classify: .models/classify/v1.onnx
```

---

## Идентификация жителей

### Добавить нового жителя

Новый житель = новый класс в Модели 2. Без переобучения модель его не знает.

```
1. Накопить 15–30 кропов жителя из разных дней (→ разная одежда)
2. Разметить вручную (label_ui: назначить person_id)
3. Добавить в датасет жителей: datasets/residents/person_01/
4. Запустить переобучение Модели 2 (см. ниже)
```

До переобучения новый житель будет попадать в `unknown_resident` — это ожидаемо.

### Переобучение Модели 2

Цикл переобучения — раз в 2–4 недели или при добавлении нового жителя.

**Важно:** каждый раз обучать на **полном накопленном датасете** (не только новые данные),
иначе модель забудет старых жителей (catastrophic forgetting).

**Стартовые веса:** от предыдущей версии Модели 2 (не от Модели 1 заново).
Исключение: если изменился состав классов кардинально (жители съехали) — тогда
перезапустить с весов backbone Модели 1.

```
Цикл 1:  backbone Модели 1  → обучение на {неделя 1}       → Модель 2 v1
Цикл 2:  Модель 2 v1        → обучение на {неделя 1 + 2}   → Модель 2 v2
Цикл 3:  Модель 2 v2        → обучение на {неделя 1+2+3}   → Модель 2 v3
```

**Команды:**

```bash
# Первый цикл — backbone от Модели 1
python scripts/train/5_train_residents.py \
    --data export_identify.zip \
    --backbone .models/classify/backbone.pt \
    --epochs 30

# Последующие циклы — веса предыдущей версии Модели 2
python scripts/train/5_train_residents.py \
    --data export_identify_full.zip \
    --init-from .models/identify/v1.pt \
    --epochs 30
```

Через PowerShell: `.\sh\train\5_train_residents.ps1 -Data export.zip -Backbone .models\classify\backbone.pt`

### Трансфер между моделями

```
ImageNet weights
    ↓
fine-tune на датасете групп (resident/courier/...)
    ↓
Модель 1  (.models/classify/v1.onnx)
    │
    └── сохранить backbone без головы
            ↓
        fine-tune на датасете жителей
            ↓
        Модель 2 v1  (.models/identify/v1.onnx)
            ↓ (от неё же)
        Модель 2 v2  (.models/identify/v2.onnx)
            ↓ ...
```

Смысл: Модель 1 учит backbone понимать «что такое человек в нашем ракурсе сверху»
лучше, чем ImageNet. Модель 2 стартует с этих весов — ей легче выучить тонкие
различия между жителями.

---

## Структура моделей

```
.models/
  detect/
    yolov8n.onnx        — детекция людей (готово)
    yolov8s.onnx        — альтернатива: точнее, но медленнее
  classify/
    v0.onnx             — Модель 1: болванка (низкая точность, нужно обучить)
    v1.onnx             — после первого обучения на размеченных данных
    backbone.pt         — только features (PyTorch) для инициализации Модели 2
  identify/
    v1.onnx             — Модель 2: после первого обучения на жителях
    v2.onnx             — после первого цикла переобучения
    ...
  osd/
    osd_templates.npz     — шаблоны цифр OSD (HI-поток)
    osd_templates_low.npz — шаблоны цифр OSD (LOW-поток)
```

---

## Аугментация при обучении

**Модель 1 (группы):** стандартная — случайный crop, flip, нормализация яркости.

**Модель 2 (жители):** обязательно сильный **цветовой jitter** (HSV ±30–40%) чтобы модель
не запоминала цвет одежды как основной признак. Без этого одежда доминирует над
формой тела, и модель плохо обобщается на другие дни.

```python
transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.15)
```

---

## Параметры качества (config.yaml)

```yaml
thresholds:
  detection:
    min_confidence: 0.35   # YOLO — порог детекции человека
  classification:
    min_confidence: 0.65   # Модель 1 — ниже → "uncertain" (не назначать группу)
  identification:
    min_confidence: 0.70   # Модель 2 — ниже → unknown_resident (в очередь на разметку)
```

Порог идентификации 0.70 — эмпирический старт. На ранних версиях Модели 2
(мало данных) его стоит поднять до 0.80, чтобы уменьшить количество неверных
авто-меток.

---

## Тесты

```bash
# Unit-тесты ML, utils и camera-скриптов
pytest tests/test_ml_pipeline.py tests/test_cameras_6x.py tests/test_motion_utils.py tests/test_osd_time.py -v

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

Шаблоны цифр хранятся в `.models/osd/osd_templates.npz`.
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
