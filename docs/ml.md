# ML-пайплайн: классификация и идентификация

## Модели

Все модели в формате **ONNX** — единый рантайм, не требует PyTorch/TF в production.

| Задача | Модель | Размер | Скорость на CPU | Статус |
|--------|--------|--------|-----------------|--------|
| Детекция людей | YOLOv8n | ~13 MB | ~20–40 мс/кадр | **Готово** `.models/detect/yolov8n.onnx` |
| Классификация группы (Модель 1) | MobileNetV3-Small | ~10 MB | ~5–10 мс/crop | **v1 обучена** `.models/classify/v1.onnx` |
| Идентификация жителя (Модель 2) | MobileNetV3-Small | ~10 MB | ~5–10 мс/crop | Нужно обучить |

---

## Классы Модели 1

Три класса (папки датасета = имена классов):

| Класс | Описание |
|-------|----------|
| `1_resident` | Житель дома |
| `2_delivery` | Курьер / доставка |
| `3_utilities` | Коммунальные службы |
| `4_guest` | Гость (приходит надолго, заходит в квартиры, но не житель) |

Дополнительные папки датасета (не являются классами модели):

| Папка | Назначение |
|-------|------------|
| `skip/` | Намеренно пропущены при разметке (нечёткие кропы и т.п.) |
| `unknown/` | Не определился класс |
| `new/` | Новые кропы без разметки (→ `2_label_ui` → `1_dataset_groups apply`) |

---

## Почему две модели с одной архитектурой

| | Модель 1 (группы) | Модель 2 (жители) |
|---|---|---|
| Классы | 1_resident / 2_delivery / 3_utilities | конкретные жители + unknown_resident |
| Множество классов | закрытое, меняется редко | открытое (новый житель = новый класс) |
| Датасет | все группы, данных относительно много | только жители, данных мало |
| Переобучение | редко (новый тип посетителя) | раз в 2–4 недели по мере накопления |

Та же архитектура для обеих потому что:
- вход одинаковый (кроп сверху, тот же ракурс и освещение)
- датасет жителей маленький → более тяжёлый backbone даст переобучение, а не точность
- позволяет использовать трансфер цепочкой (см. ниже)

---

## Почему не cosine similarity по эмбеддингам для жителей

- **Угол камеры сверху** — стандартные ReID-модели обучены на виде сбоку/спереди,
  лицо часто не видно вообще
- **Разная одежда** — appearance-based эмбеддинги кодируют цвет куртки как основной
  признак; тот же человек в другой одежде окажется дальше по косинусному расстоянию,
  чем другой человек в похожей одежде
- **Нет контрольных фото** — у нас только кропы из тех же камер с тем же углом

Классификатор, обученный на накопленных кропах, учит признаки специфичные для нашего
ракурса (силуэт сверху, ширина плеч, форма головы) и с каждым циклом переобучения
становится лучше, включая новые outfit.

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
│  → 1_resident / 2_delivery /        │
│    3_utilities                      │
│  conf < 0.65 → uncertain            │
└───────────────┬─────────────────────┘
                │
        group == "1_resident"?
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

results = pipeline.run(bgr_frame, yolo_detections)
for r in results:
    # r.group_class   — "1_resident" / "2_delivery" / "3_utilities"
    # r.group_conf    — уверенность Модели 1
    # r.person_id     — "person_01" или "" если unknown_resident
    # r.identify_conf — уверенность Модели 2 (0 если group != "1_resident")
    print(r.group_class, r.person_id, r.identify_conf)
```

### Без обученных моделей

`GroupClassifier` и `PersonIdentifier` работают без моделей — возвращают `("unknown", 0.0)`.

---

## Workflow разметки и обучения Модели 1

```
1. pipeline/2_yolo_boxes_files → кропы в .output/pipeline/2_yolo_boxes_files/run_XXX/

2. Добавить в датасет:
   .\sh\train\1_dataset_groups.ps1 add -Src .output\pipeline\2_yolo_boxes_files\run_XXX -Dataset .data\groups\v1

3. Проверить на дубли с уже размеченным:
   .\sh\train\1_dataset_groups.ps1 check -Src .data\groups\v1\new -Dataset .data\groups\v1
   → уникальные → new/unique/, дубли → new/double/
   → пустые подкаталоги Src удаляются автоматически

4. Разметить:
   .\sh\train\2_label_ui.ps1          # Windows
   ./sh/train/2_label_ui.sh           # Linux (сервер)
   Порт — `LABEL_UI_PORT` в `.env` (по умолчанию 8750).

5. Применить разметку в датасет:
   .\sh\train\1_dataset_groups.ps1 apply -Move
   → файлы перемещаются из new/ в 1_resident/, 2_delivery/, 3_utilities/
   → если new/ стала пустой — удаляется автоматически

6. Обучить:
   .\sh\train\3_train_groups.ps1
   → .models/classify/v<N>.onnx

7. (опционально) Проверить промежуточный результат во время обучения:
   python scripts/train/3a_eval_pt.py
```

---

## Скрипты обучения

| Скрипт | Назначение |
|--------|-----------|
| `sh/train/1_dataset_groups.ps1` | Управление датасетом: `build / apply / check / add / status` |
| `sh/train/2_label_ui.ps1` | Запуск веб-разметчика кропов (Windows) |
| `sh/train/2_label_ui.sh` | То же на Linux-сервере |
| `sh/train/3_train_groups.ps1` | Обучение Модели 1 |
| `sh/train/5_train_residents.ps1` | Обучение Модели 2 |

---

## Веб-разметчик (2_label_ui)

```powershell
# Windows
.\sh\train\2_label_ui.ps1
.\sh\train\2_label_ui.ps1 -InputDir ".output\pipeline\2_yolo_boxes_files\run_XXX"
.\sh\train\2_label_ui.ps1 -UnlabeledOnly $true   # только неразмеченные
```

```bash
# Linux (сервер)
./sh/train/2_label_ui.sh
./sh/train/2_label_ui.sh --input .output/pipeline/2_yolo_boxes_files/run_XXX
./sh/train/2_label_ui.sh --port 8789   # переопределяет LABEL_UI_PORT
```

**Порт:** `LABEL_UI_PORT` в `.env` (дефолт — `8750`; на публичном сервере задайте `8750` или другой из диапазона `87**`, напр. `8789`).

**Доступ с сервера:** процесс слушает `0.0.0.0`. Если задан `ALLOWED_IPS` — те же правила, что у Transfer (отдельные IP и CIDR; localhost всегда разрешён). Без `ALLOWED_IPS` — предупреждение при старте, открыт для всех.

Открывает три URL (подставьте хост и порт; локально — `127.0.0.1`, с домашней машины — `<публичный-IP-сервера>`):

- `http://<host>:<port>/` — разметчик (одиночный кроп + кнопки)
- `http://<host>:<port>/gallery` — галерея сессии (все кропы, сгруппированы по метке)
- `http://<host>:<port>/gallery/dataset` — галерея датасета (файлы из папок датасета, перемещение между классами)

**Горячие клавиши:**

| Клавиша | Действие |
|---------|----------|
| `1` | 1_resident |
| `2` | 2_delivery |
| `3` | 3_utilities |
| `→` | следующий без разметки |
| `←` | предыдущий |
| `U` | пропустить (unknown) |

---

## Обучение Модели 1 (3_train_groups)

```powershell
# Стандартный запуск (данные из .data/groups/v1)
.\sh\train\3_train_groups.ps1

# Явный путь и параметры
.\sh\train\3_train_groups.ps1 -Data ".data\groups\v1" -Epochs 30

# Отключить коррекцию дисбаланса
.\sh\train\3_train_groups.ps1 -ClassWeights $false -WeightedSampling $false
```

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `-Data` | `.data\groups\v1` | Папка датасета или zip |
| `-Epochs` | `20` | Число эпох |
| `-BatchSize` | `32` | Размер батча |
| `-Lr` | `1e-3` | Learning rate |
| `-ValSplit` | `0.2` | Доля валидации (стратифицированная) |
| `-ClassWeights` | `$true` | Взвешенная функция потерь |
| `-WeightedSampling` | `$true` | WeightedRandomSampler |

**Обработка дисбаланса классов:**

Оба флага включены по умолчанию — актуально при дисбалансе (текущий датасет ~15:1).
- `ClassWeights` — веса классов в `CrossEntropyLoss`, обратно пропорционально частоте
- `WeightedSampling` — `WeightedRandomSampler`: равномерная выборка из каждого класса в батче
- Разбивка train/val — **стратифицированная**: `val_split` % берётся из каждого класса отдельно

**Фазы обучения:**
1. Эпохи 1 → N/2: обучается только классификационная голова (backbone заморожен)
2. Эпохи N/2+1 → N: размораживается вся сеть, lr × 0.1

**Результаты:**
```
.models/classify/v<N>.onnx             — модель для production
.models/classify/backbone.pt           — backbone для инициализации Модели 2
.output/train/3_train_groups/run/training_results.json
```

**Аугментация при обучении:**
```python
transforms.RandomCrop(224)
transforms.RandomHorizontalFlip()
transforms.RandomRotation(10)               # небольшой наклон камеры
transforms.RandomPerspective(0.2, p=0.5)   # угол обзора камеры
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
transforms.RandomErasing(p=0.3)            # частичное перекрытие людей
```

**Проверка во время обучения (пока идут эпохи):**
```powershell
& "$env:USERPROFILE\miniconda3\envs\conda_video\python.exe" scripts\train\3a_eval_pt.py
```
Показывает точность по каждому классу на случайной выборке из датасета.

---

## Идентификация жителей (Модель 2)

### Добавить нового жителя

```
1. Накопить 15–30 кропов из разных дней (разная одежда)
2. Разметить в 2_label_ui
3. Запустить переобучение Модели 2
```

До переобучения новый житель попадёт в `unknown_resident`.

### Переобучение

Цикл переобучения — раз в 2–4 недели или при добавлении нового жителя.
Всегда обучать на **полном накопленном датасете** (иначе catastrophic forgetting).

```
Цикл 1:  backbone Модели 1  → обучение на {неделя 1}       → Модель 2 v1
Цикл 2:  Модель 2 v1        → обучение на {неделя 1 + 2}   → Модель 2 v2
Цикл 3:  Модель 2 v2        → обучение на {неделя 1+2+3}   → Модель 2 v3
```

```powershell
# Первый цикл — backbone от Модели 1
.\sh\train\5_train_residents.ps1 -Data export.zip -Backbone .models\classify\backbone.pt

# Последующие циклы — веса предыдущей версии
.\sh\train\5_train_residents.ps1 -Data export_full.zip -InitFrom .models\identify\v1.pt
```

### Трансфер между моделями

```
ImageNet weights
    ↓
fine-tune на датасете групп (1_resident / 2_delivery / 3_utilities)
    ↓
Модель 1  (.models/classify/v1.onnx)
    │
    └── backbone.pt (только features, без головы)
            ↓
        fine-tune на датасете жителей
            ↓
        Модель 2 v1  (.models/identify/v1.onnx)
            ↓ (от неё же)
        Модель 2 v2  (.models/identify/v2.onnx)
```

---

## Структура моделей

```
.models/
  detect/
    yolov8n.onnx        — детекция людей (готово)
  classify/
    v1.onnx             — Модель 1: обучена на датасете .data/groups/v1
    backbone.pt         — только features (PyTorch) для инициализации Модели 2
  identify/
    v1.onnx             — Модель 2: после первого обучения на жителях
  osd/
    osd_templates.npz     — шаблоны цифр OSD (HI-поток)
    osd_templates_low.npz — шаблоны цифр OSD (LOW-поток)
```

---

## Параметры качества (config.yaml)

```yaml
thresholds:
  detection:
    min_confidence: 0.35   # YOLO — порог детекции человека
  classification:
    min_confidence: 0.65   # Модель 1 — ниже → "uncertain"
  identification:
    min_confidence: 0.70   # Модель 2 — ниже → unknown_resident
```

На ранних версиях Модели 2 (мало данных) стоит поднять порог идентификации до 0.80.

---

## Аугментация Модели 2 (жители)

Обязательно сильный **цветовой jitter** — модель не должна запоминать цвет одежды:

```python
transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.15)
```

---

## Тесты

```bash
pytest tests/test_ml_pipeline.py tests/test_cameras_6x.py tests/test_motion_utils.py tests/test_osd_time.py -v
pytest tests/ -q
```
