# ML-пайплайн: классификация и идентификация

## Модели

Все модели в формате **ONNX** — единый рантайм, не требует PyTorch/TF в production.

| Задача | Модель | Размер | Скорость на CPU | Статус |
|--------|--------|--------|-----------------|--------|
| Детекция людей | YOLOv8n | ~13 MB | ~20–40 мс/кадр | **Готово** `.models/detect/yolov8n.onnx` |
| Классификация группы (Модель 1) | MobileNetV3-Small | ~6 MB | ~5–10 мс/crop | **Обучена** `.models/classify/v1_1.onnx` (датасет v1, val_acc 79%) |
| Идентификация жителя (Модель 2) | MobileNetV3-Small | ~10 MB | ~5–10 мс/crop | Нужно обучить |

---

## Классы Модели 1

Единственный источник правды — `src/common/utils/classes.py`:

```python
GROUP_CLASSES = ["1_resident", "2_delivery", "3_utilities", "4_guest"]
RESIDENT_CLASS = "1_resident"   # для идентификации жителей (Модель 2)
EXTRA_DATASET_DIRS = ["skip", "unknown", "new"]
GROUP_CLASS_COLORS  # цвета для графиков pipeline
GROUP_CLASS_RU      # подписи в Telegram-боте
LEGACY_CLASS_MIGRATIONS  # resident → 1_resident, courier → unknown, …
```

Импорт: `from common.utils.classes import GROUP_CLASSES, RESIDENT_CLASS`.

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
  versions.py     — версии моделей: v1_1.onnx, манифесты, activate
```

### Использование

```python
from ml.pipeline import MLPipeline, MLConfig
from pathlib import Path

pipeline = MLPipeline(MLConfig(
    classify_model=Path(".models/classify/v1_1.onnx"),
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

Автоматические watch-скрипты (запускать параллельно с пайплайном):

```bash
# Терминал A — детекция + кропы
./sh/pipeline/2_yolo_boxes_files.sh        # watch + delete-after по умолчанию

# Терминал B — проверка кропов на дубли с датасетом → new/
./sh/train/1_1_dataset_groups_check.sh     # watch: .output/pipeline/2_yolo_boxes_files/images → .data/groups/v1/new/

# Терминал C — дедупликация внутри new/ (одинаковые имена из разных прогонов)
./sh/train/1_2_dataset_groups_check_new.sh # watch: .data/groups/v1/new/
```

Полный цикл разметки и обучения:

```
1. Накопить кропы:
   2_yolo_boxes_files.sh + 1_1_dataset_groups_check.sh + 1_2_dataset_groups_check_new.sh
   → уникальные кропы в .data/groups/v1/new/

2. Разметить:
   ./sh/train/2_label_ui.sh
   Порт — LABEL_UI_PORT в .env (по умолчанию 8750).
   Три экрана с навигацией: / (разметчик) ↔ /gallery (галерея сессии) ↔ /gallery/dataset.
   Галерея /gallery: SHIFT+click для выделения диапазона (в рамках одной секции).

3. Применить разметку в датасет:
   ./sh/train/1_dataset_groups.sh apply --move
   → файлы из v1/new/ → v1/dataset/1_resident/, v1/dataset/2_delivery/, … по меткам

4. Обучить:
   ./sh/train/4_train_groups.sh
   → .models/classify/v1_1.onnx (+ v1_1.json манифест)

5. (опционально) Проверить во время обучения:
   python scripts/train/3a_eval_pt.py
```

Ручные операции с датасетом (при необходимости):

```bash
# Добавить кропы из прогона в v1/new/ (с проверкой дублей против датасета)
python scripts/train/dataset_groups.py add \
    --src .output/pipeline/2_yolo_boxes_files/images \
    --dataset .data/groups/v1/dataset

# Проверить v1/new/ против датасета (вручную)
python scripts/train/dataset_groups.py check \
    --src .data/groups/v1/new \
    --dataset .data/groups/v1/dataset

# Статус датасета
python scripts/train/dataset_groups.py status --dataset .data/groups/v1/dataset
```

---

## Инференс Модели 1 (3_classify_groups.sh)

Watch-скрипт непрерывно следит за кропами и классифицирует их:

```bash
# Терминал — инференс (запускать параллельно с 2_yolo_boxes_files.sh)
./sh/pipeline/3_classify_groups.sh                  # watch, poll 60s, move по умолчанию
./sh/pipeline/3_classify_groups.sh --poll-sec 30    # другой интервал
./sh/pipeline/3_classify_groups.sh --copy           # копировать вместо перемещения
./sh/pipeline/3_classify_groups.sh --once           # один прогон и выход
./sh/pipeline/3_classify_groups.sh --classify-conf 0.70
```

**Источник:** `.output/pipeline/2_yolo_boxes_files/images` — все `*.jpg` рекурсивно (любая вложенность).

**Выход:** `.data/groups/v1/inference/`
```
inference/
  images/YYYYMMDD/
    1_resident/crop.jpg
    2_delivery/crop.jpg
    unknown/crop.jpg        ← conf < CLASSIFY_CONF
  meta/YYYYMMDD/
    classifications.csv
    run_params.json
```

По умолчанию файлы **перемещаются** из источника; `--copy` оставляет оригиналы.
Расширение файлов: `--ext jpg` (по умолчанию).

Порог уверенности: `CLASSIFY_CONF` из `.env` (дефолт `0.65`); ниже → класс `unknown`.

Для автоматического формирования `labels.json` по итогам инференса (совместимого с `2_label_ui` и `apply`):

```bash
# Терминал — labels из инференса (watch-режим)
./sh/train/1_3_dataset_groups_inference_labels.sh

# Применить в датасет (вручную, после проверки)
./sh/train/1_dataset_groups.sh apply \
    --labels .data/groups/v1/inference/images/YYYYMMDD/labels.json \
    --dataset .data/groups/v1/dataset --move
```

---

## Сборка датасета v2 (итеративное улучшение)

После накопления инференса Модели 1 собирается новый датасет для переобучения.
Источники разбиты на 8 стратегий по ценности данных.

### Пороги уверенности

```dotenv
# .env
CLASSIFY_CONF=0.65        # основной порог инференса (conf < → uncertain/)
CLASSIFY_CONF_HIGH=0.85   # второй порог: выше — псевдо-метки, ниже — переобучение
```

Зона между двумя порогами — «серая»: модель уверена достаточно чтобы не попасть в uncertain,
но недостаточно чтобы быть надёжным псевдо-лейблом.

### Стратегии отбора

| # | Суть | Источник | Условие | Метка берётся из |
|---|------|----------|---------|-----------------|
| 1 | Модель ошиблась — мы исправили вручную | `inference/images/YYYYMMDD/<class>/` | класс в `labels.json` ≠ subdirectory | `labels.json` (ручная переразметка) |
| 2 | Модель не была уверена — мы разметили вручную | `inference/images/YYYYMMDD/uncertain/` | файл есть в `labels.json` | `labels.json` (ручная разметка) |
| 3 | Модель угадала, но без уверенности — серая зона | `classifications.csv` | `CLASSIFY_CONF ≤ conf < CLASSIFY_CONF_HIGH` | `labels.json` |
| 4 | Старый датасет — модель на нём колеблется | `v1/dataset/` | инференс на датасете: `conf < CLASSIFY_CONF_HIGH` | имя подкаталога датасета |
| 5 | Модель уверена и права — берём бесплатно | `classifications.csv` | `conf ≥ CLASSIFY_CONF_HIGH` И метка совпадает с subdirectory | subdirectory (псевдо-метка, без переразметки) |
| 6 | То же что 1–3 но по другим дням (разная одежда, свет) | Другие даты `inference/images/YYYYMMDD/` | стратегии 1–3 по каждой дате | `labels.json` соответствующей даты |
| 7 | Модель уверена, но систематически врёт | `v1/dataset/` | инференс: `conf ≥ CLASSIFY_CONF_HIGH` И predicted ≠ true\_class | имя подкаталога датасета |
| 8 | Два класса почти одинаково вероятны — граничный случай | `classifications.csv` | `margin = conf_top1 − conf_top2 < порог` | `labels.json` |

**Стратегия 7** (уверенные ошибки на датасете) — наиболее ценная из дополнительных:
модель уверена, но систематически ошибается. Без таких примеров переобучение не исправит баг.

**Стратегия 8** (margin sampling) — лучшая эвристика активного обучения.
`clf.classify()` возвращает вероятности всех 4 классов — margin вычисляется из них.
Малый margin = граничный случай между двумя классами, максимально информативен.

Пример сравнения:
```
[0.68, 0.63, 0.05, 0.04]  conf=0.68, margin=0.05  ← брать (граница 4_guest/1_resident)
[0.68, 0.15, 0.10, 0.07]  conf=0.68, margin=0.53  ← не брать (просто немного неуверен)
```

### Обязательные фильтры поверх всех стратегий

**Временно́е дедублирование** — одна сцена за 5 минут даёт 200+ похожих кропов.
Лимит: не более 1 кропа на камеру на 15-минутное окно (из имени файла).
Применять к стратегиям 3, 5, 6.

**Квота на класс** — не добавлять пропорционально текущему балансу (усилит дисбаланс).
Целевое распределение: равное по классам или с перевесом в сторону дефицитных.

### Предлагаемое распределение по скриптам

```
sh/train/
  1_1_dataset_groups_check.sh              — (существует) watch: yolo → new/
  1_2_dataset_groups_check_new.sh          — (существует) watch: dedup new/
  1_3_dataset_groups_inference_labels.sh   — (существует) inference/ → labels.json

  3_dataset_from_inference.sh              — стратегии 1,2,3,5,8: один каталог инференса за дату
                                             принимает аргумент: inference/images/YYYYMMDD/
  3_dataset_from_prev.sh                   — стратегии 4,7: прогон модели на предыдущем датасете
                                             принимает аргумент: путь к датасету (напр. v1/dataset)
  3_dataset_build.sh                       — мастер-скрипт:
                                             вызывает 3_dataset_from_prev.sh один раз
                                             затем циклом по всем датам — 3_dataset_from_inference.sh

scripts/train/
  dataset_from_inference.py   — логика стратегий 1,2,3,5,8 (читает labels.json + classifications.csv)
  dataset_from_prev.py        — логика стратегий 4,7 (запускает инференс на датасете)
  # или один модуль dataset_v2.py с функциями для обоих сценариев
```

`3_dataset_from_inference.sh` вызывается по одному разу на каждую дату — идемпотентен,
можно перезапускать при изменении labels.json или порогов.

`3_dataset_build.sh`:
```bash
# Старый датасет — один раз
./sh/train/3_dataset_from_prev.sh .data/groups/vN/dataset

# Инференс — по каждой дате
for date_dir in .data/groups/vN/inference/images/*/; do
    ./sh/train/3_dataset_from_inference.sh "$date_dir"
done
```

### Структура датасета v2

```
.data/groups/
  v1/
    dataset/     ← исходный датасет, не трогать
    inference/   ← накопленный инференс
  v2/
    dataset/     ← v1/dataset + новые примеры из стратегий 1–8
      1_resident/
      2_delivery/
      3_utilities/
      4_guest/
    sources.csv  ← откуда взят каждый файл (стратегия, исходный путь, conf, margin)
```

`sources.csv` — аудит-лог: для каждого файла в v2/dataset фиксируется стратегия
и метрики на момент отбора, чтобы потом понять что помогло, а что нет.

---

## Скрипты обучения

| Скрипт | Назначение |
|--------|-----------|
| `sh/train/1_1_dataset_groups_check.sh` | Watch: кропы из 2_yolo_boxes/images → v1/new/ (дубли против датасета) |
| `sh/train/1_2_dataset_groups_check_new.sh` | Watch: дедуп внутри v1/new/ по имени файла |
| `sh/train/1_3_dataset_groups_inference_labels.sh` | Watch: inference/images/YYYYMMDD/ → labels.json по структуре класс-каталогов |
| `sh/train/1_dataset_groups.sh` | Ручное управление датасетом: `build / apply / check / add / status` |
| `sh/train/2_label_ui.sh` | Запуск веб-разметчика кропов |
| `sh/train/4_train_groups.sh` | Обучение Модели 1 (датасет из v1/dataset/) |
| `sh/train/5_train_residents.sh` | Обучение Модели 2 |

---

## Веб-разметчик (2_label_ui)

```bash
./sh/train/2_label_ui.sh
./sh/train/2_label_ui.sh --input .output/pipeline/2_yolo_boxes_files/run_XXX
./sh/train/2_label_ui.sh --port 8789   # переопределяет LABEL_UI_PORT
./sh/train/2_label_ui.sh --probs       # показывать вероятности из classifications.csv
```

**Порт:** `LABEL_UI_PORT` в `.env` (дефолт — `8750`; на публичном сервере задайте `8750` или другой из диапазона `87**`, напр. `8789`).

**Доступ с сервера:** процесс слушает `0.0.0.0`. Если задан `ALLOWED_IPS` — те же правила, что у Transfer (отдельные IP и CIDR; localhost всегда разрешён). Без `ALLOWED_IPS` — предупреждение при старте, открыт для всех.

Открывает три URL (подставьте хост и порт; локально — `127.0.0.1`, с домашней машины — `<публичный-IP-сервера>`):

- `http://<host>:<port>/` — разметчик (одиночный кроп + кнопки)
- `http://<host>:<port>/gallery` — галерея сессии (все кропы, сгруппированы по метке)
- `http://<host>:<port>/gallery/dataset` — галерея датасета (файлы из папок датасета, перемещение между классами)

**Горячие клавиши (разметчик `/`):**

| Клавиша | Действие |
|---------|----------|
| `1` | 1_resident |
| `2` | 2_delivery |
| `3` | 3_utilities |
| `→` | следующий без разметки |
| `←` | предыдущий |
| `U` | пропустить (unknown) |

**Галерея `/gallery` — множественный выбор:**

| Действие | Результат |
|----------|-----------|
| Клик по ✓ | выбрать / снять одну картинку |
| Ctrl+клик | то же |
| Shift+клик | диапазон в пределах секции (от последнего выбранного до текущего) |
| ☑ Все | выбрать все |
| ☐ Снять | снять выделение |

Диапазон Shift+клик ограничен визуальным порядком: если между двумя картинками есть уже
размеченные (они стоят в другой секции), в выделение они не попадут.

---

## Обучение Модели 1 (3_train_groups)

```bash
# Стандартный запуск (данные из .data/groups/v1/dataset)
./sh/train/4_train_groups.sh

# Явный путь и параметры
./sh/train/4_train_groups.sh --data .data/groups/v1/dataset --epochs 30

# Отключить коррекцию дисбаланса
./sh/train/4_train_groups.sh --no-class-weights --no-weighted-sampling
```

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `-Data` | `.data\groups\v1\dataset` | Папка датасета или zip |
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
.models/classify/v1_1.onnx            — ONNX (opset 18, один файл ~6 MB)
.models/classify/v1_1.json            — манифест: dataset_version, metrics
.models/classify/backbone.pt          — backbone для инициализации Модели 2
.output/train/3_train_groups/run/training_results.json
```

**Версионирование моделей** (логика в `src/ml/versions.py`):

| Датасет | 1-й прогон | 2-й прогон на том же датасете |
|---------|------------|-------------------------------|
| `.data/groups/v1` | `v1_1.onnx` | `v1_2.onnx` |
| `.data/groups/v2` | `v2_1.onnx` | `v2_2.onnx` |
| export.zip (без версии) | `v3.onnx` (legacy) | `v4.onnx` |

CLI:
```bash
python scripts/train/0_model_versions.py list
python scripts/train/0_model_versions.py activate v1_1
python scripts/train/0_model_versions.py info v1_1
```

**ONNX export:** `opset_version=18`, `dynamo=False` — один `.onnx` без внешнего `.onnx.data`.

**Batch size на сервере 2 GB RAM:** batch=32 работает, но активен swap; для следующих прогонов попробовать `--batch-size 16`.

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

```bash
# Первый цикл — backbone от Модели 1
./sh/train/5_train_residents.sh --data export.zip --backbone .models/classify/backbone.pt

# Последующие циклы — веса предыдущей версии
./sh/train/5_train_residents.sh --data export_full.zip --init-from .models/identify/v1.pt
```

### Трансфер между моделями

```
ImageNet weights
    ↓
fine-tune на датасете групп (1_resident / 2_delivery / 3_utilities)
    ↓
Модель 1  (.models/classify/v1_1.onnx)
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
    v1_1.onnx           — Модель 1: датасет v1, 1-й прогон (val_acc 79%)
    v1_1.json           — манифест (dataset_version, metrics)
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

models:
  detect:   .models/detect/yolov8n.onnx
  classify: .models/classify/v1_1.onnx
  # identify: .models/identify/v1.onnx
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
