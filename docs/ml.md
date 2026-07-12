# ML-пайплайн: классификация и идентификация

## Модели

Все модели в формате **ONNX** — единый рантайм, не требует PyTorch/TF в production.

| Задача | Модель | Размер | Скорость на CPU | Статус |
|--------|--------|--------|-----------------|--------|
| Детекция людей | YOLOv8n | ~13 MB | ~20–40 мс/кадр | **Готово** `.models/detect/yolov8n.onnx` |
| Классификация группы (Модель 1) | MobileNetV3-Small | ~6 MB | ~5–10 мс/crop | **Обучена** `.models/classify/v2_3.onnx` (датасет v2) → v3 в работе |
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
./sh/train/groups/1_1_dataset_groups_check.sh     # watch: .output/.../images → new/

# Терминал C — дедупликация внутри new/
./sh/train/groups/1_2_dataset_groups_check_new.sh # watch: new/
```

Полный цикл разметки и обучения:

```
1. Накопить кропы (watch-скрипты выше) → уникальные кропы в new/

2. Разметить:
   ./sh/train/2_label_ui.sh --unlabeled-only
   Порт — LABEL_UI_PORT в .env (по умолчанию 8750).
   Экраны: / (разметчик) ↔ /gallery (галерея сессии) ↔ /gallery/dataset.
   Галерея /gallery: SHIFT+click для выделения диапазона в секции.

3. Применить разметку в датасет:
   ./sh/train/groups/1_dataset_groups.sh apply --move
   → файлы из new/ → vN/dataset/1_resident/, vN/dataset/2_delivery/, … по меткам

4. Собрать новый датасет из всех источников:
   ./sh/train/groups/3_dataset_build.sh \
       --dataset   .data/groups/v1/dataset \
       --dataset   .data/groups/v2/dataset \
       --inference .data/groups/v1/inference/images \
       --inference .data/groups/v2/inference/images \
       --output    .data/groups/v3/dataset \
       --always-fill

5. Обучить:
   ./sh/train/groups/4_train_groups.sh --data .data/groups/v3/dataset --epochs 30
   → .models/classify/v3_1.onnx (+ v3_1.json манифест)

6. (опционально) Проверить во время обучения:
   python scripts/train/3a_eval_pt.py
```

Ручные операции с датасетом (при необходимости):

```bash
# Добавить кропы из прогона в new/ (с проверкой дублей против датасета)
python scripts/train/dataset_groups.py add \
    --src .output/pipeline/2_yolo_boxes_files/images \
    --dataset .data/groups/v2/dataset

# Статус датасета
python scripts/train/dataset_groups.py status --dataset .data/groups/v3/dataset
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

**Выход:** `.data/groups/v2/inference/`
```
inference/
  images/YYYYMMDD/
    1_resident/crop.jpg
    2_delivery/crop.jpg
    uncertain/crop.jpg      ← conf < CLASSIFY_CONF
    classifications.csv     ← накопительный (все прогоны за дату)
    labels.json             ← накопительный (пред. модель + ручная переразметка)
  meta/YYYYMMDD/
    run_params.json
```

По умолчанию файлы **перемещаются** из источника; `--copy` оставляет оригиналы.
Расширение файлов: `--ext jpg` (по умолчанию).

Порог уверенности: `CLASSIFY_CONF` из `.env` (дефолт `0.65`); ниже → класс `unknown`.

Для автоматического формирования `labels.json` по итогам инференса (совместимого с `2_label_ui` и `apply`):

```bash
# Терминал — labels из инференса (watch-режим)
./sh/train/groups/1_3_dataset_groups_inference_labels.sh

# Применить в датасет (вручную, после проверки)
./sh/train/groups/1_dataset_groups.sh apply \
    --labels .data/groups/v2/inference/images/YYYYMMDD/labels.json \
    --dataset .data/groups/v2/dataset --move
```

---

## Сборка нового датасета (итеративное улучшение)

После накопления инференса Модели 1 собирается новый датасет для переобучения.
Источники разбиты на стратегии по ценности данных.

### Пороги уверенности

```dotenv
# .env
CLASSIFY_CONF=0.65        # основной порог инференса (conf < → uncertain/)
CLASSIFY_CONF_HIGH=0.85   # второй порог: выше — псевдо-метки, ниже — серая зона
```

### Стратегии отбора

| # | Суть | Источник | Условие | Метка берётся из |
|---|------|----------|---------|-----------------|
| 1 | Ошибка с низкой уверенностью — исправлена вручную | `inference/YYYYMMDD/<class>/` | `labels.json` ≠ subdir, `conf < conf_high` | `labels.json` (ручная переразметка) |
| 2 | Модель не уверена — размечено вручную | `inference/YYYYMMDD/uncertain/` | файл есть в `labels.json` с валидным классом | `labels.json` (ручная разметка) |
| 3 | Серая зона — conf < conf_high, есть метка | `classifications.csv` | `conf < CLASSIFY_CONF_HIGH` И есть `labels.json` | `labels.json` |
| 5 | Уверенная правильная метка — псевдо-лейбл | `classifications.csv` | `conf ≥ CLASSIFY_CONF_HIGH` И pred = true | subdir / `labels.json` |
| 7 | Уверенная ошибка — наиболее ценные примеры | `classifications.csv` | `conf ≥ CLASSIFY_CONF_HIGH` И pred ≠ true | `labels.json` (из стрт 1 с высокой уверенностью) |
| 8 | Граничный случай между двумя классами | `classifications.csv` | `margin = conf_top1 − conf_top2 < порог` | `labels.json` |
| F | Дополнение минорных классов из предыдущего датасета | предыдущий датасет | new\_count < prev\_count | subdir датасета-источника |

Стратегии 1 и 7 выделяются автоматически из одного пула ошибок через `_split_errors_with_quota`:
- `conf ≥ conf_high` → стрт 7 (уверенная ошибка, всегда включается)
- `conf < conf_high` → стрт 1 (неуверенная ошибка, ограничивается дефицит-квотой по классу)

**Стрт 7** — самая ценная: модель уверена, но систематически ошибается. Без таких примеров переобучение не исправит ошибку.

**Стрт 3** покрывает весь диапазон `conf < conf_high` (включая `conf < conf_low`) — все неуверенные случаи, где есть ручная метка. Это же покрывает то, что раньше называлось «стрт 4».

**Стрт 8** (margin sampling): `margin = conf_top1 − conf_top2`. Малый margin — граничный случай между двумя классами. Пример:
```
[0.68, 0.63, 0.05, 0.04]  conf=0.68, margin=0.05  ← брать (граница 4_guest/1_resident)
[0.68, 0.15, 0.10, 0.07]  conf=0.68, margin=0.53  ← не брать (просто немного неуверен)
```

### Полная унификация: датасеты и инференс в единой системе

Предыдущие датасеты обрабатываются так же, как даты инференса:

```bash
# 1. Прогнать модель на датасете → inference-style каталог
python scripts/train/dataset_run_inference.py \
    .data/groups/v1/dataset \
    --output .data/groups/v1/inference/images/dataset
# Создаёт: hardlinks в pred_class/, classifications.csv, labels.json (true = исходные dirs)

# 2. Применить все стратегии (1,2,3,5,7,8) как для обычной даты
python scripts/train/dataset_v2_from_inference.py \
    .data/groups/v1/inference/images/dataset \
    --output .data/groups/v3/dataset \
    --dedup-min 0   # без временной дедупликации (не видео-поток)
```

Это позволяет единой системой извлекать из датасетов не только ошибки (стрт 4/7), но и
уверенные правильные примеры (стрт 5), неуверенные (стрт 3), uncertain (стрт 2).

### Фильтры

**Временно́е дедублирование** (стрт 5): не более 1 кропа на (камера, 15-минутное окно).
Применяется только к данным из инференса; для датасетов — `--dedup-min 0`.

**Квота ошибок** (стрт 1/7): `score = conf × (target / current_cls_size)`, где target — geomean
размеров классов в выходном датасете. Стрт 7 (conf ≥ conf_high) всегда включается;
стрт 1 (conf < conf_high) — до дефицита (target − current) штук на класс.

**Квота на класс** (`--max-per-class`): жёсткий лимит поверх всех стратегий.

### Скрипты сборки

```
sh/train/groups/
  3_dataset_run_inference.sh  — прогон модели на датасете → inference-style каталог
  3_dataset_from_inference.sh — стратегии 1,2,3,5,7,8: один каталог (дата или dataset)
  3_dataset_from_prev.sh      — стратегии 4(→3),7: устаревший вариант (без стрт 2,5)
  3_dataset_fill_minor.sh     — стратегия F: дополнение минорных классов
  3_dataset_build.sh          — мастер-скрипт (все источники)

scripts/train/
  dataset_run_inference.py    — прогон модели на датасете, создание inference-style dir
  dataset_v2_from_inference.py — стратегии 1,2,3,5,7,8
  dataset_v2_from_prev.py      — стратегии 4,7 (устаревший, оставлен для совместимости)
  dataset_fill_minor.py        — стратегия F
```

Полная сборка одной командой:
```bash
./sh/train/groups/3_dataset_build.sh \
    --dataset   .data/groups/v1/dataset \   # прогон модели + все стратегии
    --dataset   .data/groups/v2/dataset \   # можно несколько
    --inference .data/groups/v1/inference/images \  # все даты в каталоге
    --inference .data/groups/v2/inference/images \
    --output    .data/groups/v3/dataset \
    --always-fill                           # стрт F: объединить old ∪ new
```

`--dataset` заменяет старый `--prev`: запускает инференс модели, затем обрабатывает
все стратегии. `--prev` оставлен для обратной совместимости (только стрт 4,7).

### Структура данных

```
.data/groups/
  v1/
    dataset/              ← исходный датасет (не трогать)
    inference/
      images/
        20260705/         ← дата инференса: <class>/, uncertain/, classifications.csv, labels.json
        dataset/          ← создаётся dataset_run_inference.py (hardlinks + CSV + labels)
  v2/
    dataset/              ← v1 + новые примеры
    inference/
      images/
        20260708/ …
        dataset/
  v3/
    dataset/              ← текущая версия (собрана из v1+v2 датасетов + v1+v2 инференса)
      1_resident/
      2_delivery/
      3_utilities/
      4_guest/
      sources.csv         ← аудит: стратегия, исходный путь, conf, margin для каждого файла
```

---

## Полу-автоматическая разметка датасета жителей

### Проблема diff-фильтра

`filter_residents.py --min-diff 40` убирает похожие соседние кадры — это нужно для датасета,
но делает intra-person дифы (~49 медиана) неотличимыми от inter-person дифов (~61 медиана).
Поэтому прямая полу-авторазметка "по порогу дифа" не работает.

### Решение: сцены

Без diff-фильтра intra-person дифы 5–15, inter-person 30–80 — разделение чистое.
Сцена — непрерывный поток от одного человека (граница = большой разрыв по времени или diff).

### Версионирование

Рабочий каталог задаётся через `.env`:
```dotenv
RESIDENTS_VER=v1   # → .data/residents/v1/scene_pool/
```

Скрипты `1_b_1_collect_scenes.sh` и `1_b_2_propagate_scenes.sh` читают эту переменную автоматически.

### Пайплайн (4 шага)

**Шаг 1_b_1 — Собрать без diff-фильтра, разбить на сцены:**
```bash
./sh/train/residents/1_b_1_collect_scenes.sh
# Опции:
#   --scene-gap 120   # разрыв >120s → новая сцена
#   --scene-diff 50   # pixel-diff >50 → новая сцена (только по времени: --scene-diff 0)
#   --min-blur 40     # убираем совсем размытые
#   --dry-run         # только статистика, без копирования

# Выход: .data/residents/$RESIDENTS_VER/scene_pool/
#   *.jpg               — все кандидаты (без diff-фильтра)
#   scenes.json         — сцены: scene_id, cam, date, frames[], representative
#   representatives/    — по 1 резкому кадру на сцену → для ручной разметки
```

Входные каталоги прописаны явно в `SOURCES=(...)` внутри скрипта.
Для добавления новой даты — раскомментировать строку или добавить `--src`.

**Перенос существующих меток (опционально, перед шагом 2):**

Если есть уже размеченный пул (например, `.data/residents/v0/new_labeld/labels.json`),
метки можно перенести в scene_pool двумя способами:
1. По имени файла — прямое совпадение
2. По (cam, date, sod) — файл из старого пула попадает во временной диапазон сцены → метка присваивается representative сцены

После переноса запустить `1_b_2_propagate_scenes.sh` для распространения.

**Шаг 2 — Разметить representatives/ (по 1 кадру на сцену):**
```bash
./sh/train/2_label_ui.sh \
    --input   .data/residents/v1/scene_pool/representatives \
    --labels  .data/residents/v1/scene_pool/labels.json \
    --dataset .data/residents/v1/dataset \
    --unlabeled-only
# Кадры из representatives/ имеют вид: scene0042_cam01_9_d_...conf0.85.jpg
# Кнопки берутся из подпапок --dataset (реальные классы: 141_resident_man_1, ...)
# labels.json — в корне scene_pool, НЕ в representatives/
```

**Шаг 1_b_2 — Авторазметка внутри сцен:**
```bash
./sh/train/residents/1_b_2_propagate_scenes.sh
# Опции:
#   --intra-diff 25   # кадры с diff>25 от representative → не распространять
#   --dry-run         # только статистика

# Читает: scene_pool/scenes.json + scene_pool/labels.json
# Пишет: дополняет scene_pool/labels.json для всех кадров сцены
# Конфликт (2 метки в одной сцене) → сцена пропускается, выводится предупреждение
# Имена с префиксом scene0042_ (из representatives/) автоматически нормализуются
```

При необходимости шаги 2 и 1_b_2 повторяются для граничных / проблемных сцен.

**Шаг 1_b_3 — Финальный diff-фильтр для разнообразия в датасете:**
```bash
./sh/train/residents/1_b_3_filter_scenes.sh
# Дефолты: --min-diff 40, путь из RESIDENTS_VER
# Опции:
#   --min-diff 30     # другой порог
#   --dry-run
```

### Структура выхода

```
.data/residents/
  v1/
    scene_pool/
      *.jpg                — кандидаты (без diff-фильтра, все кадры)
      scenes.json          — сцены: scene_id, cam, date, frames[], representative
      representatives/     — scene0001_<name>.jpg  ← размечать только эти
      labels.json          — {"version":1, "labels":{...}}  ручные + авто-метки
    dataset/
      141_resident_man_1/  — после финального фильтра, готово для обучения
      141_resident_woman_1/
      ...
```

### Конфликты и граничные случаи

- **Несколько person_id в одной сцене** — `propagate_scenes.py` выводит предупреждение
  и пропускает сцену. Решение: уменьшить `--scene-diff` в шаге 1_b_1 и пересобрать.
- **Смена одежды в длинной сцене** — если diff не изменился, авторазметка правильная
  (одна сцена = один человек). Проверить визуально gallery в label_ui.
- **Несколько человек подряд без разрыва** — сцена может быть "загрязнена".
  `--intra-diff N` в шаге 1_b_2 отсечёт кадры с большим отклонением от representative.

---

## Скрипты обучения

**Модель 1 (классификация групп):**

| Скрипт | Назначение |
|--------|-----------|
| `sh/train/groups/1_1_dataset_groups_check.sh` | Watch: кропы из 2_yolo_boxes/images → new/ (дубли против датасета) |
| `sh/train/groups/1_2_dataset_groups_check_new.sh` | Watch: дедуп внутри new/ по имени файла |
| `sh/train/groups/1_3_dataset_groups_inference_labels.sh` | Watch: inference/images/YYYYMMDD/ → labels.json по структуре класс-каталогов |
| `sh/train/groups/1_dataset_groups.sh` | Ручное управление датасетом: `build / apply / check / add / status` |
| `sh/train/groups/3_dataset_run_inference.sh` | Прогон модели на датасете → inference-style каталог (`dataset/`) |
| `sh/train/groups/3_dataset_from_inference.sh` | Стратегии 1,2,3,5,7,8: один каталог (дата инференса или dataset) |
| `sh/train/groups/3_dataset_from_prev.sh` | Стратегии 4,7: устаревший (без стрт 2,5); оставлен для совместимости |
| `sh/train/groups/3_dataset_fill_minor.sh` | Стратегия F: дополнение минорных классов |
| `sh/train/groups/3_dataset_build.sh` | Мастер-скрипт сборки (`--dataset`, `--inference`, `--prev`, `--output`) |
| `sh/train/groups/4_train_groups.sh` | Обучение Модели 1 |

**Модель 2 (идентификация жителей) — Путь A (прямой сбор):**

| Скрипт | Назначение |
|--------|-----------|
| `sh/train/residents/1_a_1_collect_residents.sh` | Сбор сырого пула из датасета и инференса (1_resident + 4_guest) |
| `sh/train/residents/1_a_2_filter_residents.sh` | Фильтрация пула: conf, blur, >1 чел., покрытие, pixel-diff (`--min-diff 20`) |

**Модель 2 (идентификация жителей) — Путь B (полу-авторазметка через сцены):**

| Скрипт | Назначение |
|--------|-----------|
| `sh/train/residents/1_b_1_collect_scenes.sh` | Сбор без diff-фильтра + разбивка на сцены; входные каталоги прописаны явно в `SOURCES` |
| `sh/train/residents/1_b_2_propagate_scenes.sh` | Авторазметка кадров внутри сцен по 1 ручной метке; нормализует префикс `scene0042_` |
| `sh/train/residents/1_b_3_filter_scenes.sh` | Финальный diff-фильтр (`--min-diff 40`), путь из `RESIDENTS_VER` в `.env` |

**Модель 2 — общая цепочка:**

| Скрипт | Назначение |
|--------|-----------|
| `sh/train/residents/2_analyze_pool.sh` | Анализ пула: графики frames/window, conf, pixel-diff |
| `sh/train/residents/3_train_residents.sh` | Обучение Модели 2 → `.models/identify/v<N>.onnx` |

**Общее:**

| Скрипт | Назначение |
|--------|-----------|
| `sh/train/2_label_ui.sh` | Веб-разметчик кропов (группы и жители) |

---

## Веб-разметчик (2_label_ui)

```bash
# Группы (Модель 1)
./sh/train/2_label_ui.sh
./sh/train/2_label_ui.sh --input .output/pipeline/2_yolo_boxes_files/run_XXX
./sh/train/2_label_ui.sh --port 8789   # переопределяет LABEL_UI_PORT
./sh/train/2_label_ui.sh --probs       # показывать вероятности из classifications.csv

# Жители — Путь B (сцены, representatives/)
./sh/train/2_label_ui.sh \
    --input   .data/residents/v1/scene_pool/representatives \
    --labels  .data/residents/v1/scene_pool/labels.json \
    --dataset .data/residents/v1/dataset \
    --unlabeled-only
# Кнопки классов берутся из подпапок --dataset (141_resident_man_1, ...)
# labels.json поддерживает формат {"version":1,"labels":{...}} и плоский {name:class}
# Имена файлов вида scene0042_cam_01_... нормализуются — уже размеченные не показываются
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
# Стандартный запуск (данные из .data/groups/v2/dataset по умолчанию)
./sh/train/groups/4_train_groups.sh

# Явный путь и параметры
./sh/train/groups/4_train_groups.sh --data .data/groups/v3/dataset --epochs 30

# Отключить коррекцию дисбаланса
./sh/train/groups/4_train_groups.sh --no-class-weights --no-weighted-sampling
```

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--data` | `.data/groups/v2/dataset` | Папка датасета или zip |
| `--epochs` | `20` | Число эпох |
| `--batch-size` | `64` | Размер батча |
| `--lr` | `1e-3` | Learning rate |
| `--val-split` | `0.2` | Доля валидации (стратифицированная) |
| `--class-weights` | вкл | Взвешенная функция потерь |
| `--weighted-sampling` | вкл | WeightedRandomSampler |

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
.models/classify/v3_1.onnx            — ONNX (opset 18, один файл ~6 MB)
.models/classify/v3_1.json            — манифест: dataset_version, metrics
.models/classify/backbone.pt          — backbone для инициализации Модели 2
.output/train/3_train_groups/run/training_results.json
```

**Версионирование моделей** (логика в `src/ml/versions.py`):

| Датасет | 1-й прогон | 2-й прогон на том же датасете |
|---------|------------|-------------------------------|
| `.data/groups/v1` | `v1_1.onnx` | `v1_2.onnx` |
| `.data/groups/v2` | `v2_1.onnx` | `v2_2.onnx` |
| `.data/groups/v3` | `v3_1.onnx` | `v3_2.onnx` |
| export.zip (без версии) | `v4.onnx` (legacy) | `v5.onnx` |

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
```bash
~/miniconda3/envs/conda_video/bin/python scripts/train/3a_eval_pt.py
```
Показывает точность по каждому классу на случайной выборке из датасета.

---

## Идентификация жителей (Модель 2)

### Структура датасета

```
.data/residents/
  v0/
    new/          ← собранные кропы до разметки (временный буфер)
  v1/
    dataset/      ← размеченный датасет (папки по person_id)
      person_01/
      person_02/
      …
```

Источники кропов: `1_resident` + `4_guest` из датасета групп и инференса.
Гости включены — они тоже ходят в конкретные квартиры и идентифицируемы.

### Сбор и фильтрация кропов

```bash
# Шаг 1: собрать кропы
./sh/train/residents/1_collect_residents.sh
#   --out .data/residents/v0/new   (default)
#   --interval 1                   (1с окно на камеру, default)
#   --conf-delta 0.1               (из окна брать все кадры с conf ≥ max-0.1, default)
#   --max-persons 1                (только кадры где один человек, default)
#   --classes 1_resident 4_guest   (default)

# Шаг 1а: (опционально) анализ пула до дедупликации
./sh/train/residents/3_analyze_pool.sh
#   --interval 2   (размер окна для статистики)
#   --out .data/residents/v0/analysis
# → графики: frames_per_window, conf_distribution, pixel_diff

# Шаг 2: отфильтровать плохие кропы
./sh/train/residents/2_filter_residents.sh
#   --min-conf 0.5      (conf Model 1 из имени файла, default)
#   --min-blur 40       (резкость по дисперсии Лапласиана, default)
#   --yolo-persons 1    (max людей в кропе по YOLO, default; conf=0.12)
#   --min-coverage 0.15 (min доля кадра под bbox человека, default)
#   --min-diff 20       (pixel-diff фильтр: следующий кадр той же камеры/даты
#                        берётся только если mean-abs-diff от предыдущего ≥ N, default)
#   --min-body 0        (доля видимого тела по pose, default=отключён)
```

Фильтры убирают: кадры с несколькими людьми (YOLO пропустил слияние bbox),
motion blur, низкий confidence Модели 1, кропы где детектор не нашёл человека,
и похожие соседние кадры (`--min-diff`).

`--min-diff` — жадный greedy-алгоритм: кадры группируются по `(cam, date)`, сортируются
по времени. Следующий кадр добавляется только если mean-abs-diff от последнего оставленного
≥ порога. Работает глобально по дню, не ограничен окном из шага 1.

`--min-body` требует `yolov8n-pose.pt` (`.models/detect/`). Pose estimation даёт
долю видимых keypoints из 17 (COCO). Из-за вида камеры сверху даже хорошие кадры
дают ~35–65%, поэтому порог ставить не выше 0.25.

```bash
# Посмотреть что будет удалено без реального удаления:
./sh/train/residents/2_filter_residents.sh --dry-run
./sh/train/residents/2_filter_residents.sh --dry-run --min-diff 30
```

### Разметка

```bash
# Дефолт — жители (v0/new → v1/dataset)
./sh/train/2_label_ui.sh

# Явно указать пути
./sh/train/2_label_ui.sh --input .data/residents/v0/new --dataset .data/residents/v1/dataset
```

Классы (`person_01`, `person_02`, …) читаются из подпапок `v1/dataset/` автоматически.
Чтобы добавить нового человека — создать папку `person_NN/` и обновить страницу в браузере.

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
./sh/train/residents/5_train_residents.sh --data .data/residents/v1/dataset --backbone .models/classify/backbone.pt

# Последующие циклы — веса предыдущей версии
./sh/train/residents/5_train_residents.sh --data .data/residents/v1/dataset --init-from .models/identify/v1.pt
```

### Добавить нового жителя

```
1. Запустить 1_collect_residents.sh + 2_filter_residents.sh (подберут новые кропы)
2. Создать папку .data/residents/v1/dataset/person_NN/
3. Разметить новые кропы через 2_label_ui.sh
4. Переобучить Модель 2 на полном датасете (--init-from предыдущей версии)
```

До переобучения новый житель попадёт в `unknown_resident`.

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
    yolov8n.onnx        — детекция людей (пайплайн)
    yolov8n-pose.pt     — pose estimation (filter_residents: доля тела в кадре)
  classify/
    v2_3.onnx           — Модель 1 (текущая): датасет v2
    v3_1.onnx           — Модель 1 (новая, датасет v3, 10 015 файлов)
    v2_3.json, v3_1.json — манифесты (dataset_version, metrics)
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
  classify: .models/classify/v2_3.onnx
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
