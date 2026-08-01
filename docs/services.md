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

> **Актуально (реализовано в скриптах, а не в этом FastAPI-спеке):** классификация Модели 1 —
> **multi-label**, метки в формате v2 `{"version": 2, "labels": {img: [classes]}}` (набор классов
> на кроп; sigmoid + пороги). Псевдо-метки `skip/unknown/uncertain/new` исключаются из обучения.
> Подробно — раздел «Multi-label Модели 1 (датасет v4)» в [ml.md](ml.md).

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

## Windows (Watchdog)

`sh\system\watchdog.ps1` — следит за двумя скриптами-обёртками и перезапускает их при падении.

| Скрипт | Что запускает |
|--------|--------------|
| `sh\pipeline\1_motion_diff.ps1` | детекция движения с камер |
| `sh\transfer\2_send.ps1` | отправка дифф-кадров на Linux-сервер |

Watchdog определяет, запущен ли процесс, через `Win32_Process.CommandLine` (ищет имя `.ps1` файла). Каждые 30 секунд (настраивается `-CheckSec N`) — проверка, при отсутствии — немедленный рестарт. Лог: `.output\logs\watchdog.log`.

### Регистрация как системная задача (один раз)

```powershell
# От имени Администратора:
powershell -ExecutionPolicy Bypass -File "C:\Work\video\sh\system\watchdog.ps1" -Register
```

Создаёт задачу `VideoWatchdog`:
- **Триггеры:** BootTrigger (через 10 сек после старта системы) + EventTrigger (пробуждение из сна, EventID 1 Microsoft-Windows-Power-Troubleshooter).
- **LogonType = S4U** — задача выполняется **вне зависимости от входа пользователя** (headless, от имени пользователя, без пароля). Пайплайн авто-стартует по загрузке **без входа в Windows**. Для RTSP/HTTP ограничений сетевого токена S4U нет (у них своя авторизация — user/pass в URL, API-ключ).
- `MultipleInstancesPolicy = IgnoreNew` — повторный запуск игнорируется, дублей нет.

> **Почему S4U (важно):** раньше задача регистрировалась с `LogonType=InteractiveToken` — она стартовала только когда пользователь **вошёл** в Windows. Загрузка происходит ДО входа, поэтому на чистом ребуте (без логина) watchdog и весь пайплайн **не поднимались**. Работало лишь потому, что машина оставалась залогиненной. Исправлено — теперь S4U.
>
> **После обновления `watchdog.ps1` задачу нужно ПЕРЕ-регистрировать** на каждой машине — сам по себе `git pull` её не меняет, только `-Register` пересоздаёт:
> ```powershell
> powershell -ExecutionPolicy Bypass -File "<repo>\sh\system\watchdog.ps1" -Register   # от Администратора
> ```
> Проверить тип: `(Get-ScheduledTask VideoWatchdog).Principal` → должно быть `LogonType=S4U`.
>
> Состояние на 2026-07-18: **CAMERAS_1** — перерегистрирована (S4U, проверено headless-стартом). **CAMERAS_3, DEVELOP** — при следующем включении выполнить `-Register` (были простаивающими).

**CPU в статусе:** клиент `2_send` (`scripts/transfer/client.py`) фоновым потоком пишет свой CPU% в `.output/transfer_client/meta/<дата>/cpu.csv` (каждые 10с, как motion_diff в свой `meta/<дата>/cpu.csv`). `sh/status/status_win.ps1` читает его и показывает CPU у `2_send` в колонке статистики.

```powershell
# Удалить задачу:
.\sh\system\watchdog.ps1 -Unregister

# Запустить watchdog вручную (в текущем терминале, без регистрации):
.\sh\system\watchdog.ps1
```

### Ручной контроль через Task Scheduler

```
Win + R → taskschd.msc → Task Scheduler Library → VideoWatchdog
```

Там можно запустить/остановить/посмотреть историю запусков.

---

## Systemd (Linux)

Все постоянные процессы запускаются как systemd-сервисы. Установка:

```bash
./sh/system/setup_systemd.sh            # установить и запустить
./sh/system/setup_systemd.sh --with-data  # + data-pipeline (watch-скрипты обучения)
./sh/system/setup_systemd.sh --remove   # удалить
./sh/system/setup_systemd.sh --status   # статус
./sh/system/setup_systemd.sh --logs video-classify --follow  # логи
```

| Сервис | Скрипт | Описание |
|--------|--------|----------|
| `video-transfer` | `sh/transfer/1_start_server.sh` | Приём файлов с Windows (HTTP :8765) |
| `video-yolo` | `sh/pipeline/2_yolo_boxes_files.sh` | YOLO детекция людей → кропы |
| `video-classify` | `sh/pipeline/3_classify_groups.sh` | Классификация групп (Модель 1): 1_resident / 2_delivery / 3_utilities / 4_guest / uncertain |
| `video-smooth` | `sh/pipeline/3b_smooth_groups.sh` | Темпоральное сглаживание классов (Viterbi/HMM или окно); пишет сайдкар `smoothed/<date>/` (images/ не трогает); watermark для identify |
| `video-identify` | `sh/pipeline/4_identify_residents.sh` | Идентификация жителей (Модель 2): вход — 1_resident + 4_guest из classify (после сглаживания) |
| `video-smooth-identity` | `sh/pipeline/4b_smooth_identity.sh` | Темпоральное сглаживание идентификации (Модель 2) — ТО ЖЕ ядро (`smooth_core`), классы-жители из p_* колонок; сайдкар `residents/<ver>/inference/smoothed/<date>/`; ПОСЛЕ identify |

Опциональные (только во время обучения, `--with-data`):

| Сервис | Скрипт | Описание |
|--------|--------|----------|
| `video-data-check` | `sh/train/groups/1_1_dataset_groups_check.sh` | Дедупликация кропов с датасетом |
| `video-data-dedup` | `sh/train/groups/1_2_dataset_groups_check_new.sh` | Дедупликация внутри new/ |
| `video-data-labels` | `sh/train/groups/1_3_dataset_groups_inference_labels.sh` | Авто-labels из инференса |

Потоки данных:

```
Windows-клиент
    │  HTTP POST → :8765
    ▼
video-transfer  →  .output/transfer/diff/
                   .output/transfer/service/
    ▼
video-yolo      →  .output/pipeline/2_yolo_boxes_files/images/
                   (исходники удаляются, poll 60s)
    ▼
video-classify  →  .data/groups/v4/inference/images/{date}/
                   single/{1_resident,2_delivery,3_utilities,4_guest}/ multi/ uncertain/
                   classifications.csv  labels.json   (poll 60s)
    ▼
video-smooth    images/ НЕ трогает → сайдкар .data/groups/v4/inference/smoothed/{date}/:
                classifications_smoothed.csv (crop,…,smoothed_class[col4],rel,p_*), labels.json (v2 —
                самодостаточно для разметки: 2_label_ui.sh --smoothed …), smooth_state.json, smooth_viz/
                (Viterbi/HMM или бегущее окно; poll 120s)
    ▼
video-identify  вход: кропы со smoothed_class ∈ {1_resident,4_guest} из smoothed/{date}/CSV,
                jpg берётся из images/ по имени (fallback: физ. single/ + multi/ если сайдкара нет)
                ⏸ ждёт watermark smoothed/{date}/smooth_state.json: берёт дату, когда smooth догнал CSV
                  (SMOOTH_WAIT=1; fallback SMOOTH_WAIT_TIMEOUT=600с если smooth выкл)
             →  .data/residents/v1/inference/images/{date}/
                {person_id}/  unknown_resident/
                identifications.csv  labels.json   (poll 60s)
    ▼
video-smooth-identity   ТО ЖЕ ядро smooth_core, что у video-smooth, но для Модели 2.
                images/ НЕ трогает → сайдкар .data/residents/v1/inference/smoothed/{date}/:
                classifications_smoothed.csv (…smoothed_class[col4],rel,p_*), labels.json (v2),
                smooth_state.json, smooth_viz/
                (классы-жители из p_* колонок identifications.csv — открытый набор; poll 120s)
```

Версии каталогов (`v4`, `v1`) берутся из `.env`: `GROUPS_VER`, `RESIDENTS_VER`.

**Синхронизация smooth→identify.** Без неё identify успевает опознать «ложного резидента»
(ошибку классификатора) до того, как smooth пересчитает класс → неверная личность не откатывается
(identify идемпотентен, skip-if-exists). Поэтому identify обрабатывает дату только когда
`smoothed/{date}/smooth_state.json.csv_rows ≥` числа строк `classifications.csv`. Если сглаживание
выключено — выставить `SMOOTH_WAIT=0` в `.env` (или положиться на `SMOOTH_WAIT_TIMEOUT`: identify
пойдёт, когда CSV перестанет меняться дольше таймаута).

### Адаптивное замедление (ЦПУ)

Чтобы под нагрузкой не занимать все ядра, у сервисов есть адаптивный лимитер
(`AdaptiveRateLimiter`, `src/common/utils/adaptive_rate.py`): держит **долю работы**
(work / (work+sleep)) ≤ `*_ADAPT_HIGH`, засыпая между единицами работы. Цель — **~50% ЦПУ**
(раньше classify/identify упирались в 100% на пачке кропов).

| Сервис | env-префикс | Что троттлит | MAX_FPS |
|--------|-------------|--------------|---------|
| `video-yolo` | `YOLO_` | инференс между кадрами | `YOLO_MAX_FPS=1` |
| `video-classify` | `CLASSIFY_` | между кропами | `CLASSIFY_MAX_FPS=20` |
| `video-identify` | `IDENTIFY_` | между кропами | `IDENTIFY_MAX_FPS=20` |
| `2_send` (Windows) | `TRANSFER_` | между отправками | — |

`.env`: `*_ADAPT_HIGH=0.50` (порог «перегружен» → замедлить), `*_ADAPT_LOW=0.25`
(«недогружен» → ускорить), `*_ADAPT_FACTOR=2.0` (во сколько раз менять интервал),
`*_ADAPT_WINDOW=10` (окно оценки). `*_MAX_FPS=0` полностью выключает замедление.

**Сглаживание (`video-smooth`, `video-smooth-identity`)** адаптивного лимитера НЕ имеет — вместо него
в юнитах задан пониженный приоритет: `Nice=10` (уступает ЦПУ реальным стадиям пайплайна при конкуренции)
и `IOSchedulingClass=idle` (уступает диск; на текущем планировщике `mq-deadline` — no-op, форвард-совместимо
с BFQ). Само сглаживание лёгкое (~0.2с/дата), стоит только рендер viz (~5с на большую новую дату, CPU-bound,
≤1 ядро, редко) — `Nice` гарантирует, что этот всплеск не мешает transfer/yolo/classify/identify.

> **Подхват `.env` (важно):** `.sh`-обёртки сервисов сорсят `.env` ОДИН раз при старте.
> После правки `*_ADAPT_*`/`*_MAX_FPS` в `.env` нужен `sudo systemctl restart video-<svc>` —
> иначе служба работает со старыми значениями (python берёт их из окружения `.sh`).
> Изменения самого КОДА (дефолты, логирование) подхватываются САМИ: `.sh` (while-true)
> переинвокает python каждый цикл, а python читает `.py` заново.

Логи через journald:
```bash
journalctl -u video-classify -f          # live
journalctl -u video-classify -n 200      # последние 200 строк
journalctl -u video-classify --since "1h ago"
```

Управление отдельным сервисом:
```bash
sudo systemctl restart video-classify
sudo systemctl status  video-classify
sudo systemctl stop    video-yolo
```

---

## Статус-отчёт (`sh/status/status.sh`)

Мастер-скрипт собирает состояние всех машин: Linux-сервер + Windows-клиенты. Выводит
таблицу в терминал и сохраняет `.md` в `.output/status/<дата>/`. В заголовке — версия
проекта (из `VERSION`). Запуск: `bash sh/status/status.sh` (см. [remote.md](remote.md)).

**Как опрашиваются узлы:**
- **Linux** — локально, если скрипт запущен на сервере (определяется по локальному
  Tailscale-IP == `TS_SERVER`); иначе по SSH (`TS_SERVER_USER`/`TS_SERVER_REPO`). Значит
  `status.sh` можно запускать с любой машины. Системные метрики — `scripts/utils/sys_stats.py`.
- **Windows** — всегда по SSH: `ssh <user>@<ip> powershell -File <repo>\sh\status\status_win.ps1`.
  Путь репо каждой машины — `TS_*_REPO` в `.env`. Имена машин в таблице — алиасы `TS_*_LABEL`, не hostname.

**Колонки «Вход» / «Выход»** — состояние входного/выходного каталога сервиса:
- дерево путей (родитель + дочерние, `├─ └─`); ниже — `N файл. · M кат.` + дата последнего файла;
- **каталоги инференса** (`…/inference/images`) — вместо счётчика **разбивка по датам**
  (`YYYYMMDD: N` + `ИТОГО (K дат)`); не-даты (напр. залётный `dataset/`) — строкой
  `[!] <имя>: N (не дата)`, чтобы сразу видеть лишнее;
- **`—`** = файлов нет → очередь не копится (пайплайн успевает). Накопление = сервис отстаёт.
- у `video-transfer` «Вход» — реальные машины-клиенты из логов (IP из `POST /file` → метка `TS_*`),
  иначе адрес прослушивания `TRANSFER_HOST:TRANSFER_PORT`.

**Последняя колонка** — статистика за адаптивное окно (10м → 60м → 24ч), 3 строки:
1. `N прогонов (X кадров/файлов)` — см. ниже;
2. **`счёт/кадр min/avg/max мс`** — ЧИСТОЕ время вычисления кадра (только счёт, без сна
   адаптивного лимитера и батч-оверхеда) → видно, успевает ли ЦПУ; min/max по окну, avg взвешен
   по кадрам. Источник — строка лога `счёт/кадр: …` (пишут пайплайн-скрипты через
   `camera_run.compute_per_frame_log`). У `video-transfer` вместо этого `приём min/avg/max с` (I/O приёма
   файла), у Windows `2_send` — `отправка min/avg/max с` (I/O отправки). Все три — ЧИСТОЕ время операции,
   единый формат `ярлык мин/ср/макс ед` (3 значения); idle-интервалы (с простоем) **не показываются**;
3. **`цпу% (мин/ср/макс/посл) (цпу)`** — из `cpu.csv` сервиса, тем же фолбэком окон.

Первая строка ячейки — **единообразно `N прогонов (X ...)`**, но `X` и что такое «прогон»
зависят от типа стадии (важно: `N` — не число файлов; один прогон обрабатывает пачку):
- **Батч-стадии** `video-yolo/classify/identify` — `N прогонов (X кадров)`: `N` — poll-циклы
  (лог `Готово. Время`), `X` — **обработанные кадры** (сумма из лога `1 батч (X кадров)`, который
  пишут сами скрипты). `X` считает ЛЮБУЮ обработку, даже когда ничего не найдено (напр. yolo
  `10 прогонов (0 кадров)`).
- **Непрерывный приём** `video-transfer` — `N прогонов (X файлов)`: `N` — всплески приёма
  (события, разделённые паузой > `BURST_GAP_SEC`), `X` — принятые файлы.
- **Сглаживание** `video-smooth` / `video-smooth-identity` — идемпотентные писатели сайдкара
  `smoothed/<date>/` (без по-кадрового тайминга): «Вход» — `images/` инференса соответствующей
  модели, «Выход» — каталог `smoothed/` (виз-конкаты + CSV). Состояние читается по лог-колонкам:
  `сглажено …` (работа) / `Изменений нет (K дат пропущено)` (skip — CSV не рос, конфиг не менялся).
- **Windows** (`status_win.ps1`): `motion_diff` — обычно `1 прогон (X файлов)` (непрерывный цикл,
  X = сохранённые кадры с движением), но при рестарте(ах) в окне — `N прогонов` (N = число стартов
  «Порог:»; читаются 2 свежих date-каталога, т.к. при рестарте/полуночи строки прогона попадают в
  разные каталоги); `2_send` — `N прогонов (X файлов)` (N = батчи отправки `[в батче 1/K]`).

**Дата в колонках** берётся из времени записи файла/лога, а не «сегодня» — выключенная
машина показывает реальную дату последней активности и нулевые счётчики, а не «фантомную» свежесть.

## Конфигурация (.env)

Модели и пороги задаются в `.env`:

```dotenv
CLASSIFY_MODEL=.models/classify/v3_1.onnx
IDENTIFY_MODEL=.models/identify/v5.onnx
DETECT_MODEL=.models/detect/yolov8n.onnx

CLASSIFY_CONF=0.65
IDENTIFY_CONF=0.70
YOLO_CONF=0.25
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
