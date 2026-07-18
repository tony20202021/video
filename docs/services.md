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
| `video-identify` | `sh/pipeline/4_identify_residents.sh` | Идентификация жителей (Модель 2): вход — 1_resident + 4_guest из classify |

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
video-classify  →  .data/groups/v3/inference/images/{date}/
                   1_resident/ 2_delivery/ 3_utilities/ 4_guest/ uncertain/
                   (poll 60s)
    ▼
video-identify  вход: 1_resident/ + 4_guest/
             →  .data/residents/v1/inference/images/{date}/
                {person_id}/  unknown_resident/
                identifications.csv  labels.json
                (poll 60s)
```

Версии каталогов (`v3`, `v1`) берутся из `.env`: `GROUPS_VER`, `RESIDENTS_VER`.

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
- **`—`** = файлов нет → очередь не копится (пайплайн успевает). Накопление = сервис отстаёт.
- у `video-transfer` «Вход» — реальные машины-клиенты из логов (IP из `POST /file` → метка `TS_*`),
  иначе адрес прослушивания `TRANSFER_HOST:TRANSFER_PORT`.

**Последняя колонка** — статистика за адаптивное окно (10м → 60м → 24ч), тайминг
`1кадра` (мин/ср/макс/посл), `цпу%` (мин/ср/макс/посл). CPU — из `cpu.csv` сервиса
с тем же фолбэком окон (Windows: и motion_diff, и 2_send пишут свои cpu.csv).

Первая строка ячейки — счётчик за окно, но он **разный по смыслу**:
- `video-transfer` — `N×` = принятые **кадры** (по одному событию `[recv]` на файл).
- `video-yolo/classify/identify` — `N прогонов (X файлов)`: `N` — это **poll-циклы**
  (батчи, лог `Готово. Время`), а `X` — реально произведённые выходные `*.jpg` за то же окно
  (из `out_dir` по mtime). Важно: `N` — не число файлов; один прогон обрабатывает пачку,
  и большинство прогонов yolo — «нет детекций» (motion ≠ люди). Поэтому `N` мал, а `X` показывает
  фактический выход (напр. `9 прогонов (1ч) (330 файлов)`).

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
