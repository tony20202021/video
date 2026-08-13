# Камеры и сетевой доступ

## Скрипты: сравнительная таблица

| Скрипт | Что делает | Требует камеру | Выход | Когда запускать |
|--------|-----------|:--------------:|-------|-----------------|
| `1_scan_cameras.py` | Сканирует подсеть по портам 554/8899/34567, определяет дубли по MAC | Нет | `.output/1_scan_cameras/scan_<UTC>/report.json` | Первый раз при настройке, чтобы найти IP камер |
| `2_probe_channels.py` | Перебирает все комбинации channel×stream по RTSP, сохраняет кадры | Да | `.output/2_probe_channels/probe_<UTC>/` — кадры + report.json | После нахождения IP, чтобы подобрать правильный channel/stream |
| `3_verify_cameras.py` | Проверяет URL из `.env`, читает один кадр с обрезкой | Да | `.output/3_cam_verify/cam_verify_<UTC>/` — кадры + report.json | Проверить что `.env` настроен правильно |
| `4_motion_diff_low.py` | Цикл: детектирует движение (frame diff по LOW), сохраняет LOW-кадры и heartbeat | Да | `.output/cameras/4_motion_diff_low/run_<ts>/` — baseline + motion + heartbeat | Накопление кадров без ML; отладка порога |
| `5_1_diff_yolo_boxes_low.py` | То же что 4, но после детекции движения прогоняет YOLOv8n и сохраняет только кадры с людьми | Да + модель | `.output/cameras/5_1_diff_yolo_boxes_low/run_<UTC>/` — baseline + кадры с людьми (bbox) | Запускать постоянно для детекции людей |
| `5_2_yolo_boxes_files.py` | Офлайн-переобработка: YOLO-детекция на уже сохранённых diff-кадрах прогонов 4 и 5_1 | Нет (файлы) | `.output/cameras/5_2_yolo_boxes_files/run_<ts>/` — детекции, timeline, cpu_chart | Ретроспективный анализ накопленных прогонов |
| `6_2_classify_groups_files.py` | Берёт кропы 5_2, прогоняет GroupClassifier (Модель 1) | Нет (файлы) | `.output/cameras/6_2_classify_groups_files/run_<ts>/` | Классификация кропов по группам |
| `6_3_identify_residents_files.py` | Берёт кропы `1_resident/` из 6_2, прогоняет PersonIdentifier (Модель 2) | Нет (файлы) | `.output/cameras/6_3_identify_residents_files/run_<ts>/` | Идентификация конкретных жителей |

**Порядок первичной настройки:** `1` → `2` → `3` → `4` и/или `5_1`

**Ключевые отличия live (4, 5_1) vs офлайн (5_2, 6_2, 6_3):**
- `4_motion_diff_low` сохраняет любое движение без ML — только LOW-поток, легковесный
- `5_1_diff_yolo_boxes_low` запускает YOLOv8n на каждый motion-кадр — тяжелее, но фильтрует только людей; требует `.models/detect/yolov8n.onnx`
- `5_2` обрабатывает накопленные прогоны 4 или 5_1 офлайн, вырезает кропы YOLO
- `6_2` берёт кропы `5_2` → GroupClassifier (Модель 1) → `classified/1_resident/`, `classified/2_delivery/`, …
- `6_3` берёт `classified/1_resident/` из `6_2` → PersonIdentifier (Модель 2) → `classified/<person_id>/`
- Оба live-скрипта можно запускать одновременно, но они открывают одни и те же RTSP-потоки

---

## Характеристики

**Модель:** iCSee, уличная поворотная PTZ, 8MP (2048×1080 UWHD), H.264/H.264+/H.265/H.265+, IP66

**Маркировка на коробке / OEM:** Model **A31** (использовать при поиске прошивки, RTSP-путей и обсуждений по железу).

**Розница (эталонная карточка):** Ozon арт. [1385832177](https://www.ozon.ru/product/kamera-videonablyudeniya-wifi-icsee-ulichnaya-povorotnaya-8mp-belaya-1385832177/) — «Камера видеонаблюдения wifi, ICsee, уличная, поворотная, 8МП, белая».

**Объективы (реальный экземпляр A31):** **два сенсора** в одном корпусе — **фиксированный** (широкий обзор) и **PTZ** (поворотный блок). Карточка магазина часто перечисляет одно «макс. разрешение» на весь блок; для интеграции важны **два логических канала** в прошивке, а не один.

- **Субпоток** `stream=1` vs **основной** `stream=0` — это качество **внутри одного канала** (`channel=N`), а не переключение между фиксированной и PTZ-линзой.
- Две **раздельные картинки** обычно берут как **два RTSP URL** с **разным `channel`** на тот же IP (нумерация задаётся прошивкой: часто `1` и `2`, иногда `0` и `1` — см. ниже). Если вместо двух потоков отдаётся **один склеенный кадр**, остаётся разрезание по половинам кадра в софте или поиск альтернативного URL в веб-интерфейсе камеры.

**Аудио / прочее:** двусторонняя связь, ИК до ~30 м, microSDXC, датчики (в т.ч. шум), автослежение, Alexa (по карточке). Питание 12 В (по карточке).

**Количество в проекте:** 4 камеры (2 модели одного производителя)

**Подключение:** Wi‑Fi; на данной карточке также указан **RJ‑45 (Ethernet)** — при наличии порта на конкретном экземпляре предпочтительнее кабель для стабильного RTSP.

---

## Топология сети

```
[Камера 1] ──── [Домашний роутер] ──── [Сервер разработки, Windows]
                       │
                  Интернет
                       │
[Камеры 2–4] ── [Внешний роутер] ─────────────────────────────────
                       │
                  Интернет
                       │
                 [Прод-сервер, Linux] ←── БД MongoDB
```

- Камера 1 — тот же роутер, что и сервер разработки (прямой доступ по IP)
- Камеры 2–4 — за внешним роутером (нужен port forwarding или VPN)

---

## RTSP-поток

Камеры поддерживают RTSP напрямую. Формат URL:

```
rtsp://<IP>:554/user=<USER>&password=<PASSWORD>&channel=1&stream=0.sdp?real_stream
```

- `stream=0` — основной поток (макс. разрешение канала, до ~25 fps) — для полноразмерных кадров (архив, ML)
- `stream=1` — субпоток (уменьшенное разрешение того же канала) — для лёгкого анализа (меньше нагрузка на CPU)

**`4_motion_diff_low.py`** работает только с **`CAM_<stem>_URL`** — субпоток (часто `stream=1`), детекция движения и сохранение. `CAM_<stem>_HI_URL` игнорируется.

**Два объектива → два URL в `.env`:** один и тот же `rtsp://<IP>:554/...`, отличается только **`channel`** (и при необходимости параллельно подобрать `stream`). Имена переменных — **`CAM_<stem>_URL`**, где `stem` — любой непустой идентификатор (цифры, хвост вроде `_9_U` / `_10_D` для IP и половины склейки). Пример с парой низкий/высокий поток для одной склейки (один `channel`, разный `stream`):

```text
CAM_01_9_U_URL=rtsp://<IP>:554/user=...&password=...&channel=0&stream=1.sdp?real_stream
CAM_01_9_U_HI_URL=rtsp://<IP>:554/user=...&password=...&channel=0&stream=0.sdp?real_stream
CAM_01_9_D_URL=rtsp://<IP>:554/user=...&password=...&channel=0&stream=1.sdp?real_stream
CAM_01_9_D_HI_URL=rtsp://<IP>:554/user=...&password=...&channel=0&stream=0.sdp?real_stream
CAM_01_URL=rtsp://<IP>:554/user=...&password=...&channel=1&stream=1.sdp?real_stream
```

**Один RTSP, картинка склеена из двух линз:** устройство A31 (проверено на `<ip>`) возвращает **один склеенный кадр 2304×2592 на всех channel=0..3** — две камеры расположены вертикально. HTTP snapshot (`/webcapture.jpg`, `/snapshot.jpg` и пр.) возвращает HTTP 400. Используйте один URL (`channel=0&stream=1`) и обрезку: **`MOTION_CROP_REL`** или **`CAM_<stem>_CROP_REL`** (доли **x,y,w,h**) — см. раздел про `4_motion_diff_low.py` ниже.

**Важно:** переменные **`CAM_*_HI_URL`** не являются отдельными «камерами» в `.env` (в список для скриптов не попадают — только **`CAM_*_URL`**). Пара задаётся так: **`CAM_<stem>_URL`** (субпоток) + **`CAM_<stem>_HI_URL`** (main).

Если второй канал не открывается — перебрать **`channel=0`**, **`channel=1`**, **`channel=2`** (разные партии прошивок нумеруют по-разному). Два потока с одной камеры иногда стабильнее открывать **последовательно** (второй `VideoCapture` после закрытия первого или в отдельном процессе), если прошивка не любит два одновременных клиента. Справочник по вариантам путей: [iSpy — icsee](https://ispyconnect.com/camera/icsee); про HTTP-снимок с номером канала у линеек XM см. [ansice — XM RTSP / snapshot](https://www.ansice.net/en-ch/blogs/installation-wiring-and-setting/the-xm-series-and-ts-series-rtsp-url-and-image-capture-url-and-the-network-ports).

Порты камеры:

| Протокол | Порт  |
|----------|-------|
| RTSP     | 554   |
| HTTP     | 80    |
| ONVIF    | 8899  |
| DVRIP    | 34567 |

---

## Результаты работы скриптов (`.output/`)

Скрипты 2–5 сохраняют результаты в `.output/` (в `.gitignore`):

```
.output/
  1_scan_cameras/
    scan_<UTC>/
      report.json        — найденные устройства, MAC, дубли, suggested URLs
  2_probe_channels/
    probe_<UTC>/
      rtsp_ch0_st0.jpg   — кадр channel=0, stream=0 (основной поток)
      rtsp_ch0_st1.jpg   — кадр channel=0, stream=1 (субпоток)
      ...
      report.json        — все результаты RTSP + HTTP
  3_cam_verify/
    cam_verify_<UTC>/
      cam_01_9_u_frame.jpg   — кадр после обрезки (CAM_01_9_U_URL)
      cam_01_9_d_frame.jpg
      report.json
  4_motion_diff_low/
    run_<UTC>/
      images/
        <cam>/                         — по одному подкаталогу на камеру
          diff/
            <cam>_<UTC>_diff<N>.jpg    — LOW-кадр при обнаружении движения
          <cam>_<UTC>_baseline.jpg     — baseline при старте
          <cam>_<UTC>_heartbeat.jpg    — периодический снимок (по таймеру)
      frames.csv, diffs.csv, saves.csv, pts.csv
      cpu.csv                          — 5 колонок: mono_s, ts_msk, cpu_pct, freq_mhz_pdh, freq_mhz_step
      charts.png, pts_chart.png       — по умолчанию рендерит СЕРВЕР из CSV (MOTION_RENDER_CHARTS=0)
      run_stats.json, run_params.json, run.log
  5_1_diff_yolo_boxes_low/
    run_<UTC>/
      images/
        <cam>/
          diff/
            <cam>_<UTC>_diff<N>.jpg   — raw motion-кадр (до YOLO)
          <cam>_<UTC>_baseline.jpg
          <cam>_<UTC>_heartbeat.jpg
          <cam>_<UTC>_p<N>.jpg        — кадр с N людьми (bbox нанесены)
          crops/                       — вырезанные кропы по bbox
      frames.csv, diffs.csv, saves.csv, pts.csv
      cpu.csv                          — 5 колонок (см. выше)
      charts.png, pts_chart.png       — по умолчанию рендерит СЕРВЕР из CSV (MOTION_RENDER_CHARTS=0)
      run_stats.json, run_params.json, run.log
  5_2_yolo_boxes_files/
    run_<UTC>/
      <run_name>/
        <cam>/
          <img>_yolo.jpg              — кадр с bbox (только при детекции)
          crops/
      detections.csv                  — все детекции: image_ts, run, cam, file, x1..y2, conf
      timeline_chart.png              — события из входных прогонов + новые YOLO
      cpu.csv, cpu_chart.png          — загрузка ЦПУ + YOLO timing + частота
      yolo_timing.csv                 — per-image: mono_s, inference_ms, sleep_ms
      run_stats.json, run_params.json, run.log
```

**Формат `<UTC>` в именах файлов:** `YYYYMMDD_HHMMSS_ffffff`, где последние 6 цифр — микросекунды (`%f` в Python). Пример: `cam_01_9_d_20260418_111316_116613_baseline.jpg` → снято 2026-04-18 в 11:13:16.116613 UTC. Микросекунды нужны для уникальности при нескольких кадрах в одну секунду.

**Время = момент СЪЁМКИ кадра (PTS), не обработки.** `1_motion_diff.py` штампует имя по PTS кадра
(`CAP_PROP_POS_MSEC`), привязанному к wall-clock ПК якорем `(pts0, pc0)` на URL (сброс при реконнекте):
`время = pc0 + (pts − pts0)`. Иммунно к задержке доставки/столлам RTSP — буферизованный после столла кадр
получает своё истинное время, а не время обработки (иначе непрерывное действие ~1-2с показывалось как +6.7с).
Фолбэк на wall-clock при недоступном PTS (`pts ≤ 0`). Формат имени не меняется — только значение точнее.
Деплой синхронно со сменой версии модели (меняется семантика `ts_epoch` → сегментация визитов в 3b/4b).

**Что пишется по порогу, что без:** heartbeat-кадры (`*_heartbeat.jpg`) записываются по таймеру независимо от движения — если сцена статичная, они будут одинаковые. Кадры движения (`*_motion.jpg` / `*_p<N>.jpg`) — только когда `diff > threshold`.

---

## IP-адреса — как узнать

1. **Через приложение iCSee**: Device → Device Info → IP Address
2. **Через роутер**: DHCP-таблица в интерфейсе роутера, MAC-адрес камеры начинается на `DC:` или `AC:` (производитель XMTech)
3. **nmap**:
   ```bash
   nmap -p 554,8899,34567 192.168.X.0/24
   ```
   Открытые порты 554 или 8899 = iCSee камера

---

## Логин и пароль

По умолчанию: `admin` / (пустой) или `admin` / `admin`

Задаётся при первой настройке в приложении iCSee. Если неизвестен — сброс физической кнопкой на камере.

**Хранение** в проекте — файл `.env` (в `.gitignore`):

```
CAM_01_URL=rtsp://192.168.1.XXX:554/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream
CAM_02_URL=rtsp://<external_ip>:8741/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream
CAM_03_URL=rtsp://<external_ip>:8742/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream
CAM_04_URL=rtsp://<external_ip>:8743/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream
```

### Перебор каналов/стримов (`2_probe_channels.py`)

Скрипт `scripts/cameras/2_probe_channels.py` перебирает все комбинации `channel × stream` по RTSP (3 формата URL на комбинацию) и несколько HTTP-snapshot URL для XM-чипов. Credentials автоматически берутся из первой `CAM_<stem>_URL` в `.env`.

```bash
python scripts/cameras/2_probe_channels.py
python scripts/cameras/2_probe_channels.py --ip <ip> --user <user> --password <password>
python scripts/cameras/2_probe_channels.py --channels 0 1 2 3 --streams 0 1
python scripts/cameras/2_probe_channels.py --http-only
```

Результаты — `.output/probe_channels/probe_<UTC>/`: кадры `rtsp_ch0_st0.jpg` и `report.json`.

---

### Проверка подключения (`3_verify_cameras.py`)

Скрипт `scripts/cameras/3_verify_cameras.py` загружает `.env`, для каждой переменной **`CAM_<stem>_URL`** (напр. `CAM_01_URL`, `CAM_01_9_U_URL`, `CAM_01_10_D_URL`) с реальным `rtsp://` открывает поток, читает **один кадр** и собирает **свойства потока** (разрешение, fps, backend OpenCV и т.д.). Переменные **`CAM_*_HI_URL`** скрипт **не** опрашивает — при необходимости проверьте главный поток отдельно (временно подставив URL в тест или второй вызов). Обрезка: **`MOTION_CROP_REL`**, **`CAM_<stem>_CROP_REL`**, **`--crop-rel`** — как в `4_motion_diff_low.py`; в JPEG и `report.json` — кадр **после** обрезки, плюс `crop_rel` / `frame_shape_before_crop`. Плейсхолдеры вроде `<external_ip>` пропускаются. Результаты — `.output/cam_verify/cam_verify_<метка>/`: `report.json` и `cam_01_9_u_frame.jpg` и т.д. (имя файла из stem).

Из корня репозитория:

```bash
python scripts/cameras/3_verify_cameras.py
python scripts/cameras/3_verify_cameras.py --tcp
python scripts/cameras/3_verify_cameras.py --crop-rel 0,0,1,0.5
```

Флаг `--tcp` включает RTSP поверх TCP — полезно через NAT или нестабильный Wi‑Fi. Поиск камер в локальной сети по портам — отдельно `scripts/cameras/1_scan_cameras.py`.

### Накопление кадров движения (`4_motion_diff_low.py`)

Скрипт `scripts/cameras/4_motion_diff_low.py` — только LOW-поток (`CAM_<stem>_URL`).

Несколько логических камер с **одинаковым** `CAM_*_URL` (например U и D с одной склейки) открывают **один** `VideoCapture`: один `read()` за такт, обрезки применяются отдельно.

При старте: читает первый годный LOW-кадр, сохраняет как baseline, задаёт внутреннее состояние детектора. Порог: **`MOTION_DIFF_THRESHOLD`** / **`--threshold`**. Пульс: **`MOTION_HEARTBEAT_SEC`** / **`--heartbeat-sec`**.

**Склеенный кадр:** обрезка в долях **x,y,w,h**. Глобально **`MOTION_CROP_REL`**, для одной переменной — **`CAM_<stem>_CROP_REL`**. Горизонтальная: сверху **`0,0,1,0.5`**, снизу **`0,0.5,1,0.5`**.

```bash
python scripts/cameras/4_motion_diff_low.py
python scripts/cameras/4_motion_diff_low.py --tcp --threshold 12
python scripts/cameras/4_motion_diff_low.py --heartbeat-sec 300
python scripts/cameras/4_motion_diff_low.py --crop-rel 0,0,1,0.5
python scripts/cameras/4_motion_diff_low.py --duration 3600   # остановиться через 1 час
```

### Детекция людей (`5_1_diff_yolo_boxes_low.py`)

Скрипт `scripts/cameras/5_1_diff_yolo_boxes_low.py` использует тот же motion-цикл (только LOW-поток), но после обнаружения движения запускает **YOLOv8n ONNX** (класс 0 — person) непосредственно на LOW-кадре. Сохраняет только кадры, на которых обнаружен хотя бы один человек; на сохраняемый кадр наносит **bounding boxes** с confidence. Имя файла: `<cam>_<UTC>_p<N>.jpg` (N — количество людей).

Перед первым запуском скачать модель (~6 MB):

```bash
mkdir -p models
wget -P models/ https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx
```

```bash
python scripts/cameras/5_1_diff_yolo_boxes_low.py
python scripts/cameras/5_1_diff_yolo_boxes_low.py --conf 0.25 --threshold 3.3
python scripts/cameras/5_1_diff_yolo_boxes_low.py --model .models/detect/yolov8n.onnx --tcp
```

Общие утилиты motion-цикла (фильтрация URL, frame diff, проверка кадра HEVC, открытие потока) вынесены в `src/common/utils/motion_utils.py`.

#### Параметры YOLO (`.env`)

| Переменная | Дефолт | Описание |
|------------|--------|----------|
| `YOLO_CONF` | `0.35` | Порог confidence (снизить до 0.25 для съёмки сверху/нестандартных ракурсов) |
| `YOLO_NMS` | `0.45` | IoU порог NMS — выше = меньше слияния боксов (0.55 для плотных сцен) |
| `YOLO_MAX_FPS` | `2.0` | Макс. частота YOLO на одну камеру, 0 = без ограничений. Ограничивает пиковую нагрузку CPU при активном движении. |

Текущие рабочие значения (откалиброваны под коридор, съёмка сверху):
```
YOLO_CONF=0.25
YOLO_NMS=0.55
YOLO_MAX_FPS=1
MOTION_DIFF_THRESHOLD=3.3
```

Параметры читаются из `.env` напрямую. Приоритет: явный параметр CLI → `.env` → встроенный дефолт.

#### Выходные файлы прогона

После завершения прогона в директории `run_<UTC>/` появляются:

| Файл | Описание |
|------|----------|
| `run_params.json` | Параметры запуска (модель, conf, nms, threshold, ...) |
| `run_stats.json` | Метрики качества: duration, fps, кадры ok/bad, реконнекты, интервалы, save counts |
| `charts.png` | Интервалы кадров + дифы + сохранения + CPU по времени. **Рендер по умолчанию на СЕРВЕРЕ** (не грузим камеру): `status.sh` тянет CSV и строит через `1_motion_diff.py --regen-from`. На камере создаётся только при `MOTION_RENDER_CHARTS=1` |
| `pts_chart.png` | Метки времени (mono/wall/PTS) + дрейф + CPU внизу. См. `charts.png` — рендер на сервере из CSV |
| `frames.csv`, `diffs.csv`, `saves.csv`, `pts.csv`, `cpu.csv` | Сырые данные для анализа (**единственный выход motion_diff при `MOTION_RENDER_CHARTS=0`**; из них сервер строит графики) |

#### Скрипты запуска

| Файл | ОС | Назначение |
|------|-----|-----------|
| `sh/pipeline/1_motion_diff.ps1` | Windows-камера | захват дифф-кадров + YOLO; отправляет кропы на сервер через `sh/transfer/2_send.ps1` |
| `sh/pipeline/2_yolo_boxes_files.sh` | Linux-сервер | YOLO-детекция на принятых файлах (watch + delete-after) |
| `sh/pipeline/3_classify_groups.sh` | Linux-сервер | классификация групп (Модель 1) |
| `sh/pipeline/4_identify_residents.sh` | Linux-сервер | идентификация жителей (Модель 2) |

```powershell
# Windows: запуск захвата (1_motion_diff.ps1 + watchdog)
.\sh\pipeline\1_motion_diff.ps1
.\sh\system\watchdog.ps1 -Register   # авто-перезапуск при сбоях/пробуждении
```

```bash
# Linux: запуск обработки
./sh/pipeline/2_yolo_boxes_files.sh
./sh/pipeline/3_classify_groups.sh
./sh/pipeline/4_identify_residents.sh
```

Пути к моделям (`CLASSIFY_MODEL`, `IDENTIFY_MODEL`, `DETECT_MODEL`) и пороги (`YOLO_CONF`, `YOLO_NMS`, `YOLO_MAX_FPS`, `CLASSIFY_CONF`, `IDENTIFY_CONF`) читаются из `.env`.

**Проверить что прогон запустился** — должна появиться новая директория через ~5 сек:
```powershell
Get-ChildItem ".output\pipeline\1_motion_diff" | Sort-Object LastWriteTime -Descending | Select-Object -First 3 Name, LastWriteTime
```

**Частые ошибки:**
- `conda` не найден в PATH — sh/-скрипты используют полный путь `$env:USERPROFILE\miniconda3\Scripts\conda.exe` автоматически
- `UnicodeEncodeError cp1252` — ps1-скрипты устанавливают `$env:PYTHONIOENCODING = "utf-8"` автоматически
- Директория не появилась через 10 сек — скрипт упал при старте, смотреть вывод в консоли

---

## Доступ к камерам 2–4

### Вариант 1: Port forwarding (рекомендуется как первый шаг)

На внешнем роутере пробросить порты:

```
порт 8741 → 192.168.X.X:554  (камера 2)
порт 8742 → 192.168.X.X:554  (камера 3)
порт 8743 → 192.168.X.X:554  (камера 4)
```

Прод-сервер обращается к камерам через `<внешний_IP_роутера>:554X`.

### Варианты сравнение

| Вариант | Сложность | Надёжность | Описание |
|---------|-----------|------------|----------|
| **Port forwarding** | низкая | средняя | Пробросить порт 554 на каждом роутере |
| **WireGuard VPN** | средняя | высокая | Объединить все роутеры в одну сеть |
| **go2rtc прокси** | средняя | высокая | RTSP прокси рядом с каждым роутером |
| **P2P (DVRIP)** | высокая | низкая | Через облако iCSee — не для production |

---

## Почему P2P (DVRIP) не подходит для production

Приложение iCSee на Android работает через **DVRIP/P2P** — трафик идёт через серверы производителя в Китае.

Проблемы:
- **Надёжность**: зависит от стороннего облака
- **Задержка**: лишний хоп +100–300 мс
- **Безопасность**: видеопоток через серверы третьей стороны
- **Лимиты**: производитель может ограничить соединения
- **Отладка**: закрытый протокол без документации

Для разработки и тестирования DVRIP допустим (Python-библиотека `dvrip`), в production — только прямой RTSP.

---

## Встроенная детекция движения

iCSee камеры имеют Motion Detection на борту, но получить события из неё без DVRIP/облака нет надёжного способа. Используется frame diff на стороне сервера — см. [ml.md](ml.md).
