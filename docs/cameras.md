# Камеры и сетевой доступ

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

В **`4_motion_watch.py`** по умолчанию: **`CAM_<stem>_URL`** — субпоток (часто `stream=1`), детекция движения; **`CAM_<stem>_HI_URL`** — основной поток (часто `stream=0`), сохранение baseline / событий / heartbeat. Если `HI_URL` не задан, высокий и низкий совпадают.

**Два объектива → два URL в `.env`:** один и тот же `rtsp://<IP>:554/...`, отличается только **`channel`** (и при необходимости параллельно подобрать `stream`). Имена переменных — **`CAM_<stem>_URL`**, где `stem` — любой непустой идентификатор (цифры, хвост вроде `_9_U` / `_10_D` для IP и половины склейки). Пример с парой низкий/высокий поток для одной склейки (один `channel`, разный `stream`):

```text
CAM_01_9_U_URL=rtsp://<IP>:554/user=...&password=...&channel=0&stream=1.sdp?real_stream
CAM_01_9_U_HI_URL=rtsp://<IP>:554/user=...&password=...&channel=0&stream=0.sdp?real_stream
CAM_01_9_D_URL=rtsp://<IP>:554/user=...&password=...&channel=0&stream=1.sdp?real_stream
CAM_01_9_D_HI_URL=rtsp://<IP>:554/user=...&password=...&channel=0&stream=0.sdp?real_stream
CAM_01_URL=rtsp://<IP>:554/user=...&password=...&channel=1&stream=1.sdp?real_stream
```

**Один RTSP, картинка склеена из двух линз:** устройство A31 (проверено на `<ip>`) возвращает **один склеенный кадр 2304×2592 на всех channel=0..3** — две камеры расположены вертикально. HTTP snapshot (`/webcapture.jpg`, `/snapshot.jpg` и пр.) возвращает HTTP 400. Используйте один URL (`channel=0&stream=1`) и обрезку: **`MOTION_CROP_REL`** или **`CAM_<stem>_CROP_REL`** (доли **x,y,w,h**) — см. раздел про `4_motion_watch.py` ниже.

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

## IP-адреса — как узнать

1. **Через приложение iCSee**: Device → Device Info → IP Address
2. **Через роутер**: DHCP-таблица в интерфейсе роутера, MAC-адрес камеры начинается на `DC:` или `AC:` (производитель XMTech)
3. **nmap**:
   ```bash
   nmap -p 554,8899,34567 192.168.1.0/24
   ```
   Открытые порты 554 или 8899 = iCSee камера

---

## Логин и пароль

По умолчанию: `admin` / (пустой) или `admin` / `admin`

Задаётся при первой настройке в приложении iCSee. Если неизвестен — сброс физической кнопкой на камере.

**Хранение** в проекте — файл `.env` (в `.gitignore`):

```
CAM_01_URL=rtsp://192.168.1.101:554/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream
CAM_02_URL=rtsp://<external_ip>:5541/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream
CAM_03_URL=rtsp://<external_ip>:5542/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream
CAM_04_URL=rtsp://<external_ip>:5543/user=admin&password=XXXX&channel=1&stream=1.sdp?real_stream
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

Скрипт `scripts/cameras/3_verify_cameras.py` загружает `.env`, для каждой переменной **`CAM_<stem>_URL`** (напр. `CAM_01_URL`, `CAM_01_9_U_URL`, `CAM_01_10_D_URL`) с реальным `rtsp://` открывает поток, читает **один кадр** и собирает **свойства потока** (разрешение, fps, backend OpenCV и т.д.). Переменные **`CAM_*_HI_URL`** скрипт **не** опрашивает — при необходимости проверьте главный поток отдельно (временно подставив URL в тест или второй вызов). Обрезка: **`MOTION_CROP_REL`**, **`CAM_<stem>_CROP_REL`**, **`--crop-rel`** — как в `4_motion_watch.py`; в JPEG и `report.json` — кадр **после** обрезки, плюс `crop_rel` / `frame_shape_before_crop`. Плейсхолдеры вроде `<external_ip>` пропускаются. Результаты — `.output/cam_verify/cam_verify_<метка>/`: `report.json` и `cam_01_9_u_frame.jpg` и т.д. (имя файла из stem).

Из корня репозитория:

```bash
python scripts/cameras/3_verify_cameras.py
python scripts/cameras/3_verify_cameras.py --tcp
python scripts/cameras/3_verify_cameras.py --crop-rel 0,0,1,0.5
```

Флаг `--tcp` включает RTSP поверх TCP — полезно через NAT или нестабильный Wi‑Fi. Поиск камер в локальной сети по портам — отдельно `scripts/cameras/1_scan_cameras.py`.

### Наблюдение изменений кадра (`4_motion_watch.py`)

Скрипт `scripts/cameras/4_motion_watch.py`:

- **`CAM_<stem>_URL`** — **низкое разрешение** (субпоток, обычно `stream=1`): по нему идёт сравнение кадров (после обрезки, уменьшение до **`compare-width`**, grayscale).
- **`CAM_<stem>_HI_URL`** — **высокое разрешение** (главный поток, обычно `stream=0`): с него пишутся **baseline**, кадры при движении и **`…_heartbeat.jpg`**. Если переменная не задана, для сохранения используется тот же URL, что и у **`CAM_*_URL`**.

Несколько логических камер с **одинаковым** `CAM_*_URL` (например U и D с одной склейки) открывают **один** `VideoCapture` на этот RTSP: за такт делается **один** `read()` низкого потока, обрезки считаются отдельно — субпоток не читается дважды. Для HI при срабатывании движения нескольких «половин» с **одним** `CAM_*_HI_URL` делается **одно** чтение главного потока и две обрезки в файл.

При старте: сначала кадр LOW задаёт внутреннее состояние детектора, затем с HI сохраняются baseline JPEG в `.output/motion_watch/motion_watch_<UTC>/` (или `--output`). Порог: **`MOTION_DIFF_THRESHOLD`** / **`--threshold`**. Пульс: **`MOTION_HEARTBEAT_SEC`** / **`--heartbeat-sec`**.

**Склеенный кадр (две линзы в одном изображении):** в `.env` задаётся обрезка в долях **x,y,w,h** от 0 до 1. Глобально **`MOTION_CROP_REL`**, для одной переменной — **`CAM_<stem>_CROP_REL`**. Те же доли применяются и к LOW, и к HI. Вертикальная черта: слева **`0,0,0.5,1`**, справа **`0.5,0,0.5,1`**. Горизонтальная: сверху **`0,0,1,0.5`**, снизу **`0,0.5,1,0.5`**.

```bash
python scripts/cameras/4_motion_watch.py
python scripts/cameras/4_motion_watch.py --tcp --threshold 12
python scripts/cameras/4_motion_watch.py --heartbeat-sec 300
python scripts/cameras/4_motion_watch.py --crop-rel 0,0,1,0.5
```

### Детекция людей (`5_motion_people.py`)

Скрипт `scripts/cameras/5_motion_people.py` использует тот же motion-цикл, но после обнаружения движения запускает **YOLOv8n ONNX** (класс 0 — person). Сохраняет только кадры, на которых обнаружен хотя бы один человек; на сохраняемый кадр наносит **bounding boxes** с confidence. Имя файла: `<cam>_<UTC>_p<N>.jpg` (N — количество людей).

Перед первым запуском скачать модель (~6 MB):

```bash
mkdir -p models
wget -P models/ https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx
```

```bash
python scripts/cameras/5_motion_people.py
python scripts/cameras/5_motion_people.py --conf 0.4 --threshold 12
python scripts/cameras/5_motion_people.py --model models/yolov8n.onnx --tcp
```

Общие утилиты motion-цикла (фильтрация URL, frame diff, проверка кадра HEVC, открытие потока) вынесены в `src/common/utils/motion_utils.py`.

---

## Доступ к камерам 2–4

### Вариант 1: Port forwarding (рекомендуется как первый шаг)

На внешнем роутере пробросить порты:

```
порт 5541 → 192.168.X.X:554  (камера 2)
порт 5542 → 192.168.X.X:554  (камера 3)
порт 5543 → 192.168.X.X:554  (камера 4)
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
