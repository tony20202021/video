# Передача данных между машинами (Transfer)

Компонент для непрерывной передачи файлов (дифф-кадров) с клиентской машины на сервер.
Файлы отправляются поштучно через `POST /file` — без упаковки в архив.

## Схема

```
Клиент (Windows, дом/подъезд)               Сервер (Linux, публичный IP)
────────────────────────────────             ────────────────────────────
1_motion_diff → images/YYYYMMDD/cam/diff/
        │
        │  watch: POST /file per file
        │  X-Parent / X-Step / X-Run / X-Cam / X-Date / …
        └───────────────────────────────►  .output/transfer/
                                               diff/YYYYMMDD/<cam>/file.jpg
                                               service/YYYYMMDD/<cam>/file.jpg
                                               meta/YYYYMMDD/<cam>/file.json
                                                     │
                                             2_yolo_boxes_files
                                             3_classify_groups
                                             4_identify_residents
```

Клиент запускает `watch` — непрерывно следит за каталогом, отправляет каждый новый файл,
после подтверждения сервера удаляет локально.

## Быстрый старт

**Сервер (Linux):**
```bash
# Задать ключ в .env, затем:
./sh/transfer/1_start_server.sh
```

**Клиент (Windows), параллельно с записью:**
```powershell
# Задать TRANSFER_SERVER и TRANSFER_API_KEY в .env, затем:
.\sh\transfer\2_send.ps1 .output\pipeline\1_motion_diff\run_XXX\images
```

---

## Скрипты

| Файл | Назначение |
|------|-----------|
| `scripts/transfer/server.py` | FastAPI-сервер приёма |
| `scripts/transfer/client.py` | CLI-клиент (send / watch / health / runs) |
| `sh/transfer/1_start_server.ps1` | Windows-обёртка: запустить сервер |
| `sh/transfer/1_start_server.sh` | Linux-обёртка: запустить сервер |
| `sh/transfer/2_send.ps1` | Windows-обёртка: запустить `watch` |
| `sh/transfer/2_send.sh` | Linux-обёртка: запустить `watch` |

---

## Структура каталога images/

Дифф-кадры пишутся по датам (MSK), чтобы не переполнять один каталог:

```
images/
  20260702/
    cam_01_9_d/
      diff/
        20260702_140301_456789.jpg
  20260703/
    cam_01_9_d/
      diff/
        20260703_000012_123456.jpg
```

`watch` рекурсивно сканирует каталог и передаёт все `.jpg` (по умолчанию),
сохраняя относительный путь как `X-Rel-Path`.

---

## Сервер

```bash
# Через обёртку (читает TRANSFER_HOST/PORT из .env)
./sh/transfer/1_start_server.sh

# Напрямую
python scripts/transfer/server.py --port 8765

# Через uvicorn для продакшена
uvicorn scripts.transfer.server:app --host 0.0.0.0 --port 8765
```

### Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| `GET`  | `/health` | Проверка доступности |
| `POST` | `/file` | Принять один файл (основной) |
| `POST` | `/pipeline/{step}` | Принять tar.gz run_* (устаревший) |
| `GET`  | `/runs` | Список всех принятых прогонов |

**Доступ:** два слоя — IP-фильтр (`ALLOWED_IPS`) и API-ключ (`X-Api-Key`).

| Слой | Переменная | Если не задана |
|------|------------|----------------|
| IP | `ALLOWED_IPS` | любой IP (предупреждение при старте) |
| Ключ | `TRANSFER_API_KEY` | любой запрос без ключа |

Сервер слушает `0.0.0.0`, но запросы с IP вне whitelist получают **403 Forbidden** до проверки ключа.
`127.0.0.1` / `::1` разрешены всегда (health с самого сервера).

Если ключ **верный**, но IP не в списке — в терминале и логе появится подсказка с `/24`-подсетью
(типичная смена домашнего IP провайдера). Обновите `ALLOWED_IPS` в `.env` и перезапустите сервер.

### POST /file — заголовки

| Заголовок | Обязателен | Описание |
|-----------|-----------|----------|
| `X-Api-Key` | если ключ задан | API-ключ аутентификации |
| `X-Parent` | рекомендуется | Родитель шага (напр. `pipeline`) |
| `X-Step` | да* | Шаг пайплайна (напр. `1_motion_diff`) |
| `X-Run` | да* | Прогон (напр. `run_20260702_200143_msk`) |
| `X-Rel-Path` | да* | Путь файла относительно `WatchDir` |
| `X-Cam` | для diff/service | Камера (из имени файла) |
| `X-Date` | для diff/service | Дата `YYYYMMDD` (из имени файла) |
| `X-Time` | нет | Время `HHMMSS_micro` (из имени файла) |
| `X-Type` | нет | `diff` / `baseline` / `heartbeat` / … |

\* Для jpg с `X-Cam`+`X-Date` картинка кладётся в `diff/` или `service/`; без них — legacy `{step}/{run}/{rel_path}`.

Тело запроса — сырые байты файла.

### Маршрутизация на клиенте (parent / step / run)

| `WatchDir` | parent | step | run |
|------------|--------|------|-----|
| `.../1_motion_diff/` | `pipeline` | `1_motion_diff` | из пути каждого файла (`run_*`) |
| `.../run_XXX/images` | `pipeline` | `1_motion_diff` | `run_XXX` |

### Sidecar meta/

Параллельно с `diff/` и `service/` сервер пишет JSON с тем же `YYYYMMDD/<cam>/`:

```
.output/transfer/
  diff/20260704/cam_01_9_d/cam_01_9_d_20260704_140301_456789_msk_diff11.0.jpg
  meta/20260704/cam_01_9_d/cam_01_9_d_20260704_140301_456789_msk_diff11.0.json
```

Пример sidecar:

```json
{
  "parent": "pipeline",
  "step": "1_motion_diff",
  "run": "run_20260704_002002_msk",
  "cam": "cam_01_9_d",
  "date": "20260704",
  "time": "140301_456789",
  "timezone": "Europe/Moscow",
  "captured_at": "2026-07-04T14:03:01.456789+03:00",
  "type": "diff",
  "rel_path": "run_.../images/20260704/cam_01_9_d/diff/....jpg",
  "received_at": "2026-07-04T20:15:03.123456+03:00",
  "size_bytes": 40832,
  "dest_image": "/path/to/diff/.../file.jpg"
}
```

- `captured_at` — из имени файла (MSK, `+03:00`)
- `received_at` — момент записи на сервер (MSK, `+03:00`)

---

## Клиент

```bash
# Следить за каталогом (рекомендуется — весь шаг, новые run_* подхватываются автоматически)
python scripts/transfer/client.py watch .output/pipeline/1_motion_diff

# Или один прогон
python scripts/transfer/client.py watch .output/pipeline/1_motion_diff/run_XXX/images

# Разово отправить все файлы из run_*-каталога
python scripts/transfer/client.py send .output/pipeline/1_motion_diff/run_XXX

# Проверить доступность сервера
python scripts/transfer/client.py health

# Список принятых прогонов на сервере
python scripts/transfer/client.py runs
```

---

## Конфигурация (.env)

```dotenv
# ─── Transfer ─────────────────────────────────────────────────────────────────
TRANSFER_SERVER=http://<server-ip>:8765
TRANSFER_API_KEY=mysecretkey
TRANSFER_HOST=0.0.0.0
TRANSFER_PORT=8765

# IP-доступ к Transfer и Label UI (сервер). Через запятую: один IP или CIDR.
# Пример: домашняя подсеть провайдера — переживает смену последнего октета.
ALLOWED_IPS=109.252.161.0/24

# Параметры режима watch
TRANSFER_POLL_SEC=1.0        # интервал проверки новых файлов, сек
TRANSFER_MAX_RATE=5.0        # макс. файлов/с при низкой нагрузке
TRANSFER_MIN_RATE=0.1        # мин. файлов/с при высокой нагрузке
TRANSFER_ADAPT_WINDOW=10     # окно адаптации (последние N передач)
TRANSFER_ADAPT_HIGH=0.90     # порог "перегружен" → замедлить в FACTOR раз
TRANSFER_ADAPT_LOW=0.40      # порог "недогружен" → ускорить в FACTOR раз
TRANSFER_ADAPT_FACTOR=2.0    # коэффициент изменения скорости
```

---

## Адаптивное ограничение CPU

`watch` использует `AdaptiveRateLimiter` (`src/common/utils/adaptive_rate.py`) —
тот же модуль, что и в YOLO-инференсе.

- Скорость начинается с `TRANSFER_MAX_RATE` файлов/с
- Если CPU занят > 90% времени (`ADAPT_HIGH`) — замедляется в `ADAPT_FACTOR` раз
- Если CPU занят < 40% времени (`ADAPT_LOW`) — ускоряется в `ADAPT_FACTOR` раз
- Не выходит за пределы `MIN_RATE` .. `MAX_RATE`

---

## Полный цикл

```
Клиент (подъезд/дом):
  .\sh\pipeline\1_motion_diff.ps1         # записывает images/YYYYMMDD/...
  .\sh\transfer\2_send.ps1                  # watch .output/pipeline/1_motion_diff/

Сервер:
  ./sh/transfer/1_start_server.sh         # запущен постоянно
  ./sh/pipeline/2_yolo_boxes_files.sh     # детекция людей
  ./sh/pipeline/3_classify_groups.sh      # классификация
  ./sh/pipeline/4_identify_residents.sh   # идентификация
```

---

## Тесты

```bash
pytest tests/test_access.py tests/test_transfer.py tests/test_adaptive_rate.py -v
```
