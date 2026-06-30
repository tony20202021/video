# Передача данных между машинами (Transfer)

Компонент для передачи `run_*`-каталогов между клиентскими машинами и центральным сервером.

## Схема

```
Клиент (Windows, дом/подъезд)           Сервер (Linux, публичный IP)
────────────────────────────────         ────────────────────────────
1_motion_diff → run_XXX/
        │
        │  tar.gz  POST /pipeline/1_motion_diff
        └──────────────────────────────────────►  .output/pipeline/1_motion_diff/run_XXX/
                                                          │
                                                  2_yolo_boxes_files
                                                  3_classify_groups
                                                  4_identify_residents
```

Клиент запускает шаг 1 (камеры), упаковывает результат и отправляет. Сервер принимает и продолжает обработку со шага 2.

## Быстрый старт

**Сервер (Linux):**
```bash
export TRANSFER_API_KEY=mysecretkey
./sh/transfer/start_server.sh --port 8765
```

**Клиент (Windows), после шага 1:**
```powershell
# Добавить в .env:
# TRANSFER_SERVER=http://<server-ip>:8765
# TRANSFER_API_KEY=mysecretkey

.\sh\transfer\send.ps1 -Src .output\pipeline\1_motion_diff\run_20260629_XXX
```

---

## Скрипты

| Файл | Назначение |
|------|-----------|
| `scripts/transfer/server.py` | FastAPI-сервер приёма |
| `scripts/transfer/client.py` | CLI-клиент отправки |
| `sh/transfer/send.ps1` | PowerShell-обёртка для клиента |
| `sh/transfer/start_server.sh` | Запуск сервера на Linux |

---

## Сервер

```bash
# Запуск напрямую
python scripts/transfer/server.py --port 8765

# Или через uvicorn (для продакшена)
TRANSFER_API_KEY=secret uvicorn scripts.transfer.server:app --host 0.0.0.0 --port 8765
```

### Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| `GET`  | `/health` | Проверка доступности |
| `POST` | `/pipeline/{step}` | Принять tar.gz с run_* каталогом |
| `GET`  | `/runs` | Список всех принятых прогонов |

**Аутентификация:** заголовок `X-Api-Key`. Если `TRANSFER_API_KEY` не задан — сервер принимает без ключа.

---

## Клиент

```powershell
# Отправить run_* каталог
python scripts/transfer/client.py send .output\pipeline\1_motion_diff\run_XXX

# Явные параметры
python scripts/transfer/client.py send run_XXX --server http://1.2.3.4:8765 --key SECRET

# Проверить доступность сервера
python scripts/transfer/client.py health

# Список принятых прогонов на сервере
python scripts/transfer/client.py runs
```

Шаг пайплайна (`step`) определяется автоматически из имени родительского каталога:
```
.output/pipeline/1_motion_diff/run_XXX   →  step = 1_motion_diff
.output/pipeline/2_yolo_boxes_files/run_XXX  →  step = 2_yolo_boxes_files
```

---

## Конфигурация (.env)

```dotenv
TRANSFER_SERVER=http://<server-ip>:8765
TRANSFER_API_KEY=mysecretkey
```

---

## Полный цикл: клиент → сервер

```
Клиент (подъезд/дом):
  .\sh\pipeline\1_motion_diff.ps1          # записать кропы движения
  .\sh\transfer\send.ps1 -Src .output\pipeline\1_motion_diff\run_XXX

Сервер:
  .\sh\pipeline\2_yolo_boxes_files.ps1     # детекция людей
  .\sh\pipeline\3_classify_groups.ps1      # классификация
  .\sh\pipeline\4_identify_residents.ps1   # идентификация
```

---

## Тесты

```bash
pytest tests/test_transfer.py -v
```
