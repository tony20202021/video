# Начальная настройка окружения (Python + conda)

Краткая инструкция: установка **conda** (рекомендуется **Miniconda**), создание изолированного окружения и установка зависимостей из `requirements.txt` в корне репозитория.

См. также: [Общее описание](overview.md) (требования к Python 3.11+), [Сервисы и конфиг](services.md).

---

## Что понадобится

| Компонент | Примечание |
|-----------|------------|
| **Python** | В проекте ориентир **3.11+** (см. `overview.md`). Для окружения удобно зафиксировать **3.11** или **3.12**. |
| **conda** | Miniconda или Anaconda — на выбор; ниже команды одинаковы. |
| **Репозиторий** | Клонированный каталог `video/` с файлом `requirements.txt`. |

Установщики Miniconda: [Windows](https://docs.conda.io/en/latest/miniconda.html) / [Linux x86_64](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) (актуальные ссылки — на сайте conda).

После установки откройте новый терминал и убедитесь, что `conda` доступна:

```bash
conda --version
```

Если команды нет — перезапустите терминал или выполните инициализацию для своей оболочки (см. подсказку установщика: `conda init`).

---

## Linux

Перейдите в корень репозитория (где лежит `requirements.txt`):

```bash
cd /path/to/video
```

Создайте окружение (имя `conda_video` при желании замените на своё):

```bash
conda create -n conda_video python=3.11 -y
```

Активируйте окружение:

```bash
conda activate conda_video
```

Установите зависимости проекта:

```bash
pip install -r requirements.txt
```

Проверка:

```bash
python -c "import fastapi; print('OK')"
```

Деактивация при необходимости: `conda deactivate`.

---

## Windows

Используйте **Anaconda Prompt**, **Miniconda Prompt** или **PowerShell** / **cmd** после `conda init` (чтобы сработали `conda activate`).

Перейдите в каталог репозитория (пример):

```powershell
cd C:\path\to\video
```

Создайте окружение:

```powershell
conda create -n conda_video python=3.11 -y
```

Активируйте:

```powershell
conda activate conda_video
```

Установите зависимости:

```powershell
pip install -r requirements.txt
```

Проверка:

```powershell
python -c "import fastapi; print('OK')"
```

Если `conda` не находится в PATH, откройте меню «Пуск» → **Anaconda Prompt** / **Miniconda** и повторите шаги там.

---

## Скачивание ML-моделей

После установки зависимостей нужно подготовить ONNX-модели (папка `models/`).

Скрипт `scripts/setup_models.py` проверяет наличие файла и, если его нет, скачивает `.pt` через `ultralytics` и экспортирует в ONNX. Если модель уже есть — ничего не делает.

### Linux

```bash
python scripts/setup_models.py
```

При необходимости другая модель (по умолчанию `yolov8n`):

```bash
python scripts/setup_models.py --model yolov8s
```

### Windows

```powershell
python scripts/setup_models.py
```

Если `ultralytics` при скачивании `.pt` блокируется (GitHub недоступен без VPN):

```powershell
# Включите Psiphon, узнайте порт:
.\sh\psiphon_proxy.ps1

# Задайте прокси для Python и запустите:
$env:HTTPS_PROXY = "http://127.0.0.1:<PORT>"
python scripts/setup_models.py
```

> Модели нужны только для `scripts/cameras/motion_people.py`. Остальные скрипты (`verify_cameras.py`, `motion_watch.py`) работают без моделей.

---

## Частые замечания

- **Версия Python:** если пакет из `requirements.txt` не ставится на выбранную версию, создайте окружение с другой минорной версией Python (`python=3.12` и т.д.) и повторите `pip install -r requirements.txt`.
- **pip внутри conda:** после `conda activate` команда `pip` относится к этому окружению — так и нужно для установки из `requirements.txt`.
- **Секреты и конфиг:** переменные окружения и настройки приложения описаны в других разделах документации; само окружение conda их не создаёт — при необходимости скопируйте `.env` по образцу проекта.
