# Тестирование

## Запуск

```powershell
# Все тесты (рекомендуется)
conda run -n conda_video python scripts/tests/run_tests.py

# Быстрые тесты — без камеры, без tests/data/
conda run -n conda_video python scripts/tests/run_tests.py --fast

# Только smoke (проверка загрузки модели)
conda run -n conda_video python scripts/tests/run_tests.py --smoke

# Фильтр по имени (любые pytest-аргументы передаются напрямую)
conda run -n conda_video python scripts/tests/run_tests.py -k motion
conda run -n conda_video python scripts/tests/run_tests.py -k "not integration" -s
```

Или напрямую через pytest из корня проекта:

```powershell
conda run -n conda_video pytest
conda run -n conda_video pytest tests/test_motion_utils.py -s
```

---

## Что и когда запускать

| Режим | Команда | Нужно | Когда |
|-------|---------|-------|-------|
| Все тесты | `run_tests.py` | `yolov8n.onnx` | перед коммитом |
| Быстрые | `--fast` | ничего | в процессе разработки |
| Smoke | `--smoke` | `yolov8n.onnx` | после изменений в детекции |
| Интеграционные | `--all` или напрямую | `yolov8n.onnx` + `tests/data/` | после сбора тестовых данных |

---

## Тестовые файлы

| Файл | Описание | Требования |
|------|----------|------------|
| `test_motion_utils.py` | Утилиты motion-цикла: diff, plausibility, redact_url | нет |
| `test_rtcp_time.py` | Парсинг RTCP SR пакетов, NTP конвертация | нет |
| `test_osd_time.py` | Извлечение OSD-времени с кадра камеры | нет |
| `test_ml_pipeline.py` | ML-пайплайн: classify → identify, работает без моделей | нет |
| `test_person_detection_stub.py` | Заглушки детекции без ONNX | нет |
| `test_smoke.py` | Загрузка YOLO, формат выхода, детекция на baseline-кадрах | `yolov8n.onnx` |
| `test_person_detection.py` | Детекция людей на реальных кадрах | `yolov8n.onnx` |
| `test_integration.py` | Полный пайплайн на тестовых данных | `yolov8n.onnx` + `tests/data/` |

---

## Подготовка тестовых данных (для интеграционных тестов)

```powershell
# Собрать репрезентативные кадры из .output/cameras/5_diff_yolo_boxes_low/
conda run -n conda_video python scripts/utils/collect_test_frames.py
```

Данные сохраняются в `tests/data/`:
- `raw_pairs/` — HI-кадры при срабатывании порога
- `no_person/` — heartbeat-кадры (пустая сцена)
- `with_person/` — кадры с обнаруженными людьми

---

## Структура результатов pytest

```
tests/
  conftest.py              — добавляет src/ в sys.path
  test_*.py                — тестовые файлы
  data/                    — тестовые изображения (в .gitignore)
```

Конфигурация в `pytest.ini`: `testpaths = tests`, `-v` по умолчанию.
