"""
Запуск тестов проекта.

  python scripts/tests/run_tests.py              # все тесты
  python scripts/tests/run_tests.py --fast       # без интеграционных (не нужны tests/data/)
  python scripts/tests/run_tests.py --smoke      # только smoke (yolov8n.onnx)
  python scripts/tests/run_tests.py -k motion    # фильтр по имени (передаётся в pytest)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"

# Тесты без камеры и без tests/data/ — всегда проходят
FAST_TESTS = [
    "test_motion_utils.py",
    "test_rtcp_time.py",
    "test_osd_time.py",
    "test_ml_pipeline.py",
    "test_smoke.py",
    "test_person_detection_stub.py",
]

SMOKE_TESTS = ["test_smoke.py"]


def main() -> int:
    args = sys.argv[1:]

    pytest_args: list[str] = ["-v"]

    if "--fast" in args:
        args.remove("--fast")
        pytest_args += [str(TESTS_DIR / f) for f in FAST_TESTS if (TESTS_DIR / f).is_file()]
        print("Режим: быстрые тесты (без интеграционных)\n")
    elif "--smoke" in args:
        args.remove("--smoke")
        pytest_args += [str(TESTS_DIR / f) for f in SMOKE_TESTS if (TESTS_DIR / f).is_file()]
        print("Режим: smoke-тесты\n")
    else:
        pytest_args.append(str(TESTS_DIR))
        print("Режим: все тесты\n")

    # Остальные аргументы передаём напрямую в pytest (например -k, -s, --tb)
    pytest_args += args

    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
