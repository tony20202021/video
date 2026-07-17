"""
Запуск тестов проекта.

  python scripts/tests/run_tests.py              # все тесты БЕЗ integration (юнит/стаб, всегда зелёные)
  python scripts/tests/run_tests.py --all        # + integration (нужны реальная модель/кадры/датасет)
  python scripts/tests/run_tests.py --fast       # быстрый набор файлов (юнит/стаб)
  python scripts/tests/run_tests.py --smoke      # только smoke (на StubSession)
  python scripts/tests/run_tests.py -k motion    # фильтр по имени (передаётся в pytest)

Маркер `integration` (pytest.ini) помечает тесты, которым нужны реальные артефакты
(модель, кадры, размеченный датасет). По умолчанию они исключаются, чтобы прогон был
100% зелёным на заглушках; включаются флагом --all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"

# Быстрый набор файлов — юнит/стаб, без реальной модели и tests/data/
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
    run_integration = False

    if "--all" in args:
        args.remove("--all")
        run_integration = True

    if "--fast" in args:
        args.remove("--fast")
        pytest_args += [str(TESTS_DIR / f) for f in FAST_TESTS if (TESTS_DIR / f).is_file()]
        print("Режим: быстрые тесты (юнит/стаб)\n")
    elif "--smoke" in args:
        args.remove("--smoke")
        pytest_args += [str(TESTS_DIR / f) for f in SMOKE_TESTS if (TESTS_DIR / f).is_file()]
        print("Режим: smoke-тесты (StubSession)\n")
    else:
        pytest_args.append(str(TESTS_DIR))
        print("Режим: все тесты" + (" (+ integration)\n" if run_integration else " (без integration)\n"))

    # integration-тесты по умолчанию исключаются (нужны реальные артефакты)
    if not run_integration:
        pytest_args += ["-m", "not integration"]

    # Остальные аргументы передаём напрямую в pytest (например -k, -s, --tb)
    pytest_args += args

    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
