"""Тесты pipeline_status.py: склонение, подсчёт выходных файлов за окно, формат ячейки."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))
import pipeline_status as ps  # noqa: E402


def test_plural_runs():
    p = lambda n: ps._plural(n, "прогон", "прогона", "прогонов")
    assert p(1) == "прогон"
    assert p(2) == "прогона"
    assert p(4) == "прогона"
    assert p(5) == "прогонов"
    assert p(11) == "прогонов"   # 11-14 — исключение
    assert p(14) == "прогонов"
    assert p(21) == "прогон"
    assert p(22) == "прогона"


def test_plural_files():
    p = lambda n: ps._plural(n, "файл", "файла", "файлов")
    assert p(0) == "файлов"
    assert p(1) == "файл"
    assert p(262) == "файла"     # …2 → few
    assert p(330) == "файлов"    # …0 → many
    assert p(377) == "файлов"    # …7 → many


def test_out_files_in_window(tmp_path):
    d = tmp_path / "images" / "20260718"
    d.mkdir(parents=True)
    now = time.time()
    (d / "recent.jpg").write_bytes(b"x")           # свежий — в окне
    old = d / "old.jpg"
    old.write_bytes(b"x")
    os.utime(old, (now - 3600, now - 3600))        # час назад — вне 10-мин окна
    (d / "note.txt").write_bytes(b"x")             # не .jpg — игнор
    assert ps.out_files_in_window(tmp_path, 10) == 1      # только recent
    assert ps.out_files_in_window(tmp_path, 120) == 2     # оба (окно 2ч)
    assert ps.out_files_in_window(None, 10) == 0
    assert ps.out_files_in_window(tmp_path / "нет", 10) == 0


def _st(count, window):
    return {"count": count, "window": window,
            "min": None, "avg": None, "max": None, "last": None}


def test_fmt_stats_runs_with_files():
    # yolo/classify/identify: N прогонов (X файлов)
    assert ps.fmt_stats(_st(9, 60), False, None, n_is_runs=True, files_n=330) \
        == "9 прогонов (1ч) (330 файлов)"
    assert ps.fmt_stats(_st(1, 10), False, None, n_is_runs=True, files_n=1) \
        == "1 прогон (10м) (1 файл)"
    assert ps.fmt_stats(_st(6, 60), False, None, n_is_runs=True, files_n=262) \
        == "6 прогонов (1ч) (262 файла)"


def test_fmt_stats_frames_transfer():
    # transfer: N× (кадры), без «(файлов)»
    assert ps.fmt_stats(_st(60, 10), False, None, n_is_runs=False) == "60× (10м)"


def test_fmt_stats_zero():
    assert ps.fmt_stats(_st(0, 10), False, None, n_is_runs=True, files_n=0) == "—"
