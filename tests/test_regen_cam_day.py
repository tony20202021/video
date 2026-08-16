"""Тесты склейки полного дня cam3 — scripts/utils/regen_cam_day.py (без scp/камеры)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))
import regen_cam_day as rcd  # noqa: E402


def test_sod_from_ts():
    assert rcd.sod_from_ts("20260815_230659_052268_msk") == pytest.approx(23 * 3600 + 6 * 60 + 59 + 0.052268)
    assert rcd.sod_from_ts("20260815_000000_000000_msk") == pytest.approx(0.0)
    assert rcd.sod_from_ts("мусор") is None


def _w(d: Path, name: str, rows: list[str], header: str = "mono_s,ts_msk,c,v") -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.csv").write_text("\n".join([header] + rows) + "\n", encoding="utf-8")


def test_stitch_filters_by_date_and_gathers_neighbors(tmp_path):
    D, P, N = "20260815", "20260814", "20260816"
    _w(tmp_path / P, "frames", ["100.0,20260815_020000_000000_msk,x,1"])   # ночь D залетела в каталог D-1
    _w(tmp_path / P, "frames_ignore", [])
    _w(tmp_path / D, "frames", [
        "5.0,20260815_080000_000000_msk,x,1",
        "3.0,20260815_200000_000000_msk,x,1",     # рестарт: mono сброшен (3<5)
        "9.0,20260816_000030_000000_msk,x,1",     # чужая дата (D+1) — отфильтровать
        "1.0,20260814_235959_000000_msk,x,1",     # чужая дата (D-1) — отфильтровать
    ])
    _w(tmp_path / N, "frames", ["7.0,20260815_233000_000000_msk,x,1"])     # поздний вечер D залетел в D+1

    out = rcd.stitch_csvs([tmp_path / P, tmp_path / D, tmp_path / N], D, names=("frames",))
    rows = out["frames"][1:]  # без header
    ts = [r.split(",")[1] for r in rows]
    monos = [float(r.split(",")[0]) for r in rows]

    # 4 строки именно даты D (02:00 из D-1, 08:00 и 20:00 из D, 23:30 из D+1); чужие даты убраны
    assert len(rows) == 4
    assert all(t[:8] == D for t in ts)
    # отсортировано по времени суток; mono переписан в секунды суток (монотонно, без сброса)
    assert monos == sorted(monos)
    assert monos[0] == pytest.approx(2 * 3600)      # 02:00 → 7200 сек
    assert monos[-1] == pytest.approx(23 * 3600 + 30 * 60)


def test_stitch_dedup_identical_rows(tmp_path):
    D = "20260815"
    _w(tmp_path / D, "diffs", ["5.0,20260815_080000_000000_msk,cam,4.2"])
    _w(tmp_path / f"{D}_dup", "diffs", ["9.9,20260815_080000_000000_msk,cam,4.2"])  # тот же кадр, другой mono
    out = rcd.stitch_csvs([tmp_path / D, tmp_path / f"{D}_dup"], D, names=("diffs",))
    assert len(out["diffs"][1:]) == 1  # дубль (по содержимому без mono) убран


def test_neighbors():
    assert rcd._neighbors("20260815") == ["20260814", "20260815", "20260816"]
    assert rcd._neighbors("20260101") == ["20251231", "20260101", "20260102"]
