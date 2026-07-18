"""Тесты парсеров scripts/utils/plot_meta_charts.py (без matplotlib)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))
import plot_meta_charts as pmc  # noqa: E402


def test_ts_msk_to_sec():
    # 07:03:22.928715 → 7*3600 + 3*60 + 22 + 0.928715
    assert pmc.ts_msk_to_sec("20260718_070322_928715_msk") == pytest.approx(25402.928715)
    # полночь
    assert pmc.ts_msk_to_sec("20260718_000000_000000_msk") == pytest.approx(0.0)


def test_ts_msk_to_sec_bad_input():
    assert pmc.ts_msk_to_sec("мусор") == 0.0
    assert pmc.ts_msk_to_sec("") == 0.0


def test_parse_run_log_durations_dedup():
    text = "\n".join([
        "07:03:40  INFO   cam_b1_x_diff37.0.jpg  diff=36.97  Готово. Время: 22.8 с.",
        "07:03:40  INFO   cam_b2_x_diff37.0.jpg  diff=36.97  Готово. Время: 22.8 с.",
        "07:03:40  INFO   cam_b3_x_diff37.0.jpg  diff=36.97  Готово. Время: 22.8 с.",
        "07:03:41  INFO   cam_b1_x_diff36.9.jpg  diff=36.94  Готово. Время: 1.5 с.",
        "07:03:17  INFO   Порог: 4.0",                       # не совпадает — игнор
        "07:06:37  INFO   cam_b1_y  Готово. Время: 175.7 с.",
    ])
    got = pmc.parse_run_log_durations(text)
    assert got == [(25420.0, 22.8), (25421.0, 1.5), (25597.0, 175.7)]


def test_parse_run_log_durations_empty():
    assert pmc.parse_run_log_durations("нет подходящих строк\n") == []


def test_parse_run_log_durations_heartbeat():
    text = "08:00:00  INFO   cam_b1  Готово. Время: 600.0 с."
    got = pmc.parse_run_log_durations(text)
    assert got == [(28800.0, 600.0)]


def test_load_cpu_csv(tmp_path):
    p = tmp_path / "cpu.csv"
    p.write_text(
        "mono_s,ts_msk,cpu_pct,freq_mhz_pdh,freq_mhz_step,cpu_utility_pct\n"
        "5.25,20260718_070322_928715_msk,28.3,0.0,1601.0,0.0\n"
        "12.27,20260718_070329_948858_msk,31.0,2503.0,1601.0,53.0\n",
        encoding="utf-8",
    )
    rows = pmc.load_cpu_csv(p)
    assert len(rows) == 2
    assert rows[0][0] == pytest.approx(5.25)      # mono_s
    assert rows[0][1] == "20260718_070322_928715_msk"
    assert rows[0][2] == pytest.approx(28.3)      # cpu%
    assert rows[1][5] == pytest.approx(53.0)      # utility


def test_load_cpu_csv_skips_bad_rows(tmp_path):
    p = tmp_path / "cpu.csv"
    p.write_text(
        "mono_s,ts_msk,cpu_pct\n"
        "5.25,ts,28.3\n"
        "битая,строка,нечисло\n"          # mono не число → пропуск
        "9.9,ts2,40.0\n",
        encoding="utf-8",
    )
    rows = pmc.load_cpu_csv(p)
    assert len(rows) == 2
    assert rows[0][0] == pytest.approx(5.25)
    assert rows[1][2] == pytest.approx(40.0)


def test_unwrap_midnight():
    # 23:59:58 → 00:00:02  должно стать 86402 (не откат назад)
    out = pmc._unwrap_midnight([86398.0, 86399.0, 2.0, 3.0])
    assert out == [86398.0, 86399.0, 86402.0, 86403.0]
