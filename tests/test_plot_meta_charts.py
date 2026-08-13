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


def test_load_timing_csv(tmp_path):
    p = tmp_path / "yolo_timing.csv"
    p.write_text(
        "mono_s,inference_ms,sleep_ms\n"
        "0.5,182.3,900.0\n"
        "3.5,201.7,880.0\n",
        encoding="utf-8",
    )
    rows = pmc.load_timing_csv(p)
    assert len(rows) == 2
    assert rows[0][0] == pytest.approx(0.5)      # mono_s
    assert rows[0][1] == pytest.approx(182.3)    # inference_ms — время обработки кадра
    assert rows[1][2] == pytest.approx(880.0)    # sleep_ms


def test_load_timing_csv_skips_bad_rows(tmp_path):
    p = tmp_path / "yolo_timing.csv"
    p.write_text(
        "mono_s,inference_ms,sleep_ms\n"
        "0.5,182.3,900.0\n"
        "битая,строка,x\n"               # mono/inf не число → пропуск
        "9.9,250.0\n"                    # без sleep_ms → slp=0.0
        "мало\n",                        # < 2 колонок → пропуск
        encoding="utf-8",
    )
    rows = pmc.load_timing_csv(p)
    assert len(rows) == 2
    assert rows[0][1] == pytest.approx(182.3)
    assert rows[1][0] == pytest.approx(9.9)
    assert rows[1][2] == 0.0             # sleep_ms по умолчанию


def test_load_timing_csv_with_ts_msk(tmp_path):
    # новый 4-кол формат: ts_msk в КОНЦЕ (ось накопленного за день графика); r[1]/r[2] не сдвинуты
    p = tmp_path / "yolo_timing.csv"
    p.write_text(
        "mono_s,inference_ms,sleep_ms,ts_msk\n"
        "0.5,182.3,900.0,20260813_070322_100000\n",
        encoding="utf-8",
    )
    rows = pmc.load_timing_csv(p)
    assert rows[0][1] == pytest.approx(182.3)          # inference_ms на месте
    assert rows[0][3] == "20260813_070322_100000"      # ts_msk
    # старый 3-кол формат → ts_msk пустой (backward-compat)
    p.write_text("mono_s,inference_ms,sleep_ms\n1.0,200.0,0.0\n", encoding="utf-8")
    assert pmc.load_timing_csv(p)[0][3] == ""


def test_save_cpu_csv_append(tmp_path):
    from common.utils.camera_run import save_cpu_csv
    _row = lambda t: [float(t), f"20260813_0800{t:02d}_0", 20.0, 2400, 2400, 30]
    save_cpu_csv([_row(1), _row(2)], tmp_path, append=True)   # файла нет → заголовок + 2
    save_cpu_csv([_row(3)], tmp_path, append=True)            # append → +1 (заголовок не дублируется)
    lines = (tmp_path / "cpu.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4                                    # 1 заголовок + 3 строки
    assert lines[0].startswith("mono_s")
    save_cpu_csv([_row(9)], tmp_path)                         # append=False (дефолт) → перезапись
    assert len((tmp_path / "cpu.csv").read_text(encoding="utf-8").splitlines()) == 2


def test_unwrap_midnight():
    # 23:59:58 → 00:00:02  должно стать 86402 (не откат назад)
    out = pmc._unwrap_midnight([86398.0, 86399.0, 2.0, 3.0])
    assert out == [86398.0, 86399.0, 86402.0, 86403.0]


def test_load_frames_csv(tmp_path):
    p = tmp_path / "frames.csv"
    p.write_text(
        "mono_s,ts_msk,url_id,ok,plausible,event\n"
        "0.5,ts,CAM_A_URL,1,1,\n"
        "0.6,ts,CAM_A_URL,0,1,\n"          # ok=0 — пропуск
        "0.7,ts,CAM_A_URL,1,1,\n"
        "битая\n",                          # мало колонок — пропуск
        encoding="utf-8",
    )
    fr = pmc.load_frames_csv(p)
    assert fr == [(0.5, "CAM_A_URL"), (0.7, "CAM_A_URL")]


def test_compute_fps_series():
    # 12 кадров за 1 секунду одной камеры → ~12 fps
    frames = [(i / 12.0, "CAM_A_URL") for i in range(13)]  # 0..1.0 c
    s = pmc.compute_fps_series(frames, bin_s=1.0)
    assert "CAM_A_URL" in s
    # 13 кадров за ~1.083 c → avg ≈ 12
    assert s["CAM_A_URL"]["avg"] == pytest.approx(13 / (12 / 12.0), rel=0.01)
    assert len(s["CAM_A_URL"]["t"]) == len(s["CAM_A_URL"]["fps"])


def test_compute_fps_series_two_cams():
    frames = [(0.1, "CAM_A_URL"), (0.2, "CAM_A_URL"),
              (0.1, "CAM_B_URL"), (0.5, "CAM_B_URL"), (0.9, "CAM_B_URL")]
    s = pmc.compute_fps_series(frames, bin_s=1.0)
    assert set(s) == {"CAM_A_URL", "CAM_B_URL"}


def test_short_cam():
    assert pmc._short_cam("CAM_01_9_D_URL") == "01_9_D"
    assert pmc._short_cam("CAM_B1_URL") == "B1"
    assert pmc._short_cam("other") == "other"
