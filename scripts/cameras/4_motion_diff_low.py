"""
Цикл: читать кадры с каждой доступной RTSP-камеры из .env — только LOW-поток (CAM_<stem>_URL).
Детекция движения — frame diff по субпотоку.
При срабатывании — сохраняет LOW-кадр.

Побочные потоки:
  _RtcpWorker  — RTCP NTP калибровка (один на URL, запускается при --cam-ts)
  _CpuMonitor  — замер загрузки ЦПУ раз в N сек (требует psutil)
  _CapReader   — непрерывное чтение кадров с VideoCapture (свежий кадр без буферного лага)

Выход (run_<ts>/):
  images/          — все сохранённые кадры (baseline, diff, heartbeat)
  frames.csv       — метка времени каждого кадра LOW-потока
  diffs.csv        — сырые значения diff для каждого кадра
  pts.csv          — метки времени FFmpeg PTS для анализа дрейфа
  cpu.csv          — загрузка ЦПУ с периодичностью --cpu-interval
  charts.png       — совмещённый график: интервалы кадров + дифы + сохранения + ЦПУ
  pts_chart.png    — анализ дрейфа PTS / wall clock / mono
  osd_chart.png    — сравнение OSD-метки камеры с wall-clock (только --regen-from)
  osd_times.csv    — OSD-метки из сохранённых изображений
  run_stats.json   — метрики качества прогона
  run_params.json
  run.log          — stdout+stderr прогона

Usage:
    python scripts/cameras/4_motion_diff_low.py
    python scripts/cameras/4_motion_diff_low.py --threshold 15 --tcp
    python scripts/cameras/4_motion_diff_low.py --heartbeat-sec 300
    python scripts/cameras/4_motion_diff_low.py --crop-rel 0,0,1,0.5
    python scripts/cameras/4_motion_diff_low.py --duration 3600
    python scripts/cameras/4_motion_diff_low.py --cam-ts           # RTCP NTP метка в имени файла
    python scripts/cameras/4_motion_diff_low.py --regen-from /path/to/run_dir
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.utils.cam_crop import apply_crop_optional, crop_map_for_cameras, resolve_global_crop
from common.utils.cam_urls import collect_cam_urls as _collect_cam_urls
from common.utils.motion_utils import (
    ffmpeg_capture_options as _ffmpeg_capture_options,
    frame_decode_plausible,
    mean_abs_diff as _mean_abs_diff,
    open_cap as _open_cap,
    prepare_gray as _prepare_gray,
    redact_url,
    skip_url as _skip_url,
    stem_from_var as _stem_from_env_var,
)
from common.utils.time_msk import ts_cam_for_file, ts_for_dir, ts_for_file

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT_PARENT = REPO_ROOT / ".output" / "cameras" / "4_motion_diff_low"


# ─── Вспомогательные классы ───────────────────────────────────────────────────

class _Tee:
    """Пишет одновременно в оригинальный поток и в файл."""
    def __init__(self, stream, fobj):
        self._stream = stream
        self._fobj = fobj

    def write(self, data):
        self._stream.write(data)
        try:
            self._fobj.write(data)
            self._fobj.flush()
        except Exception:
            pass

    def flush(self):
        self._stream.flush()
        try:
            self._fobj.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _RtcpWorker:
    """Фоновый поток: получает и периодически обновляет RTCP SR калибровку."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._calib = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"rtcp-{url[-25:]}")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def get_calib(self):
        with self._lock:
            return self._calib

    def _run(self) -> None:
        try:
            from common.utils.rtcp_time import RtcpTimingReader
        except ImportError:
            return
        reader = RtcpTimingReader(self._url)
        while not self._stop.is_set():
            calib = reader.get_calibration(timeout=10.0)
            if calib is not None:
                with self._lock:
                    self._calib = calib
            self._stop.wait(300 if calib else 30)


class _CpuMonitor:
    """Фоновый поток: периодически замеряет cpu_percent() через psutil."""

    def __init__(self, interval: float = 2.0) -> None:
        self._interval = interval
        self._log: list[list] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name="cpu-monitor")
        self._psutil = None
        self._t_start = 0.0

    def start(self, t_start: float) -> bool:
        try:
            import psutil
            psutil.cpu_percent()
            self._psutil = psutil
        except ImportError:
            return False
        self._t_start = t_start
        self._thread.start()
        return True

    def stop(self) -> list[list]:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(self._interval + 1, 3))
        with self._lock:
            return list(self._log)

    def _run(self) -> None:
        while not self._stop.is_set():
            cpu = self._psutil.cpu_percent(interval=self._interval)
            with self._lock:
                self._log.append([
                    round(time.monotonic() - self._t_start, 2),
                    ts_for_file(),
                    round(cpu, 1),
                ])


class _CapReader:
    """Фоновый поток: непрерывно читает VideoCapture, держит только последний кадр.

    Queue(maxsize=1) гарантирует: main-поток всегда получает самый свежий кадр
    без буферного лага, даже если обработка занимает > 1 кадра.
    """

    _MAX_FAILS = 5

    def __init__(self, url: str, cap: "cv2.VideoCapture",
                 open_timeout_ms: int, read_timeout_ms: int):
        self._url = url
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
        self._stop_evt = threading.Event()
        self._reconnect_evt = threading.Event()
        self.reconnects = 0

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"cap-{url[-24:]}"
        )
        self._thread.start()

    def _reopen(self) -> "cv2.VideoCapture | None":
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cap = _open_cap(self._url,
                        open_timeout_ms=self._open_timeout_ms,
                        read_timeout_ms=self._read_timeout_ms)
        if cap is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.reconnects += 1
        return cap

    def _run(self):
        fails = 0
        while not self._stop_evt.is_set():
            if self._reconnect_evt.is_set():
                self._reconnect_evt.clear()
                self._cap = self._reopen()
                fails = 0

            if self._cap is None:
                time.sleep(1.0)
                self._cap = self._reopen()
                continue

            try:
                ok, frame = self._cap.read()
                pts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            except Exception:
                ok, frame, pts = False, None, -1.0
            mono = time.monotonic()

            if ok and frame is not None and frame.size > 0:
                fails = 0
                item = (True, frame, pts, mono)
            else:
                fails += 1
                item = (False, None, pts, mono)
                if fails >= self._MAX_FAILS:
                    self._cap = self._reopen()
                    fails = 0
                    continue

            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put(item)

    def read(self, timeout: float = 0.5) -> "tuple[bool, cv2.Mat | None, float, float]":
        """(ok, frame, pts_ms, mono) — всегда самый свежий кадр."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return False, None, -1.0, time.monotonic()

    def request_reconnect(self):
        self._reconnect_evt.set()

    def stop(self):
        self._stop_evt.set()
        if self._cap is not None:
            self._cap.release()
        self._thread.join(timeout=3.0)


# ─── Статистика прогона ───────────────────────────────────────────────────────

def _save_run_stats(frame_log: list, pts_log: list, saves_log: list,
                    diffs_log: list, out_dir: Path) -> None:
    import json as _json
    import numpy as _np

    total = len(frame_log)
    bad   = sum(1 for r in frame_log if r[3] == 0)
    ok    = total - bad

    mono_ok = [r[0] for r in frame_log if r[3] == 1]
    duration = (max(mono_ok) - min(mono_ok)) if len(mono_ok) >= 2 else 0.0

    by_url: dict = {}
    for r in frame_log:
        if r[3] == 1:
            by_url.setdefault(r[2], []).append(r[0])

    interval_stats: dict = {}
    for url, times in by_url.items():
        times.sort()
        ivs = _np.diff(times) * 1000
        if len(ivs):
            interval_stats[url] = {
                "mean_ms":   round(float(_np.mean(ivs)),          1),
                "median_ms": round(float(_np.median(ivs)),        1),
                "p95_ms":    round(float(_np.percentile(ivs, 95)),1),
                "p99_ms":    round(float(_np.percentile(ivs, 99)),1),
                "max_ms":    round(float(_np.max(ivs)),           1),
            }

    reconnects = 0
    prev_pts_r: dict = {}
    for r in pts_log:
        url, pts = r[2], r[3]
        if pts > 0:
            if url in prev_pts_r and pts < prev_pts_r[url] - 500:
                reconnects += 1
            prev_pts_r[url] = pts

    save_counts: dict = {}
    for r in saves_log:
        save_counts[r[3]] = save_counts.get(r[3], 0) + 1

    diff_stats: dict = {}
    if diffs_log:
        dv = _np.array([r[3] for r in diffs_log if r[3] > 0])
        if len(dv):
            diff_stats = {
                "mean":          round(float(_np.mean(dv)),          3),
                "p95":           round(float(_np.percentile(dv, 95)),3),
                "max":           round(float(_np.max(dv)),           3),
                "motion_events": len(diffs_log),
            }

    stats = {
        "duration_sec":    round(duration, 1),
        "frames_total":    total,
        "frames_ok":       ok,
        "frames_bad":      bad,
        "frames_bad_pct":  round(bad / total * 100, 2) if total else 0,
        "fps_effective":   round(ok / duration, 2) if duration > 0 else 0,
        "reconnects":      reconnects,
        "frame_intervals": interval_stats,
        "saves":           save_counts,
        "saves_total":     len(saves_log),
        "diff":            diff_stats,
    }

    path = out_dir / "run_stats.json"
    path.write_text(_json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  run_stats.json → {path}")


# ─── Графики ──────────────────────────────────────────────────────────────────

def _save_pts_chart(pts_log: list, cpu_log: list, out_dir: Path) -> None:
    """График соответствия меток времени: mono_s, wall clock (ts_msk), FFmpeg PTS."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as _ticker
    except ImportError:
        return
    if not pts_log:
        return

    from datetime import datetime as _dt
    def _parse_ts(s: str) -> float:
        try:
            return _dt.strptime(s[:22], "%Y%m%d_%H%M%S_%f").timestamp()
        except Exception:
            return 0.0

    from collections import defaultdict as _dd
    by_url: dict[str, list] = _dd(list)
    for row in pts_log:
        by_url[row[2]].append(row)

    n = len(by_url)
    n_cpu = 1 if cpu_log else 0
    n_rows = n * 3 + n_cpu
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows), squeeze=False)
    fig.suptitle("Соответствие меток времени: mono / wall / FFmpeg PTS", fontsize=11)
    ax_idx = 0

    for url_id, rows in by_url.items():
        rows = [r for r in rows if r[3] >= 0]  # -1.0 = timeout, невалидный PTS
        if not rows:
            ax_idx += 3
            continue

        mono = [r[0] for r in rows]
        pts  = [r[3] for r in rows]
        wall = [_parse_ts(r[1]) for r in rows]

        mono0 = mono[0]
        pts0  = pts[0]
        wall0 = wall[0]

        mono_s   = [(m - mono0)       for m in mono]
        pts_norm = [(p - pts0) / 1000 for p in pts]
        wall_s   = [(w - wall0)       for w in wall]

        pts_clean = list(pts_norm)
        offset = 0.0
        for i in range(1, len(pts_clean)):
            if pts_clean[i] + offset < pts_clean[i-1] + offset - 0.5:
                offset += pts_clean[i-1] - pts_clean[i] + 0.1
            pts_clean[i] += offset

        ax = axes[ax_idx][0]
        ax.scatter(mono, mono_s,    color="#2255cc", s=1,  marker="o", alpha=0.2, zorder=3, label="mono, с")
        ax.scatter(mono, wall_s,    color="#228833", s=2,  marker="s", alpha=0.2, zorder=3, label="wall clock, с")
        ax.scatter(mono, pts_clean, color="#cc5500", s=2,  marker="^", alpha=0.2, zorder=3, label="FFmpeg PTS (норм.), с")
        ax.set_ylabel("с от старта")
        ax.set_title(f"{url_id} — три метки времени (все в с от первого кадра)")
        ax.legend(loc="upper left", fontsize=8)
        ax.yaxis.set_major_locator(_ticker.MaxNLocator(8))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax_idx += 1

        ax = axes[ax_idx][0]
        drift_pts = [p - m for p, m in zip(pts_clean, mono_s)]
        ax.scatter(mono, drift_pts, color="#cc5500", s=2, marker="^", alpha=0.2)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_ylabel("с")
        ax.set_title(f"{url_id} — дрейф PTS − mono (с)")
        ax.yaxis.set_major_locator(_ticker.MaxNLocator(8))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax_idx += 1

        ax = axes[ax_idx][0]
        drift_wall = [w - m for w, m in zip(wall_s, mono_s)]
        ax.scatter(mono, drift_wall, color="#228833", s=8, marker="s", alpha=0.6)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_ylabel("с")
        ax.set_xlabel("время от старта, с")
        ax.set_title(f"{url_id} — дрейф wall − mono (с)")
        ax.yaxis.set_major_locator(_ticker.MaxNLocator(8))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax_idx += 1

    if cpu_log:
        ax = axes[ax_idx][0]
        cpu_t = [r[0] for r in cpu_log]
        cpu_v = [r[2] for r in cpu_log]
        ax.plot(cpu_t, cpu_v, color="#2255cc", linewidth=1.0)
        ax.set_ylabel("ЦПУ, %")
        ax.set_xlabel("время от старта, с")
        ax.set_title("CPU usage, %")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_locator(_ticker.MultipleLocator(20))
        ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    path = out_dir / "pts_chart.png"
    plt.savefig(str(path), dpi=120)
    plt.close()
    print(f"  pts_chart.png → {path}")


def _save_charts(frame_log: list, cpu_log: list, saves_log: list, diffs_log: list,
                 threshold: float, out_dir: Path) -> None:
    """Строит и сохраняет совмещённый PNG: интервалы кадров + дифы + сохранения + ЦПУ."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as _ticker
        import numpy as _np
    except ImportError:
        return
    if not frame_log:
        return

    _CAM_COLORS  = ["#e05c00", "#0066cc", "#228833", "#aa22aa"]
    _SAVE_COLORS = {"baseline": "#888888", "diff": "#cc3333",
                    "heartbeat": "#8844bb", "raw": "#dd8800"}
    _LINE_STYLES = ["-", "--", "-.", ":"]
    _SAVE_LEVELS = {"baseline": 1, "diff": 2, "heartbeat": 3, "raw": 4}
    _MARKERS     = ["*", (5, 1, 0), (6, 2, 0), (4, 1, 0)]

    by_url: dict[str, list] = defaultdict(list)
    for row in frame_log:
        by_url[row[2]].append(row)

    n_frame_rows = len(by_url)
    n_diffs_rows = 2 if diffs_log else 0
    n_saves_rows = 1 if saves_log else 0
    n_cpu_rows   = 1 if cpu_log else 0
    n_rows = n_frame_rows + n_diffs_rows + n_saves_rows + n_cpu_rows
    if n_rows == 0:
        return

    height_ratios = (
        [3.0] * n_frame_rows +
        [2.5] * n_diffs_rows +
        [1.5] * n_saves_rows +
        [2.0] * n_cpu_rows
    )
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, sum(height_ratios) * 1.5),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.suptitle("4_motion_diff_low — кадры и загрузка ЦПУ", fontsize=11)

    t_max = max((row[0] for row in frame_log), default=1)

    def _ts_parts(ts_str: str):
        try:
            p = ts_str.split("_")
            t = p[1]
            return int(t[0:2]), int(t[2:4]), int(t[4:6]), int(p[2])
        except Exception:
            return None

    def _ts_hms(ts_str: str) -> str:
        r = _ts_parts(ts_str)
        return f"{r[0]:02d}:{r[1]:02d}:{r[2]:02d}" if r else ts_str

    _t0_ts = frame_log[0][1] if frame_log else ""
    _t0_parts = _ts_parts(_t0_ts)
    _t0_abs = (_t0_parts[0]*3600 + _t0_parts[1]*60 + _t0_parts[2]) if _t0_parts else 0
    _x_left = -(_t0_abs % 60)

    _tick_range = t_max * 1.02 - _x_left
    _tick_step = next((s for s in [60, 120, 180, 300, 600, 900, 1800]
                       if _tick_range / s <= 20), 1800)
    _tick_positions = list(range(int(_x_left), int(t_max * 1.02) + _tick_step, _tick_step))

    def _x_time_fmt(x, _pos):
        total = int(_t0_abs + x)
        hh, mm, ss = total // 3600, (total % 3600) // 60, total % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    for ax_idx, (uid, rows) in enumerate(by_url.items()):
        ax = axes[ax_idx][0]
        monos = [r[0] for r in rows]
        ok_flags = [r[3] for r in rows]
        plaus_flags = [r[4] for r in rows]

        if len(monos) < 2:
            ax.set_title(f"{uid}: мало данных")
            continue

        intervals_ms = [(monos[i] - monos[i - 1]) * 1000 for i in range(1, len(monos))]
        t_axis = monos[1:]

        ax.scatter(t_axis, intervals_ms, color="#44aa44", s=14, alpha=0.45, marker="*")

        mean_iv = sum(intervals_ms) / len(intervals_ms)
        ax.axhline(mean_iv, color="blue", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(t_axis[0], mean_iv * 1.05,
                f"среднее {mean_iv:.0f} мс  ({1000/mean_iv:.1f} fps)",
                fontsize=8, color="blue")

        n_diff_ev = 0
        for row in rows:
            ev = row[5] if len(row) > 5 else ""
            if ev == "diff":
                ax.axvline(row[0], color="red", linewidth=0.7, alpha=0.5)
                n_diff_ev += 1

        n_ok = sum(ok_flags)
        n_pl = sum(1 for p in plaus_flags if p == 1)
        ax.set_ylabel("интервал, мс")
        ax.set_title(f"{uid}  |  кадров: {len(rows)}  ok: {n_ok}  plausible: {n_pl}")
        ax.set_xlim(_x_left, t_max * 1.02)
        y_lim_top = max(intervals_ms) * 1.2 if intervals_ms else 1
        ax.set_ylim(0, y_lim_top)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis.set_major_formatter(_ticker.FuncFormatter(_x_time_fmt))
        ax.xaxis.set_major_locator(_ticker.FixedLocator(_tick_positions))

        import matplotlib.lines as _mlines
        legend_handles = [
            _mlines.Line2D([], [], color="blue", linestyle="--", linewidth=0.8,
                           label=f"среднее {mean_iv:.0f} мс"),
        ]
        if n_diff_ev > 0:
            legend_handles.append(_mlines.Line2D([], [], color="red", linewidth=0.7,
                                                  label=f"diff-событие ({n_diff_ev})"))
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    if diffs_log:
        by_cam: dict[str, list] = defaultdict(list)
        for row in diffs_log:
            by_cam[row[2]].append(row)

        all_diffs = [r[3] for r in diffs_log if r[3] > 0]
        p95 = float(_np.percentile(all_diffs, 95)) if all_diffs else threshold * 10

        def _draw_diffs_ax(ax, ylim=None):
            for ci, (cam, rows) in enumerate(sorted(by_cam.items())):
                dt = [r[0] for r in rows]
                dv = [r[3] for r in rows]
                color = _CAM_COLORS[ci % len(_CAM_COLORS)]
                ax.scatter(dt, dv, color=color, s=16, alpha=0.45, label=cam,
                           marker=_MARKERS[ci % len(_MARKERS)])
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0)
            ax.text(0, threshold * 1.03, f"порог {threshold}", fontsize=8, color="red")
            ax.set_ylabel("diff")
            ax.set_xlim(_x_left, t_max * 1.02)
            if ylim is not None:
                ax.set_ylim(0, ylim)
            ax.yaxis.set_major_locator(_ticker.MaxNLocator(10))
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            ax.xaxis.set_major_formatter(_ticker.FuncFormatter(_x_time_fmt))
            ax.xaxis.set_major_locator(_ticker.FixedLocator(_tick_positions))

        ax = axes[n_frame_rows][0]
        _draw_diffs_ax(ax)
        ax.set_title(f"Frame diff — полный масштаб  ({len(diffs_log)} записей)")

        ax = axes[n_frame_rows + 1][0]
        _draw_diffs_ax(ax, ylim=p95 * 2.3)
        ax.set_title(f"Frame diff — до p95={p95:.2f}  (детальный вид)")

    if saves_log:
        ax = axes[n_frame_rows + n_diffs_rows][0]

        by_type_cam: dict[str, dict[str, list]] = {}
        for row in saves_log:
            t, _, cam, stype = row
            if stype not in by_type_cam:
                by_type_cam[stype] = {}
            if cam not in by_type_cam[stype]:
                by_type_cam[stype][cam] = []
            by_type_cam[stype][cam].append(t)

        cam_list = sorted({row[2] for row in saves_log})
        cam_mk = {cam: _MARKERS[i % len(_MARKERS)] for i, cam in enumerate(cam_list)}

        def _cam_short(cam: str) -> str:
            parts = [p for p in cam.split("_") if p and p != "URL"]
            return parts[-1] if parts else cam

        y_pos: dict[tuple, float] = {}
        ytick_pos: list[float] = []
        ytick_labels: list[str] = []
        y = 1.0
        for stype in sorted(by_type_cam, key=lambda s: _SAVE_LEVELS.get(s, 99)):
            for cam in sorted(by_type_cam[stype]):
                y_pos[(stype, cam)] = y
                ytick_pos.append(y)
                ytick_labels.append(f"{stype} / {_cam_short(cam)}")
                y += 1.0
            y += 0.5

        for (stype, cam), yv in y_pos.items():
            times = by_type_cam[stype][cam]
            color = _SAVE_COLORS.get(stype, "#aaaaaa")
            ax.scatter(times, [yv] * len(times), color=color, s=35,
                       marker=cam_mk[cam], alpha=0.85, zorder=3)

        for stype in sorted(_SAVE_LEVELS, key=_SAVE_LEVELS.get):
            if stype in by_type_cam:
                ax.scatter([], [], color=_SAVE_COLORS.get(stype, "#aaaaaa"),
                           s=35, marker="o", label=stype)

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels, fontsize=7)
        ax.set_xlim(_x_left, t_max * 1.02)
        ax.set_ylim(0, y)
        ax.set_title(f"Сохранения кадров  ({len(saves_log)} событий)")
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis.set_major_formatter(_ticker.FuncFormatter(_x_time_fmt))
        ax.xaxis.set_major_locator(_ticker.FixedLocator(_tick_positions))

    if cpu_log:
        ax = axes[n_frame_rows + n_diffs_rows + n_saves_rows][0]
        cpu_t = [r[0] for r in cpu_log]
        cpu_v = [r[2] for r in cpu_log]
        ax.plot(cpu_t, cpu_v, color="#2255cc", linewidth=1.2)
        ax.fill_between(cpu_t, cpu_v, alpha=0.18, color="#2255cc")
        if cpu_v:
            mean_cpu = sum(cpu_v) / len(cpu_v)
            ax.axhline(mean_cpu, color="orange", linestyle="--", linewidth=0.8)
            ax.text(cpu_t[0] if cpu_t else 0, mean_cpu + 1,
                    f"среднее {mean_cpu:.1f}%", fontsize=8, color="orange")
        ax.set_ylabel("ЦПУ, %")
        ax.set_ylim(0, 105)
        ax.set_xlim(_x_left, t_max * 1.02)
        ax.set_title("Загрузка ЦПУ (все ядра, %)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis.set_major_formatter(_ticker.FuncFormatter(_x_time_fmt))
        ax.xaxis.set_major_locator(_ticker.FixedLocator(_tick_positions))

    axes[-1][0].set_xlabel("время МСК")

    plt.tight_layout()
    chart_path = out_dir / "charts.png"
    plt.savefig(str(chart_path), dpi=120)
    plt.close()
    print(f"  charts.png → {chart_path}")


def _regen_charts(run_dir: Path) -> None:
    """Перечитывает CSV из существующего run-каталога и перегенерирует графики."""
    import csv as _csv
    import json as _json

    def _load(name: str) -> list:
        p = run_dir / name
        if not p.exists():
            print(f"  [!] {name} не найден — пропущено")
            return []
        with open(p, encoding="utf-8") as f:
            rows = list(_csv.reader(f))
        return rows[1:] if rows else []

    def _f(v, default=0.0):
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _i(v, default=0):
        try:
            return int(v)
        except (ValueError, TypeError):
            return default

    print(f"Перегенерация графиков из: {run_dir}")

    frame_log = [[_f(r[0]), r[1], r[2], _i(r[3]), _i(r[4]), r[5] if len(r) > 5 else ""]
                 for r in _load("frames.csv") if len(r) >= 5]
    saves_log = [[_f(r[0]), r[1], r[2], r[3]]
                 for r in _load("saves.csv") if len(r) >= 4]
    diffs_log = [[_f(r[0]), r[1], r[2], _f(r[3])]
                 for r in _load("diffs.csv") if len(r) >= 4]
    cpu_log   = [[_f(r[0]), r[1], _f(r[2])]
                 for r in _load("cpu.csv") if len(r) >= 3]
    pts_log   = [[_f(r[0]), r[1], r[2], _f(r[3])]
                 for r in _load("pts.csv") if len(r) >= 4]

    threshold = 10.0
    params_path = run_dir / "run_params.json"
    if params_path.exists():
        try:
            threshold = float(_json.loads(params_path.read_text(encoding="utf-8")).get("threshold", 10.0))
        except Exception:
            pass

    _save_charts(frame_log, cpu_log, saves_log, diffs_log, threshold, run_dir)
    _save_pts_chart(pts_log, cpu_log, run_dir)
    _save_run_stats(frame_log, pts_log, saves_log, diffs_log, run_dir)
    osd_log = _regen_osd_from_images(run_dir)
    if osd_log:
        _save_osd_chart(osd_log, run_dir)


# ─── OSD-анализ по сохранённым кадрам ─────────────────────────────────────────

def _parse_img_filename(stem: str) -> "tuple[str, str, str] | None":
    """Парсит имя файла → (cam_name, ts_str 'YYYY-MM-DD HH:MM:SS', img_type).

    Формат: cam_01_9_u_YYYYMMDD_HHMMSS_ffffff_msk_type
    """
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            cam_name = "_".join(parts[:i])
            date_p = p
            time_p = parts[i + 1] if i + 1 < len(parts) else "000000"
            img_type = parts[-1]
            ts_str = (f"{date_p[:4]}-{date_p[4:6]}-{date_p[6:]} "
                      f"{time_p[:2]}:{time_p[2:4]}:{time_p[4:6]}")
            return cam_name, ts_str, img_type
    return None


def _regen_osd_from_images(run_dir: Path) -> list:
    """Извлекает OSD-метки из сохранённых изображений U-камеры.

    Первый проход: строит LOW-шаблоны из имён файлов + изображений.
    Второй проход: извлекает OSD-время, сохраняет osd_times.csv.
    Возвращает osd_log: [[wall_ts_str, cam, img_type, osd_ts_str, drift_sec], ...]
    """
    try:
        from common.utils.osd_time import (
            build_low_templates_from_image,
            extract_osd_time_low,
            low_templates_complete,
            _LOW_TEMPLATES,
        )
    except ImportError:
        return []

    images_dir = run_dir / "images"
    if not images_dir.exists():
        return []

    imgs = sorted(images_dir.rglob("*.jpg"))
    u_imgs = [p for p in imgs if "_u_" in p.name or "_9_u_" in p.name]
    if not u_imgs:
        return []

    if not low_templates_complete():
        print(f"  [OSD] Построение LOW-шаблонов из {len(u_imgs)} изображений…")
        for img_path in u_imgs:
            info = _parse_img_filename(img_path.stem)
            if info is None:
                continue
            _, ts_str, _ = info
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            build_low_templates_from_image(frame, ts_str)
            if low_templates_complete():
                break
        print(f"  [OSD] Шаблоны: {sorted(_LOW_TEMPLATES.keys())} ({len(_LOW_TEMPLATES)}/12)")

    import csv as _csv
    from datetime import datetime as _dt

    osd_log: list = []
    ok_count = 0

    for img_path in u_imgs:
        info = _parse_img_filename(img_path.stem)
        if info is None:
            continue
        cam_name, wall_ts_str, img_type = info

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        osd_dt = extract_osd_time_low(frame)
        if osd_dt is None:
            osd_ts_str = ""
            drift_sec  = None
        else:
            osd_ts_str = osd_dt.strftime("%Y-%m-%d %H:%M:%S")
            try:
                wall_dt = _dt.strptime(wall_ts_str, "%Y-%m-%d %H:%M:%S")
                drift_sec = round((osd_dt - wall_dt).total_seconds(), 1)
            except Exception:
                drift_sec = None
            ok_count += 1

        osd_log.append([wall_ts_str, cam_name, img_type, osd_ts_str,
                        "" if drift_sec is None else drift_sec])

    print(f"  [OSD] Извлечено: {ok_count}/{len(u_imgs)} меток")

    if osd_log:
        csv_path = run_dir / "osd_times.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(["wall_ts", "cam", "img_type", "osd_ts", "drift_sec"])
            writer.writerows(osd_log)
        print(f"  osd_times.csv → {csv_path}")

    return osd_log


def _save_osd_chart(osd_log: list, out_dir: Path) -> None:
    """График сравнения: OSD-время камеры vs wall-clock (по сохранённым кадрам)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as _ticker
    except ImportError:
        return

    from datetime import datetime as _dt

    rows_with_osd = [r for r in osd_log if r[3]]
    if not rows_with_osd:
        print("  [OSD] Нет распознанных меток — график пропущен")
        return

    def _parse(s: str) -> float:
        try:
            return _dt.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            return 0.0

    wall_ts   = [_parse(r[0]) for r in rows_with_osd]
    osd_ts    = [_parse(r[3]) for r in rows_with_osd]
    img_type  = [r[2]         for r in rows_with_osd]
    drift_vals = [(o - w) for o, w in zip(osd_ts, wall_ts)]

    t0 = wall_ts[0]
    wall_rel = [(t - t0) for t in wall_ts]
    osd_rel  = [(t - t0) for t in osd_ts]

    _TYPE_COLOR = {"baseline": "#888888", "heartbeat": "#8844bb",
                   "diff": "#228833", "yolo": "#cc3333"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), squeeze=False,
                             gridspec_kw={"height_ratios": [2.5, 1.5]})
    fig.suptitle("OSD-метка камеры vs wall-clock (по сохранённым кадрам)", fontsize=11)

    ax = axes[0][0]
    ax.plot(wall_rel, wall_rel, color="#2255cc", linewidth=1.2, label="wall (эталон)")
    ax.scatter(wall_rel, osd_rel, s=35, zorder=4,
               c=[_TYPE_COLOR.get(t, "#888888") for t in img_type], label=None)
    for itype, color in sorted(_TYPE_COLOR.items()):
        if any(t == itype for t in img_type):
            ax.scatter([], [], color=color, s=35, label=itype)
    ax.set_ylabel("секунды от старта")
    ax.set_xlabel("wall (с от старта)")
    ax.legend(loc="upper left", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_title("Время камеры OSD (точки) и wall-clock (прямая). Идеал = совпадение.")

    ax = axes[1][0]
    ax.scatter(wall_rel, drift_vals, s=35, zorder=4,
               c=[_TYPE_COLOR.get(t, "#888888") for t in img_type])
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    if drift_vals:
        mean_d = sum(drift_vals) / len(drift_vals)
        ax.axhline(mean_d, color="orange", linewidth=1.0, linestyle="--")
        ax.text(wall_rel[0] if wall_rel else 0, mean_d + 0.3,
                f"среднее {mean_d:+.1f}с", fontsize=8, color="orange")
    ax.set_ylabel("OSD − wall, с")
    ax.set_xlabel("wall (с от старта)")
    ax.yaxis.set_major_locator(_ticker.MaxNLocator(8))
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_title("Дрейф: OSD − wall-clock (сек). Отрицательное = камера отстаёт.")

    plt.tight_layout()
    chart_path = out_dir / "osd_chart.png"
    plt.savefig(str(chart_path), dpi=120)
    plt.close()
    print(f"  osd_chart.png → {chart_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Детекция движения по LOW-потоку CAM_*_URL, сохранение LOW-кадра при срабатывании"
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV, help="Путь к .env")
    parser.add_argument(
        "--crop-rel", type=str, default=None, metavar="X,Y,W,H",
        help="Глобальная обрезка склеенного кадра, доли 0…1 (перебивает MOTION_CROP_REL из .env)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Порог mean abs diff (0–255); иначе MOTION_DIFF_THRESHOLD из .env или 10",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Каталог для кадров (по умолчанию .output/cameras/4_motion_diff_low/run_<ts>/)",
    )
    parser.add_argument("--tcp", action="store_true", help="RTSP через TCP")
    parser.add_argument("--open-timeout-ms", type=int, default=10000)
    parser.add_argument("--read-timeout-ms", type=int, default=10000)
    parser.add_argument("--stimeout-us", type=int, default=3_000_000,
                        help="FFmpeg stimeout, мкс")
    parser.add_argument("--min-laplacian-var", type=float, default=12.0,
                        help="Мин. variance(Laplacian) — ниже считаем битым кадром (HEVC)")
    parser.add_argument("--min-gray-std", type=float, default=2.5,
                        help="Мин. std(яркость) — отсекает однотонный серый")
    parser.add_argument("--baseline-attempts", type=int, default=16,
                        help="Попыток read() для базового кадра")
    parser.add_argument(
        "--heartbeat-sec", type=float, default=None,
        help="Раз в N сек сохранять кадр без движения; 0 = выкл; иначе MOTION_HEARTBEAT_SEC из .env или 600",
    )
    parser.add_argument(
        "--duration", type=float, default=0, metavar="SEC",
        help="Остановиться через N секунд после старта. 0 = бесконечно (default).",
    )
    parser.add_argument(
        "--cam-ts", action="store_true",
        help="Добавлять время камеры из RTCP NTP в имя файла (cam_YYYYMMDD_HHMMSS). "
             "Открывает отдельное RTSP-соединение для каждого URL.",
    )
    parser.add_argument(
        "--cpu-interval", type=float, default=2.0, metavar="SEC",
        help="Интервал замера загрузки ЦПУ (сек). Требует psutil. 0 = не замерять.",
    )
    parser.add_argument(
        "--regen-from", type=Path, default=None, metavar="RUN_DIR",
        help="Перегенерировать графики из существующей директории прогона без захвата.",
    )
    args = parser.parse_args()

    if args.regen_from is not None:
        _regen_charts(args.regen_from)
        return 0

    if not args.env.is_file():
        print(f"Файл .env не найден: {args.env}", file=sys.stderr)
        return 1

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Нужен пакет python-dotenv: pip install python-dotenv", file=sys.stderr)
        return 1

    load_dotenv(args.env, override=True)

    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_from = "аргумент --threshold"
    else:
        raw_t = (os.environ.get("MOTION_DIFF_THRESHOLD") or "").strip()
        threshold = float(raw_t) if raw_t else 10.0
        threshold_from = f"MOTION_DIFF_THRESHOLD={raw_t!r} (.env)" if raw_t else "встроенное 10"

    if args.heartbeat_sec is not None:
        heartbeat_sec = max(0.0, float(args.heartbeat_sec))
        heartbeat_from = "аргумент --heartbeat-sec"
    else:
        raw_hb = (os.environ.get("MOTION_HEARTBEAT_SEC") or "").strip()
        if raw_hb:
            heartbeat_sec = max(0.0, float(raw_hb))
            heartbeat_from = f"MOTION_HEARTBEAT_SEC={raw_hb!r} (.env)"
        else:
            heartbeat_sec = 600.0
            heartbeat_from = "встроенное 600 с"

    out_dir = args.output
    if out_dir is None:
        out_dir = DEFAULT_OUTPUT_PARENT / f"run_{ts_for_dir()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Лог в файл (tee stdout+stderr → run.log)
    _log_file = open(out_dir / "run.log", "w", encoding="utf-8")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _ffmpeg_capture_options(
        use_tcp=args.tcp, stimeout_us=args.stimeout_us,
    )

    cameras = _collect_cam_urls()
    active: list[tuple[str, str]] = [(k, v) for k, v in cameras if not _skip_url(v)]
    if not active:
        print("Нет активных rtsp:// URL (CAM_<stem>_URL без плейсхолдеров).", file=sys.stderr)
        return 1

    crop_cli = (args.crop_rel or "").strip() or None
    global_crop, global_crop_from = resolve_global_crop(
        crop_rel_arg=crop_cli, motion_crop_env=os.environ.get("MOTION_CROP_REL"),
    )
    if crop_cli and global_crop is None:
        return 1
    crop_by_cam = crop_map_for_cameras(active, global_crop=global_crop)

    low_unique = sorted(set(url for _, url in active))
    readers: dict[str, _CapReader] = {}
    for url in low_unique:
        cap = _open_cap(url, open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms)
        if cap is None:
            print(f"  [!] LOW не удалось открыть: {redact_url(url)[:80]}…", file=sys.stderr)
        else:
            readers[url] = _CapReader(url, cap, args.open_timeout_ms, args.read_timeout_ms)

    opened_vars: list[tuple[str, str]] = [(vn, url) for vn, url in active if url in readers]
    if not opened_vars:
        print("Ни одна камера не открылась (проверьте URL и сеть).", file=sys.stderr)
        for r in readers.values():
            r.stop()
        return 1

    vars_by_low: dict[str, list[str]] = defaultdict(list)
    for vn, lu in opened_vars:
        vars_by_low[lu].append(vn)

    url_id: dict[str, str] = {lu: vlist[0] for lu, vlist in vars_by_low.items()}

    rtcp_workers: dict[str, _RtcpWorker] = {}
    if args.cam_ts:
        for url in readers:
            w = _RtcpWorker(url)
            w.start()
            rtcp_workers[url] = w

    prev_gray: dict[str, np.ndarray | None] = {vn: None for vn, _ in opened_vars}
    last_good_frame: dict[str, np.ndarray | None] = {vn: None for vn, _ in opened_vars}
    last_heartbeat: dict[str, float] = {vn: time.monotonic() for vn, _ in opened_vars}
    prev_pts: dict[str, float] = {lu: -1.0 for lu in low_unique}
    _MAX_LOW_IMPLAUSIBLE = 40
    _low_implausible: dict[str, int] = defaultdict(int)

    frame_log: list[list] = []   # [mono_s, ts_msk, url_id, ok, plausible, event]
    saves_log: list[list] = []   # [mono_s, ts_msk, cam, save_type]
    pts_log:   list[list] = []   # [mono_s, ts_msk, url_id, pts_ms]
    diffs_log: list[list] = []   # [mono_s, ts_msk, cam, diff]
    cpu_log:   list[list] = []

    import json as _json
    from common.utils.time_msk import ts_iso as _ts_iso
    (out_dir / "run_params.json").write_text(_json.dumps({
        "started_at_msk": _ts_iso(),
        "script": "4_motion_diff_low.py",
        "threshold": threshold,
        "heartbeat_sec": heartbeat_sec,
        "tcp": args.tcp,
        "duration_sec": args.duration,
        "cam_ts": args.cam_ts,
        "cpu_interval": args.cpu_interval,
        "cameras": [vn for vn, _ in opened_vars],
        "crop_global": list(global_crop) if global_crop else None,
        "output": str(out_dir),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    def _cam_now(calib) -> "datetime | None":
        if calib is None:
            return None
        return datetime.fromtimestamp(calib.ntp_unix + (time.monotonic() - calib.received_at))

    def _ts(calib=None) -> str:
        pc = ts_for_file()
        if args.cam_ts:
            cam_dt = _cam_now(calib)
            if cam_dt is not None:
                return f"{ts_cam_for_file(cam_dt)}_{pc}"
        return pc

    duration_desc = f"{args.duration:.0f} сек" if args.duration > 0 else "бесконечно"
    print(f"Порог:     {threshold}  [{threshold_from}]")
    print(f"Пульс:     {heartbeat_sec} сек  [{heartbeat_from}]")
    print(f"Длит.:     {duration_desc}")
    print(f"Вывод:     {out_dir}")
    print(f"Камеры ({len(opened_vars)}): {', '.join(vn for vn, _ in opened_vars)}")
    print(f"Обрезка:   {global_crop!r}  [{global_crop_from}]")
    for vn, _ in opened_vars:
        c = crop_by_cam[vn]
        cr = (f"x,y,w,h={c}" if c is not None else "полный кадр")
        print(f"  └ {vn}: {cr}")
    print("Останов: Ctrl+C\n")

    t_start = time.monotonic()

    cpu_monitor = _CpuMonitor(interval=max(args.cpu_interval, 0.5))
    cpu_active = args.cpu_interval > 0 and cpu_monitor.start(t_start)
    if args.cpu_interval > 0 and not cpu_active:
        print("  [!] psutil не установлен — мониторинг ЦПУ недоступен (pip install psutil)")

    print("Базовый кадр…")
    for low_u, var_list in vars_by_low.items():
        reader_bl = readers[low_u]
        frame_l = None
        for _ in range(args.baseline_attempts):
            ok_bl, f_bl, _, _ = reader_bl.read(timeout=1.0)
            if ok_bl and f_bl is not None and frame_decode_plausible(
                    f_bl, min_laplacian_var=args.min_laplacian_var, min_gray_std=args.min_gray_std):
                frame_l = f_bl
                break
        if frame_l is None:
            for vn in var_list:
                print(f"  [!] {vn}: нет годного кадра для baseline", file=sys.stderr)
            continue
        for vn in var_list:
            fl = apply_crop_optional(frame_l, crop_by_cam[vn])
            prev_gray[vn] = _prepare_gray(fl)
            last_good_frame[vn] = fl
            stem = _stem_from_env_var(vn)
            calib0 = rtcp_workers[low_u].get_calib() if rtcp_workers else None
            bname = f"{stem}_{_ts(calib0)}_baseline.jpg"
            cv2.imwrite(str(images_dir / bname), fl)
            saves_log.append([round(time.monotonic() - t_start, 4), ts_for_file(), vn, "baseline"])
            print(f"  {bname}")
    print()

    deadline = (time.monotonic() + args.duration) if args.duration > 0 else None

    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                print(f"\nДлительность {args.duration:.0f} сек истекла — останов.")
                break

            for low_u, var_list in vars_by_low.items():
                reader = readers.get(low_u)
                if reader is None:
                    continue

                ok_l, frame_l, _pts, _t = reader.read(timeout=0.5)
                _ts_str = ts_for_file()
                pts_log.append([round(_t - t_start, 4), _ts_str, url_id[low_u], round(_pts, 1)])

                if not ok_l or frame_l is None:
                    frame_log.append([round(_t - t_start, 4), _ts_str, url_id[low_u], 0, 0, ""])
                    continue

                if _pts > 0 and _pts == prev_pts[low_u]:
                    continue
                prev_pts[low_u] = _pts

                _plausible = frame_decode_plausible(
                    frame_l, min_laplacian_var=args.min_laplacian_var, min_gray_std=args.min_gray_std)
                frame_log.append([round(_t - t_start, 4), _ts_str, url_id[low_u], 1, int(_plausible), ""])

                if not _plausible:
                    _low_implausible[low_u] += 1
                    if _low_implausible[low_u] >= _MAX_LOW_IMPLAUSIBLE:
                        print(f"  [!] {_MAX_LOW_IMPLAUSIBLE} битых кадров — переподключение", file=sys.stderr)
                        reader.request_reconnect()
                        _low_implausible[low_u] = 0
                        for vn in var_list:
                            prev_gray[vn] = None
                    continue
                _low_implausible[low_u] = 0

                _now = time.monotonic()
                for vn in var_list:
                    frame_u = apply_crop_optional(frame_l, crop_by_cam[vn])
                    gray = _prepare_gray(frame_u)
                    last_good_frame[vn] = frame_u

                    prev = prev_gray[vn]
                    if prev is None:
                        prev_gray[vn] = gray
                        continue

                    diff = _mean_abs_diff(prev, gray)
                    diffs_log.append([round(_now - t_start, 4), _ts_str, vn, round(diff, 3)])
                    prev_gray[vn] = gray

                    if diff > threshold:
                        calib = rtcp_workers[low_u].get_calib() if rtcp_workers else None
                        stem = _stem_from_env_var(vn)
                        fname = f"{stem}_{_ts(calib)}_diff{diff:.1f}.jpg"
                        cv2.imwrite(str(images_dir / fname), frame_u)
                        frame_log[-1][5] = "diff"
                        saves_log.append([round(_now - t_start, 4), _ts_str, vn, "diff"])
                        print(f"  {fname}  diff={diff:.2f}")

                for vn in var_list:
                    if heartbeat_sec <= 0:
                        continue
                    now = time.monotonic()
                    if now - last_heartbeat[vn] < heartbeat_sec:
                        continue
                    hb = last_good_frame.get(vn)
                    if hb is None:
                        continue
                    last_heartbeat[vn] = now
                    calib = rtcp_workers[low_u].get_calib() if rtcp_workers else None
                    stem = _stem_from_env_var(vn)
                    hb_name = f"{stem}_{_ts(calib)}_heartbeat.jpg"
                    cv2.imwrite(str(images_dir / hb_name), hb)
                    frame_log[-1][5] = "heartbeat"
                    saves_log.append([round(time.monotonic() - t_start, 4), ts_for_file(), vn, "heartbeat"])
                    print(f"  пульс {hb_name}")

    except KeyboardInterrupt:
        print("\nОстанов по Ctrl+C")
    finally:
        for r in readers.values():
            r.stop()
        for w in rtcp_workers.values():
            w.stop()
        cpu_log = cpu_monitor.stop()

        print("\nСохранение результатов…")

        if frame_log:
            with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "url_id", "ok", "plausible", "event"])
                writer.writerows(frame_log)
            print(f"  frames.csv: {len(frame_log)} строк")

        if saves_log:
            with open(out_dir / "saves.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cam", "type"])
                writer.writerows(saves_log)
            print(f"  saves.csv:  {len(saves_log)} записей")

        if diffs_log:
            with open(out_dir / "diffs.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cam", "diff"])
                writer.writerows(diffs_log)
            print(f"  diffs.csv:  {len(diffs_log)} записей")

        if pts_log:
            with open(out_dir / "pts.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "url_id", "pts_ms"])
                writer.writerows(pts_log)
            print(f"  pts.csv:    {len(pts_log)} записей")

        if cpu_log:
            with open(out_dir / "cpu.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cpu_pct"])
                writer.writerows(cpu_log)
            print(f"  cpu.csv:    {len(cpu_log)} замеров")

        _save_charts(frame_log, cpu_log, saves_log, diffs_log, threshold, out_dir)
        _save_pts_chart(pts_log, cpu_log, out_dir)
        _save_run_stats(frame_log, pts_log, saves_log, diffs_log, out_dir)

        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _log_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
