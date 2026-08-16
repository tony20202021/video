"""Общие примитивы для скриптов захвата/анализа камер (S4, S5, S6, ...).

Содержит:
  Tee           — дублирует stdout/stderr в лог-файл
  CapReader     — фоновый поток чтения RTSP (Queue(1), последний кадр, авто-реконнект)
  CpuMonitor    — фоновый поток замера CPU (psutil)
  save_run_stats      — run_stats.json
  save_pts_chart      — pts_chart.png
  save_charts         — charts.png (интервалы кадров + диффы + сохранения + CPU)
  parse_img_filename  — имя файла → (cam_name, ts_str, img_type)
  regen_osd_from_images — OSD-метки из сохранённых изображений → osd_log
  save_osd_chart      — osd_chart.png
"""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2

from common.utils.motion_utils import open_cap
from common.utils.time_msk import ts_for_file

logger = logging.getLogger(__name__)


# ─── Tee ──────────────────────────────────────────────────────────────────────

class Tee:
    """Пишет одновременно в оригинальный поток и в файл."""

    def __init__(self, stream, fobj):
        self._stream = stream
        self._fobj   = fobj

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


# ─── CapReader ─────────────────────────────────────────────────────────────────

class CapReader:
    """Фоновый поток: непрерывно читает VideoCapture, держит только последний кадр.

    Queue(maxsize=1) гарантирует: main-поток всегда получает самый свежий кадр
    без буферного лага, даже если обработка (YOLO, imwrite) занимает > 1 кадра.
    """

    _MAX_FAILS = 5

    def __init__(self, url: str, cap: "cv2.VideoCapture",
                 open_timeout_ms: int, read_timeout_ms: int):
        self._url              = url
        self._open_timeout_ms  = open_timeout_ms
        self._read_timeout_ms  = read_timeout_ms
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
        self._stop_evt         = threading.Event()
        self._reconnect_evt    = threading.Event()
        self.reconnects        = 0

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
        cap = open_cap(self._url,
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
        """Попросить поток переподключиться (например, при битых кадрах)."""
        self._reconnect_evt.set()

    def stop(self):
        self._stop_evt.set()
        if self._cap is not None:
            self._cap.release()
        try:
            self._thread.join(timeout=3.0)
        except KeyboardInterrupt:
            pass


# ─── CPU frequency ────────────────────────────────────────────────────────────

import sys as _sys

# Persistent PDH query state for Windows CPU frequency (% Processor Performance × base MHz).
# CallNtPowerInformation.CurrentMhz reports only P-state steps (e.g. 1600/2900);
# PDH counter gives time-weighted average like Task Manager "Speed".
_cpu_freq_pdh: dict = {}
# PDH state for % Processor Utility (Task Manager view — frequency-adjusted).
_cpu_utility_pdh: dict = {}

def _read_cpu_freq_mhz() -> float:
    """Возвращает текущую среднюю частоту CPU в МГц (кросс-платформа).

    Windows: PDH «% Processor Performance» × базовая частота (реестр).
             Первый вызов инициализирует счётчик и возвращает 0.
    Linux:   psutil.cpu_freq().current.
    """
    try:
        if _sys.platform == "win32":
            import ctypes, winreg

            _s = _cpu_freq_pdh
            if not _s:
                pdh = ctypes.windll.pdh
                try:
                    with winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE,
                        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                    ) as _k:
                        base_mhz = int(winreg.QueryValueEx(_k, "~MHz")[0])
                except Exception:
                    base_mhz = 2900

                class _FMT(ctypes.Structure):
                    _fields_ = [("CStatus", ctypes.c_ulong),
                                 ("doubleValue", ctypes.c_double)]

                hq = ctypes.c_void_p()
                hc = ctypes.c_void_p()
                if pdh.PdhOpenQueryW(None, 0, ctypes.byref(hq)) != 0:
                    return 0.0
                path = r"\Processor Information(_Total)\% Processor Performance"
                if pdh.PdhAddEnglishCounterW(hq, ctypes.c_wchar_p(path), 0, ctypes.byref(hc)) != 0:
                    pdh.PdhCloseQuery(hq)
                    return 0.0
                pdh.PdhCollectQueryData(hq)          # prime — первый замер без дельты
                _s.update(pdh=pdh, hq=hq, hc=hc, base=base_mhz, FMT=_FMT)
                return 0.0

            pdh = _s["pdh"]
            if pdh.PdhCollectQueryData(_s["hq"]) != 0:
                return 0.0
            val = _s["FMT"]()
            rc = pdh.PdhGetFormattedCounterValue(
                _s["hc"], 0x00000200, None, ctypes.byref(val))   # 0x200 = PDH_FMT_DOUBLE
            if rc == 0 and val.CStatus == 0:
                return round(_s["base"] * val.doubleValue / 100.0, 1)
        else:
            import psutil as _ps
            f = _ps.cpu_freq()
            if f:
                return round(f.current, 1)
    except Exception:
        pass
    return 0.0


def _read_cpu_utility_pct() -> float:
    """Windows: PDH «% Processor Utility» — как диспетчер задач (учитывает троттлинг).

    Первый вызов инициализирует счётчик и возвращает 0.
    На Linux всегда возвращает 0.
    """
    try:
        if _sys.platform != "win32":
            return 0.0
        import ctypes
        _s = _cpu_utility_pdh
        if not _s:
            pdh = ctypes.windll.pdh

            class _FMT(ctypes.Structure):
                _fields_ = [("CStatus", ctypes.c_ulong),
                             ("doubleValue", ctypes.c_double)]

            hq = ctypes.c_void_p()
            hc = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(hq)) != 0:
                return 0.0
            path = r"\Processor Information(_Total)\% Processor Utility"
            if pdh.PdhAddEnglishCounterW(hq, ctypes.c_wchar_p(path), 0, ctypes.byref(hc)) != 0:
                pdh.PdhCloseQuery(hq)
                return 0.0
            pdh.PdhCollectQueryData(hq)          # prime
            _s.update(pdh=pdh, hq=hq, hc=hc, FMT=_FMT)
            return 0.0
        pdh = _s["pdh"]
        if pdh.PdhCollectQueryData(_s["hq"]) != 0:
            return 0.0
        val = _s["FMT"]()
        rc = pdh.PdhGetFormattedCounterValue(
            _s["hc"], 0x00000200, None, ctypes.byref(val))
        if rc == 0 and val.CStatus == 0:
            return round(min(100.0, max(0.0, val.doubleValue)), 1)
    except Exception:
        pass
    return 0.0


def _read_cpu_freq_stepped_mhz() -> float:
    """Ступенчатая частота через CallNtPowerInformation (только P-state шаги)."""
    try:
        if _sys.platform == "win32":
            import ctypes, ctypes.wintypes, psutil as _ps

            class _PPI(ctypes.Structure):
                _fields_ = [("Number",           ctypes.wintypes.ULONG),
                             ("MaxMhz",           ctypes.wintypes.ULONG),
                             ("CurrentMhz",       ctypes.wintypes.ULONG),
                             ("MhzLimit",         ctypes.wintypes.ULONG),
                             ("MaxIdleState",     ctypes.wintypes.ULONG),
                             ("CurrentIdleState", ctypes.wintypes.ULONG)]

            n   = _ps.cpu_count() or 1
            buf = (_PPI * n)()
            rc  = ctypes.windll.powrprof.CallNtPowerInformation(
                11, None, 0, buf, ctypes.sizeof(buf))
            if rc == 0:
                return round(sum(buf[i].CurrentMhz for i in range(n)) / n, 1)
    except Exception:
        pass
    return 0.0


# ─── CpuMonitor ───────────────────────────────────────────────────────────────

class CpuMonitor:
    """Фоновый поток: периодически замеряет cpu_percent() через psutil.

    per_process=True — измеряет только текущий процесс (0-100%, как диспетчер задач).
    per_process=False — система в целом (по умолчанию, для live-скриптов).
    """

    def __init__(self, interval: float = 2.0, per_process: bool = False) -> None:
        self._interval   = interval
        self._per_process = per_process
        self._log: list[list] = []
        self._stop  = threading.Event()
        self._lock  = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name="cpu-monitor")
        self._psutil = None
        self._proc   = None
        self._t_start = 0.0

    def start(self, t_start: float) -> bool:
        try:
            import psutil
            self._psutil = psutil
            if self._per_process:
                self._proc = psutil.Process()
                self._proc.cpu_percent()        # warm-up
            else:
                psutil.cpu_percent()            # warm-up
        except ImportError:
            return False
        self._t_start = t_start
        self._thread.start()
        return True

    def snapshot(self) -> list[list]:
        """Возвращает копию накопленного лога без остановки потока."""
        with self._lock:
            return list(self._log)

    def stop(self) -> list[list]:
        self._stop.set()
        if self._thread.is_alive():
            try:
                self._thread.join(timeout=max(self._interval + 1, 3))
            except KeyboardInterrupt:
                pass
        with self._lock:
            return list(self._log)

    def _run(self) -> None:
        n_cpu = max(1, self._psutil.cpu_count() or 1)
        while not self._stop.is_set():
            if self._per_process and self._proc:
                # cpu_percent() per-process returns 0-100*N%; normalize to 0-100%
                raw = self._proc.cpu_percent(interval=self._interval)
                cpu = min(100.0, raw / n_cpu)
            else:
                cpu = self._psutil.cpu_percent(interval=self._interval)
            with self._lock:
                self._log.append([
                    round(time.monotonic() - self._t_start, 2),
                    ts_for_file(),
                    round(cpu, 1),
                    _read_cpu_freq_mhz(),          # PDH: непрерывная (% Processor Performance)
                    _read_cpu_freq_stepped_mhz(),  # PPI: ступенчатая (P-state шаги)
                    _read_cpu_utility_pct(),       # PDH: % Processor Utility (как диспетчер задач)
                ])


# ─── run_stats.json ───────────────────────────────────────────────────────────

def save_run_stats(frame_log: list, pts_log: list, saves_log: list,
                   diffs_log: list, out_dir: Path) -> None:
    import json as _json
    import numpy as _np

    total = len(frame_log)
    bad   = sum(1 for r in frame_log if r[3] == 0)
    ok    = total - bad

    mono_ok  = [r[0] for r in frame_log if r[3] == 1]
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
                "mean_ms":   round(float(_np.mean(ivs)),           1),
                "median_ms": round(float(_np.median(ivs)),         1),
                "p95_ms":    round(float(_np.percentile(ivs, 95)), 1),
                "p99_ms":    round(float(_np.percentile(ivs, 99)), 1),
                "max_ms":    round(float(_np.max(ivs)),            1),
            }

    reconnects = 0
    prev_pts: dict = {}
    for r in pts_log:
        url, pts = r[2], r[3]
        if pts > 0:
            if url in prev_pts and pts < prev_pts[url] - 500:
                reconnects += 1
            prev_pts[url] = pts

    save_counts: dict = {}
    for r in saves_log:
        save_counts[r[3]] = save_counts.get(r[3], 0) + 1

    diff_stats: dict = {}
    if diffs_log:
        dv = _np.array([r[3] for r in diffs_log if r[3] > 0])
        if len(dv):
            diff_stats = {
                "mean":          round(float(_np.mean(dv)),           3),
                "p95":           round(float(_np.percentile(dv, 95)), 3),
                "max":           round(float(_np.max(dv)),            3),
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
    logger.info("run_stats.json → %s", path)


# ─── CPU chart helper ─────────────────────────────────────────────────────────

def draw_cpu_on_ax(ax, cpu_log: list, *,
                   title: str = "Загрузка ЦПУ",
                   show_mean: bool = False) -> None:
    """Рисует ЦПУ% + утилизацию + частоту на готовый axes.

    Не создаёт фигуру — вызывается как из save_charts (subplot),
    так и из _save_cpu_chart в 5_2/6_2 (отдельный файл).
    """
    t = [r[0] for r in cpu_log]
    v = [float(r[2]) for r in cpu_log]
    ax.plot(t, v, color="#2255cc", linewidth=1.0, linestyle="--", alpha=0.65,
            label="% Proc. Time")
    utility_v = [float(r[5]) for r in cpu_log if len(r) > 5]
    if utility_v and any(u > 0 for u in utility_v):
        t_util = [r[0] for r in cpu_log if len(r) > 5]
        ax.plot(t_util, utility_v, color="#22bb44", linewidth=1.3,
                label="% Proc. Utility (Диспетчер)")
        ax.fill_between(t_util, utility_v, alpha=0.18, color="#22bb44")
    ax.legend(loc="upper right", fontsize=7)
    if show_mean and v:
        mean_cpu = sum(v) / len(v)
        ax.axhline(mean_cpu, color="orange", linestyle="--", linewidth=0.8)
        ax.text(t[0] if t else 0, mean_cpu + 1,
                f"среднее {mean_cpu:.1f}%", fontsize=8, color="orange")
    ax.set_ylabel("ЦПУ, %")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    freq_pdh  = [float(r[3]) for r in cpu_log if len(r) > 3]
    freq_step = [float(r[4]) for r in cpu_log if len(r) > 4]
    if (freq_pdh and any(f > 0 for f in freq_pdh)) or \
       (freq_step and any(f > 0 for f in freq_step)):
        t_freq   = [r[0] for r in cpu_log if len(r) > 3]
        ax2f     = ax.twinx()
        all_vals = [f for f in freq_pdh + freq_step if f > 0]
        if freq_pdh and any(f > 0 for f in freq_pdh):
            ax2f.plot(t_freq, freq_pdh, color="#ffbb55", linewidth=1.0,
                      linestyle="--", alpha=0.5,
                      marker=".", markersize=9,
                      markerfacecolor="#ffee11", markeredgewidth=0,
                      label="МГц (PDH)")
        if freq_step and any(f > 0 for f in freq_step):
            t_step = [r[0] for r in cpu_log if len(r) > 4]
            ax2f.step(t_step, freq_step, color="#aaaaaa", linewidth=0.8,
                      alpha=0.6, where="post",
                      marker=".", markersize=8,
                      markerfacecolor="#dddddd", markeredgewidth=0,
                      label="МГц (P-state)")
        ax2f.set_ylabel("частота, МГц", color="#ff8800", fontsize=8)
        ax2f.tick_params(axis="y", labelcolor="#ff8800", labelsize=7)
        ax2f.legend(loc="lower right", fontsize=7)
        if all_vals:
            f_min = min(all_vals)
            f_max = max(all_vals)
            pad   = max((f_max - f_min) * 0.15, 50)
            ax2f.set_ylim(max(0, f_min - pad), f_max + pad)


# ─── cpu.csv ──────────────────────────────────────────────────────────────────

def save_cpu_csv(cpu_log: list, out_dir: Path, append: bool = False) -> None:
    """Сохраняет cpu.csv (6 колонок). append=True — ДОПИСЫВАТЬ (для per-poll сервисов,
    которые реинвокаются каждый поллинг: накопление за день по ts_msk); заголовок — раз."""
    import csv as _csv
    if not cpu_log:
        return
    path = out_dir / "cpu.csv"
    do_append = append and path.exists()
    with open(path, "a" if do_append else "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        if not do_append:
            w.writerow(["mono_s", "ts_msk", "cpu_pct", "freq_mhz_pdh", "freq_mhz_step", "cpu_utility_pct"])
        w.writerows(cpu_log)
    logger.info("cpu.csv:    %d замеров (%s)", len(cpu_log), "append" if do_append else "write")


def compute_per_frame_log(ms_values) -> str:
    """Строка ' мсек/кадр: min/avg/max' по ЧИСТОМУ времени вычисления кадра (только счёт,
    без сна адаптивного лимитера и без батч-оверхеда). Показывает, справляется ли ЦПУ.
    Единица (мс) — в самой метке 'мсек/кадр'; парно к 'сек/кадр' (I/O). Пустой список → ''."""
    if not ms_values:
        return ""
    mn = min(ms_values)
    mx = max(ms_values)
    av = sum(ms_values) / len(ms_values)
    return f"  мсек/кадр: {mn:.0f}/{av:.0f}/{mx:.0f}"


# ─── pts_chart.png ────────────────────────────────────────────────────────────

def save_pts_chart(pts_log: list, cpu_log: list, out_dir: Path) -> None:
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

    # Ось X = ВРЕМЯ СУТОК: raw mono → HH:MM:SS через опорный первый кадр. У per-day файла за
    # сутки процесс мог стартовать накануне (mono 1-го кадра велик) → метки «от старта» путают.
    def _sod(ts: str) -> float:
        try:
            _p = ts.split("_"); _t = _p[1]
            _mc = int(_p[2]) if len(_p) > 2 and _p[2].isdigit() else 0
            return int(_t[:2]) * 3600 + int(_t[2:4]) * 60 + int(_t[4:6]) + _mc / 1e6
        except Exception:
            return 0.0
    _g_mono0 = pts_log[0][0]
    _g_sod0  = _sod(pts_log[0][1])
    def _x_tod(x, _pos):
        _tot = int(_g_sod0 + (x - _g_mono0)) % 86400
        return f"{_tot // 3600:02d}:{(_tot % 3600) // 60:02d}:{_tot % 60:02d}"
    _x_fmt = _ticker.FuncFormatter(_x_tod)

    by_url: dict[str, list] = defaultdict(list)
    for row in pts_log:
        by_url[row[2]].append(row)

    n = len(by_url)
    n_cpu  = 1 if cpu_log else 0
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

        mono_s   = [(m - mono0)        for m in mono]
        pts_norm = [(p - pts0) / 1000  for p in pts]
        wall_s   = [(w - wall0)        for w in wall]

        # PTS сабпотока «рваный»: прыгает и НАЗАД (реконнект), и ВПЕРЁД на десятки
        # секунд. Сшиваем ЛЮБОЙ разрыв |Δ|>30 с по реальному времени (mono) —
        # иначе forward-скачки копятся в фейковый «дрейф» на сотни тысяч секунд.
        pts_clean = list(pts_norm)
        offset = 0.0
        n_breaks = 0
        for i in range(1, len(pts_norm)):
            if abs(pts_norm[i] - pts_norm[i - 1]) > 30.0:     # разрыв потока
                n_breaks += 1
                step = mono_s[i] - mono_s[i - 1]              # ожидаемый шаг по mono
                offset += (pts_norm[i - 1] + step) - pts_norm[i]
            pts_clean[i] = pts_norm[i] + offset

        # прорядить для отрисовки: за полный день точек сотни тысяч → панель заливается сплошняком
        _st = max(1, len(mono) // 12000)
        _mn = mono[::_st]

        ax = axes[ax_idx][0]
        ax.scatter(_mn, mono_s[::_st],    color="#2255cc", s=1,  marker="o", alpha=0.3, zorder=3, label="mono, с")
        ax.scatter(_mn, wall_s[::_st],    color="#228833", s=2,  marker="s", alpha=0.3, zorder=3, label="wall clock, с")
        ax.scatter(_mn, pts_clean[::_st], color="#cc5500", s=2,  marker="^", alpha=0.3, zorder=3, label="FFmpeg PTS (норм.), с")
        ax.set_ylabel("с от старта")
        ax.set_title(f"{url_id} — три метки времени (все в с от первого кадра)")
        ax.legend(loc="upper left", fontsize=8)
        ax.yaxis.set_major_locator(_ticker.MaxNLocator(8))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax_idx += 1

        ax = axes[ax_idx][0]
        drift_pts = [p - m for p, m in zip(pts_clean, mono_s)]
        ax.scatter(_mn, drift_pts[::_st], color="#cc5500", s=2, marker="^", alpha=0.3,
                   label="PTS − mono (с): пила ≈ глубина буфера (кадры пачками)")
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_ylabel("с")
        ax.set_title(f"{url_id} — дрейф PTS − mono (с); разрывов PTS сшито: {n_breaks}")
        ax.legend(loc="upper left", fontsize=8)
        ax.yaxis.set_major_locator(_ticker.MaxNLocator(8))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax_idx += 1

        ax = axes[ax_idx][0]
        drift_wall = [w - m for w, m in zip(wall_s, mono_s)]
        ax.scatter(_mn, drift_wall[::_st], color="#228833", s=6, marker="s", alpha=0.35,
                   label="wall − mono (с): расхождение стенных и mono часов")
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_ylabel("с")
        ax.set_xlabel("время суток")
        ax.set_title(f"{url_id} — дрейф wall − mono (с)")
        ax.legend(loc="upper left", fontsize=8)
        ax.yaxis.set_major_locator(_ticker.MaxNLocator(8))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax_idx += 1

    if cpu_log:
        ax = axes[ax_idx][0]
        draw_cpu_on_ax(ax, cpu_log, title="CPU usage, %")
        ax.set_xlabel("время суток")

    for _row in axes:                        # ось X всех панелей → время суток (HH:MM:SS)
        _row[0].xaxis.set_major_formatter(_x_fmt)

    plt.tight_layout(rect=(0, 0, 1, 0.97))   # резерв под suptitle (не налезает)
    path = out_dir / "pts_chart.png"
    plt.savefig(str(path), dpi=120)
    plt.close()
    logger.info("pts_chart.png → %s", path)


# ─── charts.png ───────────────────────────────────────────────────────────────

def save_charts(frame_log: list, cpu_log: list, saves_log: list, diffs_log: list,
                threshold: float, out_dir: Path, *,
                title: str = "camera run",
                event_type: str = "yolo",
                save_colors: "dict | None" = None,
                save_levels: "dict | None" = None,
                thresholds: "dict | None" = None) -> None:
    """Строит и сохраняет совмещённый PNG: интервалы кадров + дифы + сохранения + ЦПУ.

    thresholds — по-зонный порог diff {cam_stem: value}. Если задан и у зон РАЗНЫЕ пороги —
    на diff-панелях рисуется линия на каждую зону в её цвете; иначе один общий `threshold`."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as _ticker
        import numpy as _np
    except ImportError:
        return
    if not frame_log and not diffs_log and not saves_log:
        return

    _CAM_COLORS = ["#e05c00", "#0066cc", "#228833", "#aa22aa"]
    _MARKERS    = ["*", (5, 1, 0), (6, 2, 0), (4, 1, 0)]
    _sc = save_colors if save_colors is not None else {
        "baseline": "#888888", "heartbeat": "#8844bb",
        "diff": "#228833", "raw": "#cc6600", "yolo": "#cc3333",
    }
    _sl = save_levels if save_levels is not None else {
        "baseline": 1, "heartbeat": 2, "diff": 3, "raw": 4, "yolo": 5,
    }

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
        figsize=(14, sum(height_ratios) * 1.05),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
        constrained_layout=True,   # авто-подгонка зазоров (меньше пустот, чем голый tight_layout)
    )
    fig.suptitle(f"{title} — кадры и загрузка ЦПУ", fontsize=11)

    _all_monos = ([r[0] for r in frame_log] + [r[0] for r in diffs_log]
                  + [r[0] for r in saves_log] + [r[0] for r in cpu_log])
    t_max = max(_all_monos, default=1)
    # mono ПЕРВОГО кадра файла: у per-day файла за сутки процесс мог стартовать накануне
    # (mono0 велик) → без сдвига данные жмутся вправо, а метки времени суток врут. Ось = mono−mono0.
    _mono0 = min(_all_monos, default=0.0)

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

    def _ts_hmsm(ts_str: str) -> str:
        r = _ts_parts(ts_str)
        return f"{r[0]:02d}:{r[1]:02d}:{r[2]:02d}.{r[3]//1000:03d}" if r else ts_str

    _t0_first = next((r[1] for r in frame_log + diffs_log + saves_log if r[1]), "")
    _t0_ts    = _t0_first
    _t0_parts = _ts_parts(_t0_ts)
    _t0_abs   = (_t0_parts[0]*3600 + _t0_parts[1]*60 + _t0_parts[2]) if _t0_parts else 0
    _x_left   = _mono0 - (int(_t0_abs) % 60)   # левый край = первый кадр, выровнен на минуту

    _tick_range = t_max * 1.02 - _x_left
    # шаг тика: держим <= ~16 меток. Крупные шаги (час/2ч/3ч/4ч) нужны для ПОЛНОГО дня —
    # иначе за сутки шаг упирался в 30 мин → до 48 меток → подписи налезали.
    _tick_step  = next((s for s in [60, 120, 180, 300, 600, 900, 1800, 3600, 7200, 10800, 14400]
                        if _tick_range / s <= 16), 14400)
    _tick_positions = list(range(int(_x_left), int(t_max * 1.02) + _tick_step, _tick_step))

    # Метка даты появляется только на первом тике нового дня
    from datetime import date as _date, timedelta as _tdelta
    _t0_date = None
    try:
        _d = _t0_ts.split("_")[0]  # "20260630"
        _t0_date = _date(int(_d[:4]), int(_d[4:6]), int(_d[6:8]))
    except Exception:
        pass

    _tick_labels: list[str] = []
    _last_day_label = -1
    for _tp in _tick_positions:
        _total = int(_t0_abs + _tp - _mono0)   # tp относительно mono0 → реальное время суток
        _day   = _total // 86400          # 0 = день старта, 1 = следующий, …
        _hh    = (_total % 86400) // 3600
        _mm    = (_total % 3600) // 60
        _lbl   = f"{_hh:02d}:{_mm:02d}"   # ЧЧ:ММ (без секунд — короче, не налезают)
        if _day > _last_day_label and _day > 0 and _t0_date is not None:
            _lbl = f"{(_t0_date + _tdelta(days=_day)).strftime('%d.%m')}\n{_lbl}"
            _last_day_label = _day
        _tick_labels.append(_lbl)

    _x_fmt = _ticker.FixedFormatter(_tick_labels)
    _x_loc = _ticker.FixedLocator(_tick_positions)

    for ax_idx, (uid, rows) in enumerate(by_url.items()):
        ax    = axes[ax_idx][0]
        monos = [r[0] for r in rows]
        ok_flags    = [r[3] for r in rows]
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

        n_events = 0
        ev_list = []
        for row in rows:
            ev = row[5] if len(row) > 5 else ""
            if ev == event_type:
                lw = 0.9 if event_type == "yolo" else 0.7
                ax.axvline(row[0], color="red", linewidth=lw, alpha=0.7)
                n_events += 1
                ev_list.append(row)

        n_ok = sum(ok_flags)
        n_pl = sum(1 for p in plaus_flags if p == 1)
        ax.set_ylabel("интервал, мс")
        ax.set_title(f"{uid}  |  кадров: {len(rows)}  ok: {n_ok}  plausible: {n_pl}")
        ax.set_xlim(_x_left, t_max * 1.02)
        y_lim_top = max(intervals_ms) * 1.2 if intervals_ms else 1
        ax.set_ylim(0, y_lim_top)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis.set_major_formatter(_x_fmt)
        ax.xaxis.set_major_locator(_x_loc)

        if event_type == "yolo":
            for i, row in enumerate(ev_list):
                ts_str = row[1] if len(row) > 1 else ""
                label  = f"YOLO\n{_ts_hmsm(ts_str)}\n+{row[0]:.1f}s"
                y_frac = 0.75 if i % 2 == 0 else 0.45
                ax.annotate(
                    label, xy=(row[0], y_lim_top * y_frac),
                    fontsize=7, ha="left", va="center",
                    xytext=(4, 0), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0",
                              edgecolor="red", alpha=0.55),
                )

        import matplotlib.lines as _mlines
        legend_handles = [
            _mlines.Line2D([], [], color="#44aa44", marker="*", linestyle="none",
                           markersize=9,
                           label="интервал между кадрами, мс (рост = столл/реконнект)"),
            _mlines.Line2D([], [], color="blue", linestyle="--", linewidth=0.8,
                           label=f"среднее {mean_iv:.0f} мс  ({1000/mean_iv:.1f} fps)"),
        ]
        if n_events > 0:
            ev_label = (f"YOLO-детекция ({n_events})" if event_type == "yolo"
                        else f"{event_type}-событие ({n_events})")
            legend_handles.append(_mlines.Line2D([], [], color="red", linewidth=0.7,
                                                  label=ev_label))
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    if diffs_log:
        by_cam: dict[str, list] = defaultdict(list)
        for row in diffs_log:
            by_cam[row[2]].append(row)

        all_diffs = [r[3] for r in diffs_log if r[3] > 0]
        p95 = float(_np.percentile(all_diffs, 95)) if all_diffs else threshold * 10

        def _draw_diffs_ax(ax, ylim=None):
            cams_sorted = sorted(by_cam.items())
            cam_color: dict = {}
            for ci, (cam, rows) in enumerate(cams_sorted):
                dt    = [r[0] for r in rows]
                dv    = [r[3] for r in rows]
                color = _CAM_COLORS[ci % len(_CAM_COLORS)]
                cam_color[cam] = color
                ax.scatter(dt, dv, color=color, s=16, alpha=0.45, label=cam,
                           marker=_MARKERS[ci % len(_MARKERS)])
            # Пороги diff: ПО ЗОНАМ (если заданы и различаются) — линия на каждую зону в ЕЁ цвете,
            # чтобы видеть какой порог к какой зоне; иначе одна общая линия (подпись у левого края ОСИ:
            # x=0 у per-day файла улетал за левое поле, т.к. _x_left велик).
            _zt = {cam: float(thresholds.get(cam, threshold)) for cam, _ in cams_sorted} if thresholds else {}
            if _zt and len(set(_zt.values())) > 1:
                for cam, thr in _zt.items():
                    _c = cam_color.get(cam, "red")
                    _zn = cam.replace("_URL", "").split("_")[-1]   # зона из CAM_..._D_URL → 'D'
                    ax.axhline(thr, color=_c, linestyle="--", linewidth=1.2, alpha=0.9)
                    ax.text(_x_left, thr * 1.03, f"порог {_zn}={thr:g}", fontsize=8, color=_c)
            else:
                ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0,
                           label=f"порог diff={threshold}")
                ax.text(_x_left, threshold * 1.03, f"порог {threshold}", fontsize=8, color="red")
            ax.set_ylabel("diff")
            ax.set_xlim(_x_left, t_max * 1.02)
            if ylim is not None:
                ax.set_ylim(0, ylim)
            ax.yaxis.set_major_locator(_ticker.MaxNLocator(10))
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=8, ncol=2)

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
        cam_mk   = {cam: _MARKERS[i % len(_MARKERS)] for i, cam in enumerate(cam_list)}

        def _cam_short(cam: str) -> str:
            parts = [p for p in cam.split("_") if p and p != "URL"]
            return parts[-1] if parts else cam

        y_pos:        dict[tuple, float] = {}
        ytick_pos:    list[float] = []
        ytick_labels: list[str]  = []
        y = 1.0
        for stype in sorted(by_type_cam, key=lambda s: _sl.get(s, 99)):
            for cam in sorted(by_type_cam[stype]):
                y_pos[(stype, cam)] = y
                ytick_pos.append(y)
                ytick_labels.append(f"{stype} / {_cam_short(cam)}")
                y += 1.0
            y += 0.5

        for (stype, cam), yv in y_pos.items():
            times = by_type_cam[stype][cam]
            color = _sc.get(stype, "#aaaaaa")
            ax.scatter(times, [yv] * len(times), color=color, s=35,
                       marker=cam_mk[cam], alpha=0.85, zorder=3)

        for stype in sorted(_sl, key=_sl.get):
            if stype in by_type_cam:
                ax.scatter([], [], color=_sc.get(stype, "#aaaaaa"),
                           s=35, marker="o", label=stype)

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels, fontsize=7)
        ax.set_xlim(_x_left, t_max * 1.02)
        ax.set_ylim(0, y)
        ax.set_title(f"Сохранения кадров  ({len(saves_log)} событий)")
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        ax.grid(True, linestyle="--", alpha=0.35)

    if cpu_log:
        ax = axes[n_frame_rows + n_diffs_rows + n_saves_rows][0]
        draw_cpu_on_ax(ax, cpu_log, title="Загрузка ЦПУ (все ядра, %)", show_mean=True)
        ax.set_xlim(_x_left, t_max * 1.02)

    for _ax in [axes[i][0] for i in range(len(axes))]:
        _ax.xaxis.set_major_formatter(_x_fmt)
        _ax.xaxis.set_major_locator(_x_loc)
        _ax.tick_params(axis="x", labelrotation=30)   # наклон — подписи времени не налезают
    axes[-1][0].set_xlabel("время МСК")

    # компоновка — через constrained_layout (см. plt.subplots выше); tight_layout не нужен
    chart_path = out_dir / "charts.png"
    plt.savefig(str(chart_path), dpi=120)
    plt.close()
    logger.info("charts.png → %s", chart_path)


# ─── OSD helpers ──────────────────────────────────────────────────────────────

def parse_img_filename(stem: str) -> "tuple[str, str, str] | None":
    """Парсит имя файла → (cam_name, ts_str 'YYYY-MM-DD HH:MM:SS', img_type).

    Формат: cam_01_9_u_YYYYMMDD_HHMMSS_ffffff_msk_type
    """
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            cam_name = "_".join(parts[:i])
            date_p   = p
            time_p   = parts[i + 1] if i + 1 < len(parts) else "000000"
            img_type = parts[-1]
            ts_str   = (f"{date_p[:4]}-{date_p[4:6]}-{date_p[6:]} "
                        f"{time_p[:2]}:{time_p[2:4]}:{time_p[4:6]}")
            return cam_name, ts_str, img_type
    return None


def regen_osd_from_images(run_dir: Path) -> list[list]:
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

    imgs   = sorted(images_dir.rglob("*.jpg"))
    u_imgs = [p for p in imgs if "_u_" in p.name or "_9_u_" in p.name]
    if not u_imgs:
        return []

    if not low_templates_complete():
        logger.info("[OSD] Построение LOW-шаблонов из %d изображений…", len(u_imgs))
        added_total: set[str] = set()
        for img_path in u_imgs:
            info = parse_img_filename(img_path.stem)
            if info is None:
                continue
            _, ts_str, _ = info
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            new = build_low_templates_from_image(frame, ts_str)
            added_total.update(new.keys())
            if low_templates_complete():
                break
        logger.info("[OSD] Шаблоны: %s (%d/12)", sorted(_LOW_TEMPLATES.keys()), len(_LOW_TEMPLATES))

    from datetime import datetime as _dt
    osd_log: list[list] = []
    ok_count = 0

    for img_path in u_imgs:
        info = parse_img_filename(img_path.stem)
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
                wall_dt   = _dt.strptime(wall_ts_str, "%Y-%m-%d %H:%M:%S")
                drift_sec = round((osd_dt - wall_dt).total_seconds(), 1)
            except Exception:
                drift_sec = None
            ok_count += 1

        osd_log.append([wall_ts_str, cam_name, img_type, osd_ts_str,
                        "" if drift_sec is None else drift_sec])

    logger.info("[OSD] Извлечено: %d/%d меток", ok_count, len(u_imgs))

    if osd_log:
        csv_path = run_dir / "osd_times.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wall_ts", "cam", "img_type", "osd_ts", "drift_sec"])
            writer.writerows(osd_log)
        logger.info("osd_times.csv → %s", csv_path)

    return osd_log


def save_osd_chart(osd_log: list, out_dir: Path) -> None:
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
        logger.info("[OSD] Нет распознанных меток — график пропущен")
        return

    def _parse(s: str) -> float:
        try:
            return _dt.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            return 0.0

    wall_ts  = [_parse(r[0]) for r in rows_with_osd]
    osd_ts   = [_parse(r[3]) for r in rows_with_osd]
    img_type = [r[2] for r in rows_with_osd]
    drift_vals = [(o - w) for o, w in zip(osd_ts, wall_ts)]

    t0       = wall_ts[0]
    wall_rel = [(t - t0) for t in wall_ts]
    osd_rel  = [(t - t0) for t in osd_ts]

    _TYPE_COLOR = {"baseline": "#888888", "heartbeat": "#8844bb",
                   "yolo": "#cc3333", "diff": "#228833"}

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
    logger.info("osd_chart.png → %s", chart_path)
