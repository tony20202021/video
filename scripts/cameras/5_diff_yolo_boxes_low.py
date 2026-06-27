"""
Детекция людей на потоках RTSP: motion detection → YOLOv8n ONNX → сохранение с bounding box.

Только LOW-поток (CAM_*_URL, субпоток 640×720): frame diff для детекции движения,
YOLO запускается непосредственно на LOW-кадре.

Побочные потоки:
  _CpuMonitor  — замер загрузки ЦПУ раз в N сек (требует psutil)

Выход (run_<ts>/):
  images/          — baseline, yolo-кадры с bbox, heartbeat
  images/crops/    — вырезки отдельных людей
  images/raw/      — сырые кадры при --save-raw
  frames.csv       — метка времени каждого кадра LOW-потока
  saves.csv        — моменты сохранений с типом (baseline/yolo/heartbeat/raw)
  cpu.csv          — загрузка ЦПУ с периодичностью --cpu-interval
  charts.png       — совмещённый график: интервалы кадров + сохранения + ЦПУ
  run_params.json

Модель: models/yolov8n.onnx (скачать: https://github.com/ultralytics/assets/releases)
  wget -P models/ https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx

Usage:
    python scripts/cameras/5_diff_yolo_boxes_low.py
    python scripts/cameras/5_diff_yolo_boxes_low.py --model models/yolov8n.onnx --conf 0.4
    python scripts/cameras/5_diff_yolo_boxes_low.py --tcp --threshold 12
    python scripts/cameras/5_diff_yolo_boxes_low.py --duration 3600
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
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.cam_crop import apply_crop_optional, crop_map_for_cameras, resolve_global_crop
from common.utils.cam_urls import collect_cam_urls
from common.utils.motion_utils import (
    ffmpeg_capture_options,
    frame_decode_plausible,
    mean_abs_diff,
    open_cap,
    prepare_gray,
    read_first_plausible_frame,
    redact_url,
    skip_url,
    stem_from_var,
)
from common.utils.time_msk import ts_cam_for_file, ts_for_dir, ts_for_file


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


class _CapReader:
    """Фоновый поток: непрерывно читает VideoCapture, держит только последний кадр.

    Queue(maxsize=1) гарантирует: main-поток всегда получает самый свежий кадр
    без буферного лага, даже если обработка (YOLO, imwrite) занимает > 1 кадра.
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

            # Выбросить устаревший кадр и положить свежий
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
        self._thread.join(timeout=3.0)


DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "cameras" / "5_diff_yolo_boxes_low"
DEFAULT_MODEL = REPO_ROOT / ".models" / "yolov8n.onnx"

YOLO_INPUT_SIZE = 640
PERSON_CLASS = 0
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─── YOLOv8n inference ────────────────────────────────────────────────────────

def _load_model(model_path: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("Нужен onnxruntime: pip install onnxruntime", file=sys.stderr)
        return None
    if not model_path.is_file():
        print(
            f"Модель не найдена: {model_path}\n"
            f"  Скачать: wget -P models/ "
            f"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
            file=sys.stderr,
        )
        return None
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return sess


def _preprocess(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Возвращает (blob, scale, pad_x, pad_y) для letterbox-ресайза в 640×640."""
    h, w = bgr.shape[:2]
    scale = min(YOLO_INPUT_SIZE / w, YOLO_INPUT_SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (YOLO_INPUT_SIZE - nw) // 2
    pad_y = (YOLO_INPUT_SIZE - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalize
    blob = blob.transpose(2, 0, 1)[np.newaxis]            # HWC→NCHW
    return blob, scale, pad_x, pad_y


def _postprocess(
    output: np.ndarray,
    *,
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    conf_threshold: float,
    nms_threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    """Возвращает список (x1, y1, x2, y2, confidence) для людей в координатах orig."""
    preds = output[0].T  # [8400, 84]
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    person_scores = class_scores[:, PERSON_CLASS]

    mask = person_scores >= conf_threshold
    if not mask.any():
        return []

    scores = person_scores[mask]
    bxywh = boxes_xywh[mask]

    cx, cy, bw, bh = bxywh[:, 0], bxywh[:, 1], bxywh[:, 2], bxywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2

    boxes_for_nms = np.stack([x1, y1, bw, bh], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), conf_threshold, nms_threshold)

    result = []
    for i in (indices.flatten() if len(indices) else []):
        rx1 = int((float(x1[i]) - pad_x) / scale)
        ry1 = int((float(y1[i]) - pad_y) / scale)
        rx2 = int((float(x1[i]) + float(bxywh[i, 2]) - pad_x) / scale)
        ry2 = int((float(y1[i]) + float(bxywh[i, 3]) - pad_y) / scale)
        rx1 = max(0, min(rx1, orig_w - 1))
        ry1 = max(0, min(ry1, orig_h - 1))
        rx2 = max(0, min(rx2, orig_w))
        ry2 = max(0, min(ry2, orig_h))
        result.append((rx1, ry1, rx2, ry2, float(scores[i])))
    return result


def detect_people(
    sess,
    bgr: np.ndarray,
    *,
    conf_threshold: float,
    nms_threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    h, w = bgr.shape[:2]
    blob, scale, pad_x, pad_y = _preprocess(bgr)
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: blob})[0]
    return _postprocess(
        output,
        orig_w=w,
        orig_h=h,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )


def draw_boxes(bgr: np.ndarray, detections: list[tuple[int, int, int, int, float]]) -> np.ndarray:
    out = bgr.copy()
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"{conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - baseline - 2), (x1 + tw, y1), BOX_COLOR, -1)
        cv2.putText(out, label, (x1, y1 - baseline - 1), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ─── Фоновый поток ЦПУ ────────────────────────────────────────────────────────

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

    # Интервалы между кадрами по url_id
    by_url: dict = {}
    for r in frame_log:
        if r[3] == 1:
            by_url.setdefault(r[2], []).append(r[0])

    interval_stats: dict = {}
    for url, times in by_url.items():
        times.sort()
        ivs = _np.diff(times) * 1000  # мс
        if len(ivs):
            interval_stats[url] = {
                "mean_ms":   round(float(_np.mean(ivs)),          1),
                "median_ms": round(float(_np.median(ivs)),        1),
                "p95_ms":    round(float(_np.percentile(ivs, 95)),1),
                "p99_ms":    round(float(_np.percentile(ivs, 99)),1),
                "max_ms":    round(float(_np.max(ivs)),           1),
            }

    # Реконнекты — обратный прыжок PTS > 500 мс
    reconnects = 0
    prev_pts: dict = {}
    for r in pts_log:
        url, pts = r[2], r[3]
        if pts > 0:
            if url in prev_pts and pts < prev_pts[url] - 500:
                reconnects += 1
            prev_pts[url] = pts

    # Сохранения по типу
    save_counts: dict = {}
    for r in saves_log:
        save_counts[r[3]] = save_counts.get(r[3], 0) + 1

    # Диффы
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

    # Парсим ts_msk → секунды с начала эпохи
    from datetime import datetime as _dt
    def _parse_ts(s: str) -> float:
        # формат: YYYYMMDD_HHMMSS_ffffff_msk
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

        mono = [r[0] for r in rows]          # сек от старта
        pts  = [r[3] for r in rows]          # FFmpeg PTS, мс
        wall = [_parse_ts(r[1]) for r in rows]

        mono0 = mono[0]
        pts0  = pts[0]
        wall0 = wall[0]

        mono_s   = [(m - mono0)         for m in mono]       # с от старта
        pts_norm = [(p - pts0) / 1000   for p in pts]        # с от первого PTS
        wall_s   = [(w - wall0)         for w in wall]        # с от первой wall

        # Убираем сброс PTS при реконнекте (резкое уменьшение)
        pts_clean = list(pts_norm)
        offset = 0.0
        for i in range(1, len(pts_clean)):
            if pts_clean[i] + offset < pts_clean[i-1] + offset - 0.5:
                offset += pts_clean[i-1] - pts_clean[i] + 0.1
            pts_clean[i] += offset

        # --- Subplot 1: все три метки на одной оси (с от старта) ---
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

        # --- Subplot 2: дрейф PTS относительно mono ---
        ax = axes[ax_idx][0]
        drift_pts = [p - m for p, m in zip(pts_clean, mono_s)]
        ax.scatter(mono, drift_pts, color="#cc5500", s=2, marker="^", alpha=0.2)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_ylabel("с")
        ax.set_title(f"{url_id} — дрейф PTS − mono (с)")
        ax.yaxis.set_major_locator(_ticker.MaxNLocator(8))
        ax.grid(True, linestyle="--", alpha=0.35)
        ax_idx += 1

        # --- Subplot 3: дрейф wall относительно mono ---
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

    # --- CPU subplot (последний) ---
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
    _SAVE_COLORS = {"baseline": "#888888", "yolo": "#cc3333",
                    "heartbeat": "#8844bb", "raw": "#dd8800", "diff": "#228833"}
    _LINE_STYLES = ["-", "--", "-.", ":"]
    _SAVE_LEVELS = {"baseline": 1, "diff": 2, "yolo": 3, "heartbeat": 4, "raw": 5}
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
    fig.suptitle("5_diff_yolo_boxes_low — кадры и загрузка ЦПУ", fontsize=11)

    t_max = max((row[0] for row in frame_log), default=1)

    # Парсим стартовое время из первого кадра для форматирования оси X
    def _ts_parts(ts_str: str):
        """'20260627_113512_395077_msk' → (hh, mm, ss, us)"""
        try:
            p = ts_str.split("_")
            t = p[1]
            return int(t[0:2]), int(t[2:4]), int(t[4:6]), int(p[2])
        except Exception:
            return None

    def _ts_hms(ts_str: str) -> str:
        """'20260627_113512_395077_msk' → '11:35:12'"""
        r = _ts_parts(ts_str)
        return f"{r[0]:02d}:{r[1]:02d}:{r[2]:02d}" if r else ts_str

    def _ts_hmsm(ts_str: str) -> str:
        """'20260627_113512_395077_msk' → '11:35:12.395'"""
        r = _ts_parts(ts_str)
        return f"{r[0]:02d}:{r[1]:02d}:{r[2]:02d}.{r[3]//1000:03d}" if r else ts_str

    # Секунд от полуночи для первого кадра → используем как t0 для форматтера
    _t0_ts = frame_log[0][1] if frame_log else ""
    _t0_parts = _ts_parts(_t0_ts)
    _t0_abs = (_t0_parts[0]*3600 + _t0_parts[1]*60 + _t0_parts[2]) if _t0_parts else 0
    _x_left = -(_t0_abs % 60)  # отступ до ближайшей круглой минуты назад

    # Тики строго по круглым минутам; шаг выбирается чтобы тиков было 10–20
    _tick_range = t_max * 1.02 - _x_left
    _tick_step = next((s for s in [60, 120, 180, 300, 600, 900, 1800]
                       if _tick_range / s <= 20), 1800)
    _tick_positions = list(range(int(_x_left), int(t_max * 1.02) + _tick_step, _tick_step))

    def _x_time_fmt(x, _pos):
        total = int(_t0_abs + x)
        hh, mm, ss = total // 3600, (total % 3600) // 60, total % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    # --- Frame interval subplots ---
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

        n_yolo = 0
        yolo_events = []
        for row in rows:
            ev = row[5] if len(row) > 5 else ""
            if ev == "yolo":
                ax.axvline(row[0], color="red", linewidth=0.9, alpha=0.7)
                n_yolo += 1
                yolo_events.append(row)

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

        # Аннотации YOLO-событий
        for i, row in enumerate(yolo_events):
            ts_str = row[1] if len(row) > 1 else ""
            label = f"YOLO\n{_ts_hmsm(ts_str)}\n+{row[0]:.1f}s"
            # чередуем высоту чтобы метки не перекрывались
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
            _mlines.Line2D([], [], color="blue", linestyle="--", linewidth=0.8,
                           label=f"среднее {mean_iv:.0f} мс"),
        ]
        if n_yolo > 0:
            legend_handles.append(_mlines.Line2D([], [], color="red", linewidth=0.7,
                                                  label=f"YOLO-детекция ({n_yolo})"))
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # --- Diff subplots ---
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
                ls = _LINE_STYLES[ci % len(_LINE_STYLES)]
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

        ax = axes[n_frame_rows][0]
        _draw_diffs_ax(ax)
        ax.set_title(f"Frame diff — полный масштаб  ({len(diffs_log)} записей)")

        ax = axes[n_frame_rows + 1][0]
        _draw_diffs_ax(ax, ylim=p95 * 2.3)
        ax.set_title(f"Frame diff — до p95={p95:.2f}  (детальный вид)")

    # --- Saves scatter subplot ---
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

        # Каждая пара (тип, камера) → своя строка по Y
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
            y += 0.5  # зазор между группами типов

        for (stype, cam), yv in y_pos.items():
            times = by_type_cam[stype][cam]
            color = _SAVE_COLORS.get(stype, "#aaaaaa")
            ax.scatter(times, [yv] * len(times), color=color, s=35,
                       marker=cam_mk[cam], alpha=0.85, zorder=3)

        # Легенда: тип → цвет
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

    # --- CPU subplot ---
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

    for _ax in [axes[i][0] for i in range(len(axes))]:
        _ax.xaxis.set_major_formatter(_ticker.FuncFormatter(_x_time_fmt))
        _ax.xaxis.set_major_locator(_ticker.FixedLocator(_tick_positions))
    axes[-1][0].set_xlabel("время МСК")

    plt.tight_layout()
    chart_path = out_dir / "charts.png"
    plt.savefig(str(chart_path), dpi=120)
    plt.close()
    print(f"  charts.png → {chart_path}")


def _regen_charts(run_dir: Path) -> None:
    """Перечитывает CSV из существующего run-каталога и перегенерирует charts.png и pts_chart.png."""
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


def _parse_img_filename(stem: str) -> tuple[str, str, str] | None:
    """Парсит имя файла → (cam_name, ts_str 'YYYY-MM-DD HH:MM:SS', img_type).

    Формат: cam_01_9_u_YYYYMMDD_HHMMSS_ffffff_msk_type
    """
    parts = stem.split("_")
    # Ищем первую часть вида YYYYMMDD (8 цифр)
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            cam_name = "_".join(parts[:i])
            date_p = p           # YYYYMMDD
            time_p = parts[i+1] if i+1 < len(parts) else "000000"  # HHMMSS
            img_type = parts[-1]
            ts_str = f"{date_p[:4]}-{date_p[4:6]}-{date_p[6:]} {time_p[:2]}:{time_p[2:4]}:{time_p[4:6]}"
            return cam_name, ts_str, img_type
    return None


def _regen_osd_from_images(run_dir: Path) -> list[list]:
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

    # Проход 1: строим шаблоны из файлов с известными именами
    if not low_templates_complete():
        print(f"  [OSD] Построение LOW-шаблонов из {len(u_imgs)} изображений…")
        added_total: set[str] = set()
        for img_path in u_imgs:
            info = _parse_img_filename(img_path.stem)
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
        print(f"  [OSD] Шаблоны: {sorted(_LOW_TEMPLATES.keys())} ({len(_LOW_TEMPLATES)}/12)")

    # Проход 2: извлекаем OSD-метки
    import csv as _csv
    from datetime import datetime as _dt

    osd_log: list[list] = []
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

    rows_with_osd = [r for r in osd_log if r[3]]  # только те, где OSD распознан
    if not rows_with_osd:
        print("  [OSD] Нет распознанных меток — график пропущен")
        return

    def _parse(s: str) -> float:
        try:
            return _dt.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            return 0.0

    wall_ts  = [_parse(r[0]) for r in rows_with_osd]
    osd_ts   = [_parse(r[3]) for r in rows_with_osd]
    img_type = [r[2] for r in rows_with_osd]
    drifts   = [float(r[4]) for r in rows_with_osd if r[4] != ""]

    t0 = wall_ts[0]
    wall_rel = [(t - t0) for t in wall_ts]
    osd_rel  = [(t - t0) for t in osd_ts]
    drift_vals = [(o - w) for o, w in zip(osd_ts, wall_ts)]

    _TYPE_COLOR = {"baseline": "#888888", "heartbeat": "#8844bb",
                   "yolo": "#cc3333", "diff": "#228833"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), squeeze=False,
                             gridspec_kw={"height_ratios": [2.5, 1.5]})
    fig.suptitle("OSD-метка камеры vs wall-clock (по сохранённым кадрам)", fontsize=11)

    # Subplot 1: wall vs OSD (оба от t0, в секундах)
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

    # Subplot 2: дрейф OSD - wall (сек)
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
        description="Motion detection (LOW) + YOLOv8n на LOW-кадре: сохранение кадров с людьми"
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Путь к yolov8n.onnx")
    parser.add_argument("--conf", type=float, default=None, help="Порог confidence (env: YOLO_CONF, default: 0.35)")
    parser.add_argument("--nms", type=float, default=None, help="Порог NMS IoU (env: YOLO_NMS, default: 0.45)")
    parser.add_argument("--crop-pad", type=float, default=0.10,
                        help="Отступ вокруг bbox при вырезке кропа (доля от bbox, default: 0.10)")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Порог mean abs diff движения; иначе MOTION_DIFF_THRESHOLD из .env или 10",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tcp", action="store_true", help="RTSP через TCP")
    parser.add_argument("--crop-rel", type=str, default=None, metavar="X,Y,W,H")
    parser.add_argument("--open-timeout-ms", type=int, default=10000)
    parser.add_argument("--read-timeout-ms", type=int, default=10000)
    parser.add_argument("--stimeout-us", type=int, default=3_000_000)
    parser.add_argument("--min-laplacian-var", type=float, default=12.0)
    parser.add_argument("--min-gray-std", type=float, default=2.5)
    parser.add_argument("--save-extra-reads", type=int, default=12)
    parser.add_argument("--baseline-attempts", type=int, default=16)
    parser.add_argument(
        "--heartbeat-sec", type=float, default=None,
        help="Раз в N сек сохранять текущий LOW-кадр независимо от детекции; 0 = выкл; "
             "иначе MOTION_HEARTBEAT_SEC из .env или 600",
    )
    parser.add_argument(
        "--yolo-max-fps", type=float, default=None, metavar="FPS",
        help="Макс. частота запуска YOLO на одну камеру (fps). 0 = без ограничений. "
             "(env: YOLO_MAX_FPS, default: 2.0). Пример: 0.5 = раз в 2с.",
    )
    parser.add_argument(
        "--save-raw", action="store_true",
        help="Отладка: сохранять сырой LOW-кадр при каждом срабатывании порога движения "
             "(до YOLO, в images/raw/).",
    )
    parser.add_argument(
        "--cam-ts", action="store_true",
        help="Читать время камеры из OSD-оверлея LOW-кадра и добавлять в имя файла "
             "(cam_YYYYMMDD_HHMMSS). Требует models/osd_templates.npz.",
    )
    parser.add_argument(
        "--duration", type=float, default=0, metavar="SEC",
        help="Остановиться через N секунд после старта. 0 = бесконечно (default).",
    )
    parser.add_argument(
        "--cpu-interval", type=float, default=2.0, metavar="SEC",
        help="Интервал замера загрузки ЦПУ (сек). Требует psutil. 0 = не замерять. "
             "По умолчанию 2 сек.",
    )
    parser.add_argument(
        "--regen-from", type=Path, default=None, metavar="RUN_DIR",
        help="Перечитать CSV из существующего каталога прогона и перегенерировать графики.",
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
        print("Нужен python-dotenv: pip install python-dotenv", file=sys.stderr)
        return 1

    load_dotenv(args.env, override=True)

    sess = _load_model(args.model)
    if sess is None:
        return 1

    if args.cam_ts:
        try:
            from common.utils.osd_time import extract_osd_time as _extract_osd_time
        except ImportError:
            print("osd_time недоступен — --cam-ts отключён", file=sys.stderr)
            args.cam_ts = False
            _extract_osd_time = None
    else:
        _extract_osd_time = None

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        raw_t = (os.environ.get("MOTION_DIFF_THRESHOLD") or "").strip()
        threshold = float(raw_t) if raw_t else 10.0

    if args.heartbeat_sec is not None:
        heartbeat_sec = max(0.0, float(args.heartbeat_sec))
    else:
        raw_hb = (os.environ.get("MOTION_HEARTBEAT_SEC") or "").strip()
        heartbeat_sec = max(0.0, float(raw_hb)) if raw_hb else 600.0

    if args.conf is None:
        raw_conf = (os.environ.get("YOLO_CONF") or "").strip()
        args.conf = float(raw_conf) if raw_conf else 0.35

    if args.nms is None:
        raw_nms = (os.environ.get("YOLO_NMS") or "").strip()
        args.nms = float(raw_nms) if raw_nms else 0.45

    if args.yolo_max_fps is None:
        raw_mfps = (os.environ.get("YOLO_MAX_FPS") or "").strip()
        args.yolo_max_fps = float(raw_mfps) if raw_mfps else 2.0

    _log_file = _orig_stdout = _orig_stderr = _log_path = None

    out_dir = args.output
    if out_dir is None:
        out_dir = DEFAULT_OUTPUT / f"run_{ts_for_dir()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    _log_path = out_dir / "run.log"
    _log_file = open(_log_path, "w", encoding="utf-8", errors="replace", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, _log_file)
    sys.stderr = _Tee(_orig_stderr, _log_file)

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_capture_options(
        use_tcp=args.tcp, stimeout_us=args.stimeout_us
    )

    cameras = collect_cam_urls()
    active = [(k, v) for k, v in cameras if not skip_url(v)]
    if not active:
        print("Нет активных rtsp:// URL.", file=sys.stderr)
        return 1

    crop_cli = (args.crop_rel or "").strip() or None
    global_crop, global_crop_from = resolve_global_crop(
        crop_rel_arg=crop_cli,
        motion_crop_env=os.environ.get("MOTION_CROP_REL"),
    )
    if crop_cli and global_crop is None:
        return 1
    crop_by_cam = crop_map_for_cameras(active, global_crop=global_crop)

    var_low_url: dict[str, str] = {vn: url for vn, url in active}

    low_unique = sorted(set(var_low_url.values()))
    readers: dict[str, _CapReader] = {}
    for url in low_unique:
        cap = open_cap(url, open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms)
        if cap is None:
            print(f"  [!] LOW не удалось открыть: {redact_url(url)[:80]}…", file=sys.stderr)
        else:
            readers[url] = _CapReader(url, cap, args.open_timeout_ms, args.read_timeout_ms)

    opened_vars: list[tuple[str, str]] = [
        (vn, lu) for vn, lu in active if lu in readers
    ]
    if not opened_vars:
        print("Ни одна камера не открылась.", file=sys.stderr)
        for r in readers.values():
            r.stop()
        return 1

    vars_by_low: dict[str, list[str]] = defaultdict(list)
    for vn, lu in opened_vars:
        vars_by_low[lu].append(vn)

    url_id: dict[str, str] = {lu: " / ".join(vlist) for lu, vlist in vars_by_low.items()}

    # Подкаталоги изображений по камерам
    cam_dirs: dict[str, Path] = {}
    for vn, _ in opened_vars:
        cam_dir = images_dir / stem_from_var(vn)
        cam_dir.mkdir(exist_ok=True)
        (cam_dir / "diff").mkdir(exist_ok=True)
        cam_dirs[vn] = cam_dir

    prev_gray: dict[str, np.ndarray | None] = {vn: None for vn, _ in opened_vars}
    last_good_frame: dict[str, np.ndarray | None] = {vn: None for vn, _ in opened_vars}
    last_heartbeat: dict[str, float] = {vn: time.monotonic() for vn, _ in opened_vars}
    last_yolo_t: dict[str, float] = {vn: 0.0 for vn, _ in opened_vars}
    prev_pts: dict[str, float] = {lu: -1.0 for lu in low_unique}
    _MAX_LOW_IMPLAUSIBLE = 40
    _low_implausible: dict[str, int] = defaultdict(int)

    frame_log: list[list] = []   # [mono_s, ts_msk, url_id, ok, plausible, event]
    saves_log: list[list] = []   # [mono_s, ts_msk, cam, save_type]
    diffs_log: list[list] = []   # [mono_s, ts_msk, cam, diff]
    pts_log:   list[list] = []   # [mono_s, ts_msk, url_id, pts_ms]
    cpu_log: list[list] = []

    import json as _json
    from common.utils.time_msk import ts_iso as _ts_iso
    (out_dir / "run_params.json").write_text(_json.dumps({
        "started_at_msk": _ts_iso(),
        "script": "5_diff_yolo_boxes_low.py",
        "threshold": threshold,
        "conf": args.conf,
        "nms": args.nms,
        "model": str(args.model),
        "tcp": args.tcp,
        "cameras": [vn for vn, _ in opened_vars],
        "crop_global": list(global_crop) if global_crop else None,
        "heartbeat_sec": heartbeat_sec,
        "output": str(out_dir),
        "stream": "LOW-only",
        "yolo_max_fps": args.yolo_max_fps,
        "duration_sec": args.duration,
        "cpu_interval": args.cpu_interval,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Модель:    {args.model}")
    print(f"Conf:      {args.conf}  NMS: {args.nms}  Threshold: {threshold}")
    print(f"TCP:       {args.tcp}  Поток: LOW-only")
    print(f"Вывод:     {out_dir}")
    print(f"Камеры ({len(opened_vars)}): {', '.join(vn for vn, _ in opened_vars)}")
    print(f"Обрезка:   {global_crop!r}  [{global_crop_from}]")
    yolo_desc = (f"≤{args.yolo_max_fps} fps (интервал {1/args.yolo_max_fps:.1f}s)"
                 if args.yolo_max_fps > 0 else "без ограничений")
    print(f"YOLO:      {yolo_desc}")
    print("Останов: Ctrl+C\n")

    def _ts(cam_dt=None) -> str:
        pc = ts_for_file()
        if args.cam_ts and cam_dt is not None:
            return f"{ts_cam_for_file(cam_dt)}_{pc}"
        return pc

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
                f_bl, min_laplacian_var=args.min_laplacian_var, min_gray_std=args.min_gray_std
            ):
                frame_l = f_bl
                break
        if frame_l is None:
            for vn in var_list:
                print(f"  [!] {vn}: нет годного LOW для baseline", file=sys.stderr)
            continue
        for vn in var_list:
            frame_u = apply_crop_optional(frame_l, crop_by_cam[vn])
            prev_gray[vn] = prepare_gray(frame_u)
            last_good_frame[vn] = frame_u
            stem = stem_from_var(vn)
            bname = f"{stem}_{ts_for_file()}_baseline.jpg"
            cv2.imwrite(str(cam_dirs[vn] / bname), frame_u)
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

                # PTS-дубль (защита: reader обычно уже даёт только свежие кадры)
                if _pts > 0 and _pts == prev_pts[low_u]:
                    continue
                prev_pts[low_u] = _pts

                _plausible = frame_decode_plausible(
                    frame_l, min_laplacian_var=args.min_laplacian_var, min_gray_std=args.min_gray_std)
                frame_log.append([round(_t - t_start, 4), _ts_str, url_id[low_u], 1, int(_plausible), ""])

                if not _plausible:
                    _low_implausible[low_u] += 1
                    if _low_implausible[low_u] >= _MAX_LOW_IMPLAUSIBLE:
                        print(f"  [!] LOW {_MAX_LOW_IMPLAUSIBLE} битых кадров — переподключение",
                              file=sys.stderr)
                        reader.request_reconnect()
                        _low_implausible[low_u] = 0
                        for vn in var_list:
                            prev_gray[vn] = None
                    continue
                _low_implausible[low_u] = 0

                _now = time.monotonic()
                for vn in var_list:
                    crop = crop_by_cam[vn]
                    frame_u = apply_crop_optional(frame_l, crop)
                    gray = prepare_gray(frame_u)
                    last_good_frame[vn] = frame_u

                    prev = prev_gray[vn]
                    if prev is None:
                        prev_gray[vn] = gray
                        continue

                    diff = mean_abs_diff(prev, gray)
                    diffs_log.append([round(_now - t_start, 4), _ts_str, vn, round(diff, 3)])
                    if diff <= threshold:
                        prev_gray[vn] = gray
                        continue

                    # Сохраняем сырой кадр при каждом превышении порога
                    stem = stem_from_var(vn)
                    diff_name = f"{stem}_{_ts_str}_diff{diff:.2f}.jpg"
                    cv2.imwrite(str(cam_dirs[vn] / "diff" / diff_name), frame_u)
                    saves_log.append([round(_now - t_start, 4), _ts_str, vn, "diff"])

                    if args.yolo_max_fps > 0 and _now - last_yolo_t[vn] < 1.0 / args.yolo_max_fps:
                        prev_gray[vn] = gray
                        continue

                    prev_gray[vn] = gray
                    cam_dt = _extract_osd_time(frame_u) if _extract_osd_time else None

                    if args.save_raw:
                        raw_dir = cam_dirs[vn] / "raw"
                        raw_dir.mkdir(exist_ok=True)
                        stem = stem_from_var(vn)
                        raw_name = f"{stem}_{_ts(cam_dt)}_raw_diff{diff:.2f}.jpg"
                        cv2.imwrite(str(raw_dir / raw_name), frame_u)
                        saves_log.append([round(time.monotonic() - t_start, 4), _ts_str, vn, "raw"])
                        print(f"  [raw] {raw_name}")

                    last_yolo_t[vn] = time.monotonic()
                    detections = detect_people(
                        sess, frame_u, conf_threshold=args.conf, nms_threshold=args.nms
                    )
                    if not detections:
                        continue

                    annotated = draw_boxes(frame_u, detections)
                    stem = stem_from_var(vn)
                    ts = _ts(cam_dt)
                    fname = f"{stem}_{ts}_p{len(detections)}.jpg"
                    cv2.imwrite(str(cam_dirs[vn] / fname), annotated)
                    frame_log[-1][5] = "yolo"
                    saves_log.append([round(time.monotonic() - t_start, 4), _ts_str, vn, "yolo"])

                    h, w = frame_u.shape[:2]
                    crops_dir = cam_dirs[vn] / "crops"
                    crops_dir.mkdir(exist_ok=True)
                    for idx, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
                        bw, bh = x2 - x1, y2 - y1
                        px = int(bw * args.crop_pad)
                        py = int(bh * args.crop_pad)
                        x1c = max(0, x1 - px)
                        y1c = max(0, y1 - py)
                        x2c = min(w, x2 + px)
                        y2c = min(h, y2 + py)
                        if x2c > x1c and y2c > y1c:
                            crop_img = frame_u[y1c:y2c, x1c:x2c]
                            crop_name = f"{stem}_{ts}_p{idx}of{len(detections)}_conf{conf:.2f}.jpg"
                            cv2.imwrite(str(crops_dir / crop_name), crop_img)

                    print(f"  {fname}  diff={diff:.2f}  люди={len(detections)}")

                for vn in var_list:
                    if heartbeat_sec <= 0:
                        continue
                    now = time.monotonic()
                    if now - last_heartbeat[vn] < heartbeat_sec:
                        continue
                    hb = last_good_frame[vn]
                    if hb is None:
                        continue
                    last_heartbeat[vn] = now
                    stem = stem_from_var(vn)
                    hb_name = f"{stem}_{_ts()}_heartbeat.jpg"
                    cv2.imwrite(str(cam_dirs[vn] / hb_name), hb)
                    frame_log[-1][5] = "heartbeat"
                    saves_log.append([round(time.monotonic() - t_start, 4), ts_for_file(), vn, "heartbeat"])
                    print(f"  пульс {hb_name}")

    except KeyboardInterrupt:
        print("\nОстанов по Ctrl+C")
    finally:
        for r in readers.values():
            r.stop()
        cpu_log = cpu_monitor.stop()

        print("\nСохранение результатов…")

        if frame_log:
            csv_path = out_dir / "frames.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "url_id", "ok", "plausible", "event"])
                writer.writerows(frame_log)
            print(f"  frames.csv: {len(frame_log)} строк")

        if diffs_log:
            diffs_path = out_dir / "diffs.csv"
            with open(diffs_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cam", "diff"])
                writer.writerows(diffs_log)
            print(f"  diffs.csv:  {len(diffs_log)} записей")

        if saves_log:
            saves_path = out_dir / "saves.csv"
            with open(saves_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cam", "type"])
                writer.writerows(saves_log)
            print(f"  saves.csv:  {len(saves_log)} записей")

        if cpu_log:
            cpu_path = out_dir / "cpu.csv"
            with open(cpu_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cpu_pct"])
                writer.writerows(cpu_log)
            print(f"  cpu.csv:    {len(cpu_log)} замеров")

        if pts_log:
            pts_path = out_dir / "pts.csv"
            with open(pts_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "url_id", "pts_ms"])
                writer.writerows(pts_log)
            print(f"  pts.csv:    {len(pts_log)} записей")

        _save_charts(frame_log, cpu_log, saves_log, diffs_log, threshold, out_dir)
        _save_pts_chart(pts_log, cpu_log, out_dir)
        _save_run_stats(frame_log, pts_log, saves_log, diffs_log, out_dir)

        if _orig_stdout is not None:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
        if _log_file is not None:
            _log_file.close()
            print(f"  run.log → {_log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
