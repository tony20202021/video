"""
Переобработка сохранённых прогонов: YOLO-детекция на уже записанных кадрах.

Входные данные (один или несколько позиционных аргументов):
  - Каталог прогона:       .output/cameras/.../run_YYYYMMDD_HHMMSS_msk
  - Родительский каталог:  .output/cameras/5_diff_yolo_boxes_low  (находит все run_*)
  Можно смешивать.

Выход:
  .output/cameras/5_2_/run_<ts>/
    <run_name>/               — по одному на каждый входной прогон
      <cam>/
        <img>_yolo.jpg        — кадр с рамками (только при детекции)
        crops/
    detections.csv            — все новые детекции: image_ts, run, cam, file, x1..y2, conf
    timeline_chart.png        — сводный график: события из входных прогонов + новые YOLO
    cpu.csv / cpu_chart.png   — загрузка ЦПУ только на переобработку
    run_stats.json
    run.log

Usage:
    python scripts/cameras/5_2_yolo_boxes_files.py .output/cameras/5_diff_yolo_boxes_low
    python scripts/cameras/5_2_yolo_boxes_files.py run_dir1 run_dir2
    python scripts/cameras/5_2_yolo_boxes_files.py \\
        .output/cameras/4_motion_diff_low .output/cameras/5_diff_yolo_boxes_low
"""

from __future__ import annotations

import argparse
import csv
import json as _json
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.camera_run import (
    CpuMonitor as _CpuMonitor,
    Tee as _Tee,
    parse_img_filename as _parse_img_filename,
)
from common.utils.time_msk import ts_for_dir

MSK = timezone(timedelta(hours=3))

DEFAULT_OUTPUT = REPO_ROOT / ".output" / "cameras" / "5_2_yolo_boxes_files"
DEFAULT_MODEL  = REPO_ROOT / ".models" / "yolov8n.onnx"

YOLO_INPUT_SIZE = 640
PERSON_CLASS    = 0
BOX_COLOR       = (0, 255, 0)
BOX_THICKNESS   = 2
FONT            = cv2.FONT_HERSHEY_SIMPLEX

_SAVE_COLORS = {
    "baseline":  "#888888",
    "heartbeat": "#8844bb",
    "diff":      "#228833",
    "raw":       "#cc6600",
    "yolo":      "#cc3333",
}
_SAVE_LEVELS = {"baseline": 1, "heartbeat": 2, "diff": 3, "raw": 4, "yolo": 5}


# ─── YOLOv8n inference (copied from 5_diff_yolo_boxes_low.py) ─────────────────

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
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _preprocess(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w = bgr.shape[:2]
    scale = min(YOLO_INPUT_SIZE / w, YOLO_INPUT_SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (YOLO_INPUT_SIZE - nw) // 2
    pad_y = (YOLO_INPUT_SIZE - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
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
    preds = output[0].T
    person_scores = preds[:, 4 + PERSON_CLASS]
    mask = person_scores >= conf_threshold
    if not mask.any():
        return []
    scores  = person_scores[mask]
    bxywh   = preds[:, :4][mask]
    cx, cy, bw, bh = bxywh[:, 0], bxywh[:, 1], bxywh[:, 2], bxywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    indices = cv2.dnn.NMSBoxes(
        np.stack([x1, y1, bw, bh], axis=1).tolist(), scores.tolist(),
        conf_threshold, nms_threshold,
    )
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


def detect_people(sess, bgr: np.ndarray, *, conf_threshold: float, nms_threshold: float):
    h, w = bgr.shape[:2]
    blob, scale, pad_x, pad_y = _preprocess(bgr)
    output = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    return _postprocess(output, orig_w=w, orig_h=h, scale=scale,
                        pad_x=pad_x, pad_y=pad_y,
                        conf_threshold=conf_threshold, nms_threshold=nms_threshold)


def draw_boxes(bgr: np.ndarray, detections) -> np.ndarray:
    out = bgr.copy()
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"{conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - baseline - 2), (x1 + tw, y1), BOX_COLOR, -1)
        cv2.putText(out, label, (x1, y1 - baseline - 1), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _expand_inputs(paths: list[Path]) -> list[tuple[Path, str]]:
    """Expand parent dirs to run_* subdirs.

    Returns list of (run_dir, parent_stem) where parent_stem is the name of the
    containing script-output dir (e.g. '4_motion_diff_low'), or '' if the path
    was itself a run dir passed directly.
    """
    runs: list[tuple[Path, str]] = []
    for p in paths:
        if not p.exists():
            print(f"  [!] Не найдено: {p}", file=sys.stderr)
            continue
        if (p / "frames.csv").exists():
            runs.append((p, ""))
        else:
            subs = sorted(
                c for c in p.iterdir()
                if c.is_dir() and c.name.startswith("run_") and (c / "frames.csv").exists()
            )
            if subs:
                for s in subs:
                    runs.append((s, p.name))
            else:
                print(f"  [!] Нет run_* с frames.csv в: {p}", file=sys.stderr)
    return runs


def _find_images(run_dir: Path) -> dict[str, list[Path]]:
    """Returns {cam_stem: [jpg_paths]} — only motion-event images.

    For runs with cam subdirs (S5/S6 style):
      - If cam/diff/ exists and has images → use only those (motion-diff frames).
      - Otherwise → all direct .jpg in the cam dir.
    For flat runs (S4 style, no cam subdirs): all .jpg in images/ directly.
    """
    images_dir = run_dir / "images"
    if not images_dir.exists():
        return {}
    cam_dirs = [d for d in sorted(images_dir.iterdir()) if d.is_dir()]
    if cam_dirs:
        result: dict[str, list[Path]] = {}
        for d in cam_dirs:
            diff_dir = d / "diff"
            if diff_dir.is_dir():
                imgs = sorted(p for p in diff_dir.iterdir() if p.suffix.lower() == ".jpg")
            else:
                imgs = sorted(p for p in d.iterdir() if p.suffix.lower() == ".jpg")
            if imgs:
                result[d.name] = imgs
        return result
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() == ".jpg")
    return {"cam": imgs} if imgs else {}


def _load_csv(path: Path, min_cols: int) -> list[list]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return [r for r in rows[1:] if len(r) >= min_cols]


def _parse_ts_msk(ts_str: str) -> float:
    """Parse YYYYMMDD_HHMMSS_ffffff_msk to epoch float."""
    try:
        parts = ts_str.split("_")
        d, t = parts[0], parts[1]
        us = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        dt = datetime(int(d[:4]), int(d[4:6]), int(d[6:]),
                      int(t[:2]), int(t[2:4]), int(t[4:6]),
                      us, tzinfo=MSK)
        return dt.timestamp()
    except Exception:
        return 0.0


def _img_ts_to_mono(img_ts_str: str, wall_t0_epoch: float, first_mono_s: float) -> float:
    """Convert 'YYYY-MM-DD HH:MM:SS' (from parse_img_filename) to mono_s for chart."""
    try:
        dt = datetime.strptime(img_ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=MSK)
        return dt.timestamp() - wall_t0_epoch + first_mono_s
    except Exception:
        return 0.0


# ─── Charts ───────────────────────────────────────────────────────────────────

def _save_combined_chart(run_data: list[dict], out_path: Path) -> None:
    """Сводный график: для каждого прогона — интервалы кадров + сохранения + новые YOLO.

    run_data list items:
      label         : str  (run dir name)
      frame_log     : [[mono_s, ts_msk, url_id, ok, plausible, event], ...]
      saves_log     : [[mono_s, ts_msk, cam, type], ...]  (old yolo filtered out)
      diffs_log     : [[mono_s, ts_msk, cam, diff_val], ...]
      threshold     : float
      wall_t0       : float  (epoch of first frame in run)
      first_mono_s  : float  (mono_s of first frame)
      yolo_new      : [[cam_stem, img_stem, n_people, mono_approx], ...]
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as _ticker
        import numpy as _np
        import matplotlib.lines as _mlines
    except ImportError:
        print("  [!] matplotlib не найден — график не сохранён")
        return
    if not run_data:
        return

    def _ts_parts(ts_str: str):
        try:
            p = ts_str.split("_")
            t = p[1]
            return int(t[0:2]), int(t[2:4]), int(t[4:6]), int(p[2])
        except Exception:
            return None

    # One subplot per run (saves + new YOLO events only)
    n_rows = sum(1 for d in run_data if d["frame_log"])
    if n_rows == 0:
        return

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(16, max(n_rows * 2.5, 4)),
        squeeze=False,
    )
    fig.suptitle("5_2 — события по прогонам (переобработка YOLO)", fontsize=11)

    ax_idx = 0

    for d in run_data:
        label     = d["label"]
        frame_log = d["frame_log"]
        saves_log = d["saves_log"]
        yolo_new  = d["yolo_new"]  # [[cam_stem, img_stem, n_people, mono_approx], ...]

        if not frame_log:
            continue

        t_max = max(row[0] for row in frame_log)

        _t0_ts    = frame_log[0][1]
        _t0_parts = _ts_parts(_t0_ts)
        _t0_abs   = (_t0_parts[0] * 3600 + _t0_parts[1] * 60 + _t0_parts[2]) if _t0_parts else 0
        _x_left   = -(_t0_abs % 60)

        _tick_range = t_max * 1.02 - _x_left
        _tick_step  = next((s for s in [60, 120, 180, 300, 600, 900, 1800]
                            if _tick_range / s <= 20), 1800)
        _tick_pos   = list(range(int(_x_left), int(t_max * 1.02) + _tick_step, _tick_step))

        def _x_fmt(x, _pos, _t0=_t0_abs):
            total = int(_t0 + x)
            return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

        # ── Saves subplot (only) ──
        by_stype: dict[str, list[float]] = defaultdict(list)
        for r in saves_log:
            try:
                by_stype[r[3]].append(float(r[0]))
            except (ValueError, IndexError):
                pass
        for _, _, n_people, mono_approx in yolo_new:
            by_stype["yolo"].append(mono_approx)

        if by_stype or yolo_new:
            ax = axes[ax_idx][0]
            y_pos: dict[str, float] = {}
            y = 1.0
            for stype in sorted(by_stype, key=lambda s: _SAVE_LEVELS.get(s, 99)):
                y_pos[stype] = y
                ax.scatter(by_stype[stype], [y] * len(by_stype[stype]),
                           color=_SAVE_COLORS.get(stype, "#888888"),
                           s=25, alpha=0.8, marker="o", label=f"{stype} ({len(by_stype[stype])})")
                y += 1.0
            # Annotate YOLO events with person count
            yolo_y = y_pos.get("yolo", y - 1.0)
            for cam_stem, img_stem, n_people, mono_approx in yolo_new:
                ax.annotate(
                    f"p{n_people}", xy=(mono_approx, yolo_y),
                    fontsize=5, ha="left", va="center", color="red",
                    xytext=(3, 0), textcoords="offset points",
                )
            ax.set_yticks(list(y_pos.values()))
            ax.set_yticklabels(list(y_pos.keys()), fontsize=7)
            ax.set_ylim(0, y + 0.5)
            ax.set_xlim(_x_left, t_max * 1.02)
            ax.set_title(
                f"{label} — сохранения  |  YOLO-новых: {len(yolo_new)}", fontsize=8
            )
            ax.legend(loc="upper right", fontsize=7, ncol=5)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.xaxis.set_major_formatter(_ticker.FuncFormatter(_x_fmt))
            ax.xaxis.set_major_locator(_ticker.FixedLocator(_tick_pos))
            ax_idx += 1

    if ax_idx > 0:
        axes[ax_idx - 1][0].set_xlabel("время МСК")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110)
    plt.close()
    print(f"  timeline_chart.png → {out_path}")


def _save_cpu_chart(cpu_log: list, out_path: Path,
                    timing_log: "list | None" = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not cpu_log:
        return

    has_timing = bool(timing_log)
    n_rows = 2 if has_timing else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), squeeze=False)

    # ── CPU subplot ──
    ax = axes[0][0]
    t = [r[0] for r in cpu_log]
    v = [float(r[2]) for r in cpu_log]
    ax.plot(t, v, color="#2255cc", linewidth=1.2, label="ЦПУ %")
    ax.fill_between(t, v, alpha=0.18, color="#2255cc")
    ax.set_ylim(0, 105)
    ax.set_ylabel("ЦПУ, %")
    ax.set_title("5_2 — загрузка ЦПУ")
    ax.grid(True, linestyle="--", alpha=0.35)
    if not has_timing:
        ax.set_xlabel("время от старта, с")

    # ── CPU freq secondary axis ──
    freq_pdh  = [float(r[3]) for r in cpu_log if len(r) > 3]
    freq_step = [float(r[4]) for r in cpu_log if len(r) > 4]
    if (freq_pdh and any(f > 0 for f in freq_pdh)) or \
       (freq_step and any(f > 0 for f in freq_step)):
        t_freq = [r[0] for r in cpu_log if len(r) > 3]
        ax2f = ax.twinx()
        all_vals = [f for f in freq_pdh + freq_step if f > 0]
        if freq_pdh and any(f > 0 for f in freq_pdh):
            ax2f.plot(t_freq, freq_pdh, color="#ffbb55", linewidth=1.0,
                      linestyle="--", alpha=0.5,
                      marker=".", markersize=4,
                      markerfacecolor="#ff8800", markeredgewidth=0,
                      label="МГц (PDH)")
        if freq_step and any(f > 0 for f in freq_step):
            t_step = [r[0] for r in cpu_log if len(r) > 4]
            ax2f.step(t_step, freq_step, color="#aaaaaa", linewidth=0.8,
                      alpha=0.6, where="post", label="МГц (P-state)")
        ax2f.set_ylabel("частота, МГц", color="#ff8800", fontsize=8)
        ax2f.tick_params(axis="y", labelcolor="#ff8800", labelsize=7)
        ax2f.legend(loc="lower right", fontsize=7)
        f_min = min(all_vals)
        f_max = max(all_vals)
        pad = max((f_max - f_min) * 0.15, 50)
        ax2f.set_ylim(max(0, f_min - pad), f_max + pad)

    # ── YOLO timing subplot ──
    if has_timing:
        ax2 = axes[1][0]
        ts   = [r[0] for r in timing_log]
        inf  = [r[1] for r in timing_log]   # inference_ms
        slp  = [r[2] for r in timing_log]   # sleep_ms
        avg_inf = sum(inf) / len(inf) if inf else 0
        avg_slp = sum(slp) / len(slp) if slp else 0
        ax2.bar(ts, inf, width=0.3, color="#cc3333", alpha=0.8,
                label=f"inference  avg {avg_inf:.0f}ms")
        ax2.bar(ts, slp, width=0.3, bottom=inf, color="#44aa44", alpha=0.6,
                label=f"sleep  avg {avg_slp:.0f}ms")
        # Dots at bar tops to show envelope when bars are dense
        ax2.scatter(ts, inf, s=6, color="#cc3333", zorder=5)
        _slp_t = [ts[i] for i in range(len(slp)) if slp[i] > 0]
        _slp_v = [inf[i] + slp[i] for i in range(len(slp)) if slp[i] > 0]
        if _slp_t:
            ax2.scatter(_slp_t, _slp_v, s=6, color="#44aa44", zorder=5)
        if inf:
            _top = max(inf[i] + slp[i] for i in range(len(inf)))
            ax2.set_ylim(0, _top * 1.12)
        ax2.set_ylabel("мс / кадр")
        ax2.set_xlabel("время от старта, с")
        ax2.set_title("YOLO: время инференса и сна между вызовами")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110)
    plt.close()
    print(f"  cpu_chart.png → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLO re-detection on saved camera run images"
    )
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Run dirs or parent dirs (with run_* subdirs)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=None,
                        help="Порог confidence (env: YOLO_CONF, default: 0.35)")
    parser.add_argument("--nms", type=float, default=None,
                        help="Порог NMS IoU (env: YOLO_NMS, default: 0.45)")
    parser.add_argument("--crop-pad", type=float, default=0.40,
                        help="Crop padding fraction (default 0.40)")
    parser.add_argument("--yolo-max-fps", type=float, default=None, metavar="FPS",
                        help="Макс. скорость YOLO-инференса, изображений/с "
                             "(0 = без ограничений). "
                             "Env: YOLO_MAX_FPS. Default 2.0.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output dir (default: .output/cameras/5_2_yolo_boxes_files/run_<ts>)")
    parser.add_argument("--cpu-interval", type=float, default=2.0)
    args = parser.parse_args()

    # ── Single-instance guard ──────────────────────────────────────────────────
    try:
        import psutil
        _this_pid  = os.getpid()
        _this_name = Path(__file__).name
        _py_names  = {"python.exe", "python", "python3", "python3.exe"}
        _others = [
            p.pid for p in psutil.process_iter(["pid", "name", "cmdline"])
            if p.pid != _this_pid
            and (p.info.get("name") or "").lower() in _py_names
            and any(_this_name in (c or "") for c in (p.info.get("cmdline") or []))
            # Exclude conda infrastructure (python.exe conda-script.py run ... python script.py ...)
            and not any("conda" in (c or "").lower() for c in (p.info.get("cmdline") or []))
        ]
        if _others:
            print(f"[!] {_this_name} уже запущен (PID: {_others}). Завершение.",
                  file=sys.stderr)
            return 1
    except ImportError:
        pass  # psutil not available — skip check

    sess = _load_model(args.model)
    if sess is None:
        return 1

    if args.conf is None:
        _raw = (os.environ.get("YOLO_CONF") or "").strip()
        args.conf = float(_raw) if _raw else 0.35

    if args.nms is None:
        _raw = (os.environ.get("YOLO_NMS") or "").strip()
        args.nms = float(_raw) if _raw else 0.45

    if args.yolo_max_fps is None:
        _raw_mfps = (os.environ.get("YOLO_MAX_FPS") or "").strip()
        args.yolo_max_fps = float(_raw_mfps) if _raw_mfps else 2.0

    _yolo_interval = (1.0 / args.yolo_max_fps) if args.yolo_max_fps > 0 else 0.0

    run_pairs = _expand_inputs(args.inputs)
    if not run_pairs:
        print("Нет run-каталогов для обработки.", file=sys.stderr)
        return 1

    out_dir = args.output or (DEFAULT_OUTPUT / f"run_{ts_for_dir()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    _log_raw    = open(out_dir / "run.log", "w", encoding="utf-8", errors="replace")
    _log_fd     = _log_raw.fileno()
    _log_last_sync = [time.monotonic()]

    class _SyncFile:
        """Wraps a text file: flush + periodic os.fsync so viewers see updates promptly."""
        def write(self, data: str) -> None:
            _log_raw.write(data)
            _log_raw.flush()
            now = time.monotonic()
            if now - _log_last_sync[0] >= 1.0:
                try:
                    os.fsync(_log_fd)
                except OSError:
                    pass
                _log_last_sync[0] = now
        def flush(self) -> None:
            _log_raw.flush()
        def close(self) -> None:
            _log_raw.flush()
            try:
                os.fsync(_log_fd)
            except OSError:
                pass
            _log_raw.close()

    _log_file   = _SyncFile()
    _orig_out   = sys.stdout
    _orig_err   = sys.stderr
    sys.stdout  = _Tee(_orig_out, _log_file)
    sys.stderr  = _Tee(_orig_err, _log_file)

    t_start     = time.monotonic()
    cpu_monitor = _CpuMonitor(interval=max(args.cpu_interval, 0.5))
    cpu_active  = args.cpu_interval > 0 and cpu_monitor.start(t_start)

    # Periodic CPU snapshot thread: saves cpu.csv + cpu_chart.png every 60s
    _periodic_stop = threading.Event()

    def _periodic_cpu_save() -> None:
        while not _periodic_stop.wait(timeout=60.0):
            _snap = cpu_monitor.snapshot()
            if not _snap:
                continue
            try:
                with open(out_dir / "cpu.csv", "w", newline="", encoding="utf-8") as _f:
                    _w = csv.writer(_f)
                    _w.writerow(["mono_s", "ts_msk", "cpu_pct", "freq_mhz_pdh", "freq_mhz_step"])
                    _w.writerows(_snap)
                _tsnap = list(timing_log)
                if _tsnap:
                    with open(out_dir / "yolo_timing.csv", "w", newline="", encoding="utf-8") as _f:
                        _w = csv.writer(_f)
                        _w.writerow(["mono_s", "inference_ms", "sleep_ms"])
                        _w.writerows(_tsnap)
                _save_cpu_chart(_snap, out_dir / "cpu_chart.png", _tsnap or None)
            except Exception:
                pass

    _periodic_thread = threading.Thread(target=_periodic_cpu_save, daemon=True, name="cpu-periodic")
    if cpu_active:
        _periodic_thread.start()

    # Save launch params immediately so the run is identifiable even if it crashes
    run_params = {
        "script":        "5_2_yolo_boxes_files",
        "inputs":        [str(p) for p in args.inputs],
        "model":         str(args.model),
        "conf":          args.conf,
        "nms":           args.nms,
        "crop_pad":      args.crop_pad,
        "yolo_max_fps":  args.yolo_max_fps,
        "cpu_interval":  args.cpu_interval,
        "out_dir":       str(out_dir),
    }
    (out_dir / "run_params.json").write_text(
        _json.dumps(run_params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    yolo_rate = (f"≤{args.yolo_max_fps} fps (интервал {_yolo_interval:.2f}s)"
                 if _yolo_interval > 0 else "без ограничений")
    print(f"Модель:       {args.model}")
    print(f"Conf / NMS:   {args.conf} / {args.nms}")
    print(f"YOLO rate:    {yolo_rate}")
    print(f"Вывод:        {out_dir}")
    print(f"Прогонов:     {len(run_pairs)}")
    for rd, parent_stem in run_pairs:
        prefix = f"{parent_stem}/" if parent_stem else ""
        _imgs_by_cam = _find_images(rd)
        if _imgs_by_cam:
            _total = sum(len(v) for v in _imgs_by_cam.values())
            _cam_counts = ", ".join(f"{c}: {len(v)}" for c, v in _imgs_by_cam.items())
            print(f"  {prefix}{rd.name}  [{_cam_counts}  →  {_total} всего]")
        else:
            print(f"  {prefix}{rd.name}  [нет diff-изображений]")
    print()

    all_detections:       list[list] = []
    run_data:             list[dict] = []
    timing_log:           list[list] = []  # [mono_s, inference_ms, sleep_ms]
    grand_total_checked   = 0
    grand_total_detected  = 0

    for run_dir, parent_stem in run_pairs:
        run_name = run_dir.name
        label    = f"{parent_stem}/{run_name}" if parent_stem else run_name
        print(f"── {label} ──────────────────────────────────────")

        # Output goes into parent_stem subdir if the source had one.
        # Created lazily — only when the first detection is actually written.
        run_out = (out_dir / parent_stem / run_name) if parent_stem else (out_dir / run_name)
        _run_out_created = [False]

        def _ensure_run_out() -> None:
            if not _run_out_created[0]:
                run_out.mkdir(parents=True, exist_ok=True)
                _run_out_created[0] = True

        # Load original CSVs
        frame_raw = _load_csv(run_dir / "frames.csv", 5)
        saves_raw = _load_csv(run_dir / "saves.csv", 4)
        diffs_raw = _load_csv(run_dir / "diffs.csv", 4)

        frame_log: list[list] = []
        for r in frame_raw:
            try:
                frame_log.append([float(r[0]), r[1], r[2], int(r[3]), int(r[4]),
                                   r[5] if len(r) > 5 else ""])
            except (ValueError, IndexError):
                pass

        saves_log: list[list] = []
        for r in saves_raw:
            if r[3] == "yolo":
                continue  # drop old YOLO saves; replaced by new detections
            try:
                saves_log.append([float(r[0]), r[1], r[2], r[3]])
            except (ValueError, IndexError):
                pass

        diffs_log: list[list] = []
        for r in diffs_raw:
            try:
                diffs_log.append([float(r[0]), r[1], r[2], float(r[3])])
            except (ValueError, IndexError):
                pass

        threshold = 10.0
        params_p = run_dir / "run_params.json"
        if params_p.exists():
            try:
                threshold = float(
                    _json.loads(params_p.read_text(encoding="utf-8")).get("threshold", 10.0)
                )
            except Exception:
                pass

        # wall_t0 for mono→time mapping on chart
        wall_t0_epoch = _parse_ts_msk(frame_log[0][1]) if frame_log else 0.0
        first_mono_s  = frame_log[0][0] if frame_log else 0.0

        images_by_cam = _find_images(run_dir)
        if not images_by_cam:
            print(f"  [!] Нет изображений в {run_dir / 'images'}")
            continue

        yolo_new: list[list] = []  # [cam_stem, img_stem, n_people, mono_approx]
        n_detected = 0
        n_total    = 0

        for cam_stem, img_paths in images_by_cam.items():
            cam_out   = run_out / cam_stem
            crops_dir = cam_out / "crops"

            print(f"  {cam_stem}: {len(img_paths)} изображений")
            for img_path in img_paths:
                n_total += 1
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                _t_yolo_start = time.monotonic()

                detections = detect_people(sess, frame,
                                           conf_threshold=args.conf,
                                           nms_threshold=args.nms)
                _inference_ms = (time.monotonic() - _t_yolo_start) * 1000

                _slept_ms = 0.0
                if _yolo_interval > 0:
                    _remaining = _yolo_interval - (time.monotonic() - _t_yolo_start)
                    if _remaining > 0:
                        _slept_ms = _remaining * 1000
                        time.sleep(_remaining)

                timing_log.append([round(_t_yolo_start - t_start, 3),
                                    round(_inference_ms, 1), round(_slept_ms, 1)])
                if not detections:
                    continue

                n_detected += 1
                _ensure_run_out()
                cam_out.mkdir(exist_ok=True)
                crops_dir.mkdir(exist_ok=True)

                annotated = draw_boxes(frame, detections)
                cv2.imwrite(str(cam_out / (img_path.stem + "_yolo.jpg")), annotated)

                # Crops (recalculated from scratch)
                h, w = frame.shape[:2]
                for idx, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
                    bw_box = x2 - x1;  bh_box = y2 - y1
                    px = int(bw_box * args.crop_pad);  py = int(bh_box * args.crop_pad)
                    x1c = max(0, x1 - px);  y1c = max(0, y1 - py)
                    x2c = min(w, x2 + px);  y2c = min(h, y2 + py)
                    if x2c > x1c and y2c > y1c:
                        crop_name = (f"{img_path.stem}"
                                     f"_p{idx}of{len(detections)}_conf{conf:.2f}.jpg")
                        cv2.imwrite(str(crops_dir / crop_name), frame[y1c:y2c, x1c:x2c])

                # mono_approx for chart
                parsed = _parse_img_filename(img_path.stem)
                mono_approx = 0.0
                if parsed and wall_t0_epoch:
                    mono_approx = _img_ts_to_mono(parsed[1], wall_t0_epoch, first_mono_s)

                yolo_new.append([cam_stem, img_path.stem, len(detections), mono_approx])
                for x1, y1, x2, y2, conf in detections:
                    all_detections.append([
                        img_path.stem, run_name, cam_stem, img_path.name,
                        x1, y1, x2, y2, round(conf, 4),
                    ])
                print(f"    {img_path.name}  →  {len(detections)} чел.")

        grand_total_checked  += n_total
        grand_total_detected += n_detected
        print(f"  Итого: {n_detected} с людьми / {n_total} проверено")
        if n_detected == 0:
            print("  (нет детекций YOLO)")
        print()

        run_data.append({
            "label":        label,
            "frame_log":    frame_log,
            "saves_log":    saves_log,
            "diffs_log":    diffs_log,
            "threshold":    threshold,
            "wall_t0":      wall_t0_epoch,
            "first_mono_s": first_mono_s,
            "yolo_new":     yolo_new,
        })

        # Промежуточное обновление графика и detections.csv после каждого прогона
        if all_detections:
            try:
                with open(out_dir / "detections.csv", "w", newline="", encoding="utf-8") as _f:
                    _w = csv.writer(_f)
                    _w.writerow(["image_ts", "run_name", "cam", "filename",
                                 "x1", "y1", "x2", "y2", "conf"])
                    _w.writerows(all_detections)
            except Exception:
                pass
        _save_combined_chart(run_data, out_dir / "timeline_chart.png")

    _periodic_stop.set()
    cpu_log = cpu_monitor.stop()

    # detections.csv
    if all_detections:
        det_path = out_dir / "detections.csv"
        with open(det_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_ts", "run_name", "cam", "filename", "x1", "y1", "x2", "y2", "conf"])
            w.writerows(all_detections)
        print(f"detections.csv: {len(all_detections)} записей → {det_path}")
    else:
        print("Детекций не найдено.")

    # cpu.csv
    if cpu_log:
        with open(out_dir / "cpu.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mono_s", "ts_msk", "cpu_pct", "freq_mhz"])
            w.writerows(cpu_log)

    # yolo_timing.csv
    if timing_log:
        with open(out_dir / "yolo_timing.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mono_s", "inference_ms", "sleep_ms"])
            w.writerows(timing_log)

    # Charts
    _save_combined_chart(run_data, out_dir / "timeline_chart.png")
    _save_cpu_chart(cpu_log, out_dir / "cpu_chart.png", timing_log or None)

    # run_stats.json
    stats = {
        "input_runs":          [f"{ps}/{rd.name}" if ps else str(rd) for rd, ps in run_pairs],
        "images_yolo_checked": grand_total_checked,
        "images_with_people":  grand_total_detected,
        "detections_total":    len(all_detections),
        "duration_sec":        round(time.monotonic() - t_start, 1),
        "model":               str(args.model),
        "conf":                args.conf,
        "nms":                 args.nms,
    }
    (out_dir / "run_stats.json").write_text(
        _json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sys.stdout = _orig_out
    sys.stderr = _orig_err
    _log_file.close()
    print(f"\nГотово. Время: {stats['duration_sec']} с.  Вывод: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
