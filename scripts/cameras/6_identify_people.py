"""
Motion → YOLO → ML-пайплайн: классификация группы + идентификация жителей.

= 5_diff_yolo_boxes_low + GroupClassifier + PersonIdentifier (src/ml/pipeline.py)

Классификатор делит людей на группы: resident / courier / delivery / utilities / other.
Идентификатор по эмбеддингам тела определяет конкретного жителя (если есть эмбеддинги).

Побочные потоки:
  _CpuMonitor  — замер загрузки ЦПУ раз в N сек (требует psutil)

Выход (run_<ts>/):
  images/              — baseline, кадры с bbox+метками группы/ID, heartbeat
  images/<cam>/diff/   — сырые кадры при срабатывании motion
  images/<cam>/crops/  — вырезки отдельных людей
  images/<cam>/raw/    — сырые кадры при --save-raw
  frames.csv           — метка времени каждого кадра LOW-потока
  saves.csv            — моменты сохранений с типом
  diffs.csv            — сырые значения frame diff
  pts.csv              — FFmpeg PTS для анализа дрейфа
  cpu.csv              — загрузка ЦПУ с периодичностью --cpu-interval
  detections.csv       — детекции людей: группа, person_id, conf
  charts.png           — интервалы кадров + дифы + сохранения + ЦПУ
  pts_chart.png        — анализ дрейфа PTS / wall clock / mono
  osd_chart.png        — OSD vs wall-clock (только --regen-from)
  person_timeline.png  — хронология детекций по person_id / группе
  run_stats.json       — метрики прогона
  run_params.json
  run.log

Модели:
  .models/yolov8n.onnx       — детектор людей (скачать через scripts/setup_models.py)
  .models/classify/v*.onnx   — классификатор группы  (путь из config.yaml)
  .models/identify.onnx      — идентификатор по эмбеддингам (путь из config.yaml)
  .models/embeddings.json    — эмбеддинги известных жителей {"name": [[...], ...]}

Usage:
    python scripts/cameras/6_identify_people.py
    python scripts/cameras/6_identify_people.py --config config.yaml
    python scripts/cameras/6_identify_people.py --embeddings .models/embeddings.json
    python scripts/cameras/6_identify_people.py --no-ml
    python scripts/cameras/6_identify_people.py --duration 3600
    python scripts/cameras/6_identify_people.py --regen-from .output/.../run_xxx
"""

from __future__ import annotations

import argparse
import csv
import json as _json
import os
import sys
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
from common.utils.camera_run import (
    CapReader as _CapReader,
    CpuMonitor as _CpuMonitor,
    Tee as _Tee,
    parse_img_filename as _parse_img_filename,
    regen_osd_from_images as _regen_osd_from_images,
    save_charts as _save_charts_base,
    save_osd_chart as _save_osd_chart,
    save_pts_chart as _save_pts_chart,
    save_run_stats as _save_run_stats,
)
from common.utils.motion_utils import (
    ffmpeg_capture_options,
    frame_decode_plausible,
    mean_abs_diff,
    open_cap,
    prepare_gray,
    redact_url,
    skip_url,
    stem_from_var,
)
from common.utils.time_msk import ts_cam_for_file, ts_for_dir, ts_for_file

DEFAULT_ENV             = REPO_ROOT / ".env"
DEFAULT_OUTPUT          = REPO_ROOT / ".output" / "cameras" / "6_identify_people"
DEFAULT_MODEL           = REPO_ROOT / ".models" / "yolov8n.onnx"
DEFAULT_CONFIG          = REPO_ROOT / "config.yaml"
DEFAULT_EMBEDDINGS      = REPO_ROOT / ".models" / "embeddings.json"

YOLO_INPUT_SIZE = 640
PERSON_CLASS    = 0
BOX_COLOR       = (0, 255, 0)
BOX_THICKNESS   = 2
FONT            = cv2.FONT_HERSHEY_SIMPLEX

_GROUP_COLOR = {
    "resident":  (0, 200, 0),
    "courier":   (0, 165, 255),
    "delivery":  (255, 165, 0),
    "utilities": (160, 0, 160),
    "other":     (128, 128, 128),
    "unknown":   (128, 128, 128),
}



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
            f"  Скачать: python scripts/setup_models.py",
            file=sys.stderr,
        )
        return None
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return sess


def _preprocess(bgr: np.ndarray) -> "tuple[np.ndarray, float, int, int]":
    h, w = bgr.shape[:2]
    scale = min(YOLO_INPUT_SIZE / w, YOLO_INPUT_SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (YOLO_INPUT_SIZE - nw) // 2
    pad_y = (YOLO_INPUT_SIZE - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
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
) -> "list[tuple[int, int, int, int, float]]":
    preds = output[0].T
    boxes_xywh   = preds[:, :4]
    person_scores = preds[:, 4 + PERSON_CLASS]

    mask = person_scores >= conf_threshold
    if not mask.any():
        return []

    scores = person_scores[mask]
    bxywh  = boxes_xywh[mask]

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
) -> "list[tuple[int, int, int, int, float]]":
    h, w = bgr.shape[:2]
    blob, scale, pad_x, pad_y = _preprocess(bgr)
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: blob})[0]
    return _postprocess(
        output,
        orig_w=w, orig_h=h,
        scale=scale, pad_x=pad_x, pad_y=pad_y,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )


def draw_boxes(bgr: np.ndarray, detections: "list[tuple[int, int, int, int, float]]") -> np.ndarray:
    out = bgr.copy()
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"{conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - baseline - 2), (x1 + tw, y1), BOX_COLOR, -1)
        cv2.putText(out, label, (x1, y1 - baseline - 1), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _draw_boxes_ml(bgr: np.ndarray, results: list) -> np.ndarray:
    """Рисует bounding boxes с цветом по группе и меткой group/person_id."""
    out = bgr.copy()
    for r in results:
        x1, y1, x2, y2 = r.bbox
        color = _GROUP_COLOR.get(r.group_class, (128, 128, 128))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if r.person_id:
            label = f"{r.person_id}({r.identify_conf:.2f})"
        else:
            label = f"{r.group_class}({r.group_conf:.2f})"

        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        bg_y1 = max(0, y1 - th - baseline - 2)
        cv2.rectangle(out, (x1, bg_y1), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - baseline - 1), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _save_charts(frame_log, cpu_log, saves_log, diffs_log, threshold, out_dir):
    _save_charts_base(frame_log, cpu_log, saves_log, diffs_log, threshold, out_dir,
                      title="6_identify_people")


def _save_person_timeline(detections_log: list, out_dir: Path) -> None:
    """График: хронология детекций по person_id / группе."""
    if not detections_log:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as _ticker
    except ImportError:
        return

    _ENT_COLORS = {
        "resident":  "#228833",
        "courier":   "#cc6600",
        "delivery":  "#0055cc",
        "utilities": "#770077",
        "other":     "#888888",
        "unknown":   "#888888",
    }

    by_cam: dict[str, dict[str, list[float]]] = {}
    for row in detections_log:
        mono_s = float(row[0])
        cam    = row[2]
        person_id = row[10] if len(row) > 10 else ""
        group     = row[8]  if len(row) > 8  else "unknown"
        entity    = person_id if person_id else group
        by_cam.setdefault(cam, {}).setdefault(entity, []).append(mono_s)

    all_entities: list[str] = []
    for cam_map in by_cam.values():
        for e in cam_map:
            if e not in all_entities:
                all_entities.append(e)

    n = len(all_entities)
    if n == 0:
        return

    y_idx = {e: i + 1 for i, e in enumerate(all_entities)}
    cam_list = sorted(by_cam.keys())
    cam_markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(14, max(3, n * 0.7 + 1.5)))
    fig.suptitle("Timeline: детекции людей по времени", fontsize=11)

    total_events = 0
    for ci, cam in enumerate(cam_list):
        cam_mk = cam_markers[ci % len(cam_markers)]
        for entity, times in sorted(by_cam[cam].items()):
            y = y_idx[entity]
            grp = entity if entity in _ENT_COLORS else "unknown"
            color = _ENT_COLORS.get(grp, "#888888")
            ax.scatter(times, [y] * len(times), s=40, alpha=0.75,
                       color=color, marker=cam_mk, zorder=3,
                       label=f"{cam}/{entity}" if ci == 0 else "_")
            total_events += len(times)

    ax.set_yticks(list(y_idx.values()))
    ax.set_yticklabels(list(y_idx.keys()), fontsize=8)
    ax.set_ylim(0, n + 1)
    ax.set_xlabel("время от старта, с")
    ax.set_title(f"Детекции: {total_events} событий, {n} уникальных объектов")
    ax.grid(True, linestyle="--", alpha=0.35)
    if len(cam_list) > 1:
        import matplotlib.lines as _mlines
        handles = [_mlines.Line2D([], [], color="gray", marker=cam_markers[i % len(cam_markers)],
                                   linestyle="None", label=cam)
                   for i, cam in enumerate(cam_list)]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    path = out_dir / "person_timeline.png"
    plt.savefig(str(path), dpi=120)
    plt.close()
    print(f"  person_timeline.png → {path}")


def _regen_charts(run_dir: Path) -> None:
    """Перечитывает CSV из существующего run-каталога и перегенерирует графики."""
    import csv as _csv

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
    dets_log  = [[_f(r[0]), r[1], r[2], _i(r[3]), _i(r[4]), _i(r[5]), _i(r[6]),
                  _f(r[7]), r[8], _f(r[9]), r[10] if len(r) > 10 else "", _f(r[11] if len(r) > 11 else "0")]
                 for r in _load("detections.csv") if len(r) >= 10]

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
    _save_person_timeline(dets_log, run_dir)
    osd_log = _regen_osd_from_images(run_dir)
    if osd_log:
        _save_osd_chart(osd_log, run_dir)



# ─── ML-пайплайн ──────────────────────────────────────────────────────────────

def _load_ml_pipeline(config_path: Path, embeddings_path: Path) -> "object | None":
    """Загружает MLPipeline из config.yaml + embeddings.json. Возвращает None при ошибке."""
    if not config_path.is_file():
        return None

    try:
        import yaml
    except ImportError:
        print("  [ML] Нужен pyyaml: pip install pyyaml — ML-пайплайн недоступен", file=sys.stderr)
        return None

    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [ML] Ошибка чтения {config_path}: {e}", file=sys.stderr)
        return None

    try:
        from ml.pipeline import MLPipeline
        pipeline = MLPipeline.from_config(cfg)
    except Exception as e:
        print(f"  [ML] Ошибка инициализации MLPipeline: {e}", file=sys.stderr)
        return None

    if embeddings_path.is_file():
        try:
            embeddings = _json.loads(embeddings_path.read_text(encoding="utf-8"))
            pipeline.load_person_embeddings(embeddings)
            print(f"  [ML] Эмбеддинги: {pipeline.identifier.person_count} жителей из {embeddings_path.name}")
        except Exception as e:
            print(f"  [ML] Ошибка загрузки эмбеддингов: {e}", file=sys.stderr)

    return pipeline


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Motion + YOLO + ML-пайплайн: классификация группы и идентификация жителей"
    )
    parser.add_argument("--env",         type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model",       type=Path, default=DEFAULT_MODEL,
                        help="Путь к yolov8n.onnx")
    parser.add_argument("--config",      type=Path, default=DEFAULT_CONFIG,
                        help="config.yaml с путями к ML-моделям")
    parser.add_argument("--embeddings",  type=Path, default=DEFAULT_EMBEDDINGS,
                        help="JSON-файл с эмбеддингами жителей")
    parser.add_argument("--no-ml",       action="store_true",
                        help="Не загружать ML-пайплайн (только YOLO)")
    parser.add_argument("--conf",        type=float, default=None,
                        help="YOLO conf threshold (env: YOLO_CONF, default: 0.35)")
    parser.add_argument("--nms",         type=float, default=None,
                        help="YOLO NMS threshold (env: YOLO_NMS, default: 0.45)")
    parser.add_argument("--crop-pad",    type=float, default=0.10,
                        help="Отступ вокруг bbox при вырезке кропа (default: 0.10)")
    parser.add_argument("--threshold",   type=float, default=None,
                        help="motion diff threshold (env: MOTION_DIFF_THRESHOLD, default: 10)")
    parser.add_argument("--output",      type=Path, default=None)
    parser.add_argument("--tcp",         action="store_true", help="RTSP через TCP")
    parser.add_argument("--crop-rel",    type=str, default=None, metavar="X,Y,W,H")
    parser.add_argument("--open-timeout-ms",   type=int, default=10000)
    parser.add_argument("--read-timeout-ms",   type=int, default=10000)
    parser.add_argument("--stimeout-us",       type=int, default=3_000_000)
    parser.add_argument("--min-laplacian-var", type=float, default=12.0)
    parser.add_argument("--min-gray-std",      type=float, default=2.5)
    parser.add_argument("--save-extra-reads",  type=int, default=12)
    parser.add_argument("--baseline-attempts", type=int, default=16)
    parser.add_argument("--heartbeat-sec",     type=float, default=None,
                        help="Раз в N сек сохранять кадр без движения; 0 = выкл; "
                             "иначе MOTION_HEARTBEAT_SEC из .env или 600")
    parser.add_argument("--yolo-max-fps",      type=float, default=None, metavar="FPS",
                        help="Макс. частота YOLO на одну камеру (env: YOLO_MAX_FPS, default: 2.0)")
    parser.add_argument("--save-raw",    action="store_true",
                        help="Сохранять сырой кадр при каждом срабатывании порога движения")
    parser.add_argument("--cam-ts",      action="store_true",
                        help="Читать время камеры из OSD-оверлея и добавлять в имя файла")
    parser.add_argument("--duration",    type=float, default=0, metavar="SEC",
                        help="Остановиться через N секунд. 0 = бесконечно (default).")
    parser.add_argument("--cpu-interval", type=float, default=2.0, metavar="SEC",
                        help="Интервал замера ЦПУ (сек). 0 = не замерять.")
    parser.add_argument("--regen-from",  type=Path, default=None, metavar="RUN_DIR",
                        help="Перегенерировать графики из существующего каталога прогона.")
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
        raw_t     = (os.environ.get("MOTION_DIFF_THRESHOLD") or "").strip()
        threshold = float(raw_t) if raw_t else 10.0

    if args.heartbeat_sec is not None:
        heartbeat_sec = max(0.0, float(args.heartbeat_sec))
    else:
        raw_hb        = (os.environ.get("MOTION_HEARTBEAT_SEC") or "").strip()
        heartbeat_sec = max(0.0, float(raw_hb)) if raw_hb else 600.0

    if args.conf is None:
        raw_conf  = (os.environ.get("YOLO_CONF") or "").strip()
        args.conf = float(raw_conf) if raw_conf else 0.35

    if args.nms is None:
        raw_nms  = (os.environ.get("YOLO_NMS") or "").strip()
        args.nms = float(raw_nms) if raw_nms else 0.45

    if args.yolo_max_fps is None:
        raw_mfps        = (os.environ.get("YOLO_MAX_FPS") or "").strip()
        args.yolo_max_fps = float(raw_mfps) if raw_mfps else 2.0

    out_dir = args.output
    if out_dir is None:
        out_dir = DEFAULT_OUTPUT / f"run_{ts_for_dir()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    _log_file = open(out_dir / "run.log", "w", encoding="utf-8", errors="replace", buffering=1)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, _log_file)
    sys.stderr = _Tee(_orig_stderr, _log_file)

    # ML-пайплайн (классификация + идентификация)
    mlpipeline = None
    if not args.no_ml:
        mlpipeline = _load_ml_pipeline(args.config, args.embeddings)
        if mlpipeline is not None:
            cl_ready = mlpipeline.classifier.ready
            id_ready = mlpipeline.identifier.ready
            print(f"  [ML] Классификатор: {'готов' if cl_ready else 'не загружен'}")
            print(f"  [ML] Идентификатор: {'готов' if id_ready else 'не загружен'}")
        else:
            print("  [ML] Пайплайн не загружен — только YOLO-детекция")
    else:
        print("  [ML] --no-ml: только YOLO-детекция")

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_capture_options(
        use_tcp=args.tcp, stimeout_us=args.stimeout_us
    )

    cameras = collect_cam_urls()
    active  = [(k, v) for k, v in cameras if not skip_url(v)]
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

    low_unique = sorted(set(url for _, url in active))
    readers: dict[str, _CapReader] = {}
    for url in low_unique:
        cap = open_cap(url, open_timeout_ms=args.open_timeout_ms,
                       read_timeout_ms=args.read_timeout_ms)
        if cap is None:
            print(f"  [!] LOW не удалось открыть: {redact_url(url)[:80]}…", file=sys.stderr)
        else:
            readers[url] = _CapReader(url, cap, args.open_timeout_ms, args.read_timeout_ms)

    opened_vars: list[tuple[str, str]] = [
        (vn, url) for vn, url in active if url in readers
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

    cam_dirs: dict[str, Path] = {}
    for vn, _ in opened_vars:
        cam_dir = images_dir / stem_from_var(vn)
        cam_dir.mkdir(exist_ok=True)
        (cam_dir / "diff").mkdir(exist_ok=True)
        cam_dirs[vn] = cam_dir

    prev_gray:       dict[str, "np.ndarray | None"] = {vn: None for vn, _ in opened_vars}
    last_good_frame: dict[str, "np.ndarray | None"] = {vn: None for vn, _ in opened_vars}
    last_heartbeat:  dict[str, float]               = {vn: time.monotonic() for vn, _ in opened_vars}
    last_yolo_t:     dict[str, float]               = {vn: 0.0 for vn, _ in opened_vars}
    prev_pts:        dict[str, float]               = {lu: -1.0 for lu in low_unique}
    _MAX_LOW_IMPLAUSIBLE = 40
    _low_implausible: dict[str, int] = defaultdict(int)

    frame_log:      list[list] = []  # [mono_s, ts_msk, url_id, ok, plausible, event]
    saves_log:      list[list] = []  # [mono_s, ts_msk, cam, save_type]
    diffs_log:      list[list] = []  # [mono_s, ts_msk, cam, diff]
    pts_log:        list[list] = []  # [mono_s, ts_msk, url_id, pts_ms]
    cpu_log:        list[list] = []
    detections_log: list[list] = []  # [mono_s, ts_msk, cam, x1, y1, x2, y2, det_conf, group, grp_conf, person_id, id_conf]

    from common.utils.time_msk import ts_iso as _ts_iso
    (out_dir / "run_params.json").write_text(_json.dumps({
        "started_at_msk":  _ts_iso(),
        "script":          "6_identify_people.py",
        "threshold":       threshold,
        "conf":            args.conf,
        "nms":             args.nms,
        "yolo_model":      str(args.model),
        "ml_config":       str(args.config) if not args.no_ml else None,
        "ml_embeddings":   str(args.embeddings) if not args.no_ml else None,
        "ml_active":       mlpipeline is not None,
        "tcp":             args.tcp,
        "cameras":         [vn for vn, _ in opened_vars],
        "crop_global":     list(global_crop) if global_crop else None,
        "heartbeat_sec":   heartbeat_sec,
        "yolo_max_fps":    args.yolo_max_fps,
        "duration_sec":    args.duration,
        "cpu_interval":    args.cpu_interval,
        "output":          str(out_dir),
        "stream":          "LOW-only",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    ml_desc = ("классификатор+идентификатор" if mlpipeline is not None else "только YOLO")
    print(f"Модель:    {args.model}")
    print(f"ML:        {ml_desc}")
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
    cpu_active  = args.cpu_interval > 0 and cpu_monitor.start(t_start)
    if args.cpu_interval > 0 and not cpu_active:
        print("  [!] psutil не установлен — мониторинг ЦПУ недоступен (pip install psutil)")

    print("Базовый кадр…")
    for low_u, var_list in vars_by_low.items():
        reader_bl = readers[low_u]
        frame_l   = None
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
            prev_gray[vn]      = prepare_gray(frame_u)
            last_good_frame[vn] = frame_u
            stem  = stem_from_var(vn)
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

                if _pts > 0 and _pts == prev_pts[low_u]:
                    continue
                prev_pts[low_u] = _pts

                _plausible = frame_decode_plausible(
                    frame_l, min_laplacian_var=args.min_laplacian_var,
                    min_gray_std=args.min_gray_std)
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
                    frame_u = apply_crop_optional(frame_l, crop_by_cam[vn])
                    gray    = prepare_gray(frame_u)
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

                    # Сохраняем кадр при каждом превышении порога
                    stem      = stem_from_var(vn)
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

                    ts = _ts(cam_dt)
                    _now_ml = time.monotonic()

                    if mlpipeline is not None:
                        ml_results = mlpipeline.run(frame_u, detections)
                        annotated  = _draw_boxes_ml(frame_u, ml_results)
                        for r in ml_results:
                            detections_log.append([
                                round(_now_ml - t_start, 4), _ts_str, vn,
                                r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3],
                                round(r.detect_conf, 3),
                                r.group_class, round(r.group_conf, 3),
                                r.person_id or "", round(r.identify_conf, 3),
                            ])
                        id_str = ",".join(
                            r.person_id or r.group_class for r in ml_results
                        )
                    else:
                        annotated = draw_boxes(frame_u, detections)
                        id_str    = str(len(detections))

                    fname = f"{stem}_{ts}_p{len(detections)}.jpg"
                    cv2.imwrite(str(cam_dirs[vn] / fname), annotated)
                    frame_log[-1][5] = "yolo"
                    saves_log.append([round(time.monotonic() - t_start, 4), _ts_str, vn, "yolo"])

                    # Вырезаем кропы
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
                            crop_name = f"{stem}_{ts}_p{idx}of{len(detections)}_conf{conf:.2f}.jpg"
                            cv2.imwrite(str(crops_dir / crop_name), frame_u[y1c:y2c, x1c:x2c])

                    print(f"  {fname}  diff={diff:.2f}  [{id_str}]")

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
                    stem    = stem_from_var(vn)
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
            with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "url_id", "ok", "plausible", "event"])
                writer.writerows(frame_log)
            print(f"  frames.csv: {len(frame_log)} строк")

        if diffs_log:
            with open(out_dir / "diffs.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cam", "diff"])
                writer.writerows(diffs_log)
            print(f"  diffs.csv:  {len(diffs_log)} записей")

        if saves_log:
            with open(out_dir / "saves.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cam", "type"])
                writer.writerows(saves_log)
            print(f"  saves.csv:  {len(saves_log)} записей")

        if cpu_log:
            with open(out_dir / "cpu.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cpu_pct"])
                writer.writerows(cpu_log)
            print(f"  cpu.csv:    {len(cpu_log)} замеров")

        if pts_log:
            with open(out_dir / "pts.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "url_id", "pts_ms"])
                writer.writerows(pts_log)
            print(f"  pts.csv:    {len(pts_log)} записей")

        if detections_log:
            with open(out_dir / "detections.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cam",
                                  "x1", "y1", "x2", "y2", "detect_conf",
                                  "group", "group_conf", "person_id", "identify_conf"])
                writer.writerows(detections_log)
            print(f"  detections.csv: {len(detections_log)} детекций")

        _save_charts(frame_log, cpu_log, saves_log, diffs_log, threshold, out_dir)
        _save_pts_chart(pts_log, cpu_log, out_dir)
        _save_run_stats(frame_log, pts_log, saves_log, diffs_log, out_dir)
        _save_person_timeline(detections_log, out_dir)

        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _log_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
