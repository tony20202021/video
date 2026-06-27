"""
Замер реальной FPS-пропускной способности компа по компонентам пайплайна.

Измеряет:
  1. LOW VideoCapture: частота успешных cap.read()
  2. HI StreamReader: частота обновления _good_frame (реальный FPS камеры на этой машине)
  3. YOLO inference: время на кадр при разных разрешениях
  4. Полный пайплайн: motion detect → YOLO → FPS ограничение

Выводит JSON с результатами и рекомендуемое ограничение FPS.

Использование:
  python scripts/bench/fps_bench.py --duration 30
  python scripts/bench/fps_bench.py --duration 30 --tcp --report-json fps_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import threading

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.utils.cam_urls import collect_cam_urls, resolve_hi_rtsp_url
from common.utils.motion_utils import (
    StreamReader,
    ffmpeg_capture_options,
    frame_decode_plausible,
    mean_abs_diff,
    open_cap,
    prepare_gray,
    skip_url,
)

class CpuMonitor:
    """Собирает cpu_percent() в фоновом потоке во время замера."""
    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ok = False

    def start(self) -> None:
        try:
            import psutil as _p
            _p.cpu_percent()  # первый вызов — 0.0, сбрасываем
            self._psutil = _p
            self._ok = True
        except ImportError:
            return
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self._psutil.cpu_percent(interval=self._interval))

    def stop(self) -> dict:
        self._stop.set()
        if self._ok:
            self._thread.join(timeout=3)
        if not self._samples:
            return {}
        arr = self._samples
        return {
            "cpu_mean_pct": round(float(sum(arr)) / len(arr), 1),
            "cpu_max_pct": round(max(arr), 1),
            "cpu_samples": len(arr),
        }


DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_MODEL = REPO_ROOT / ".models" / "yolov8n.onnx"
DEFAULT_OUTPUT = REPO_ROOT / ".output" / "bench" / "1_fps"


def bench_low_stream(url: str, duration: float, *, open_timeout_ms: int, read_timeout_ms: int) -> dict:
    """Замер FPS LOW-потока (cap.read())."""
    cap = open_cap(url, open_timeout_ms=open_timeout_ms, read_timeout_ms=read_timeout_ms)
    if cap is None:
        return {"error": "не удалось открыть"}

    ok_count = 0
    fail_count = 0
    t_start = time.monotonic()
    t_end = t_start + duration
    frame_sizes: list[tuple[int, int]] = []

    while time.monotonic() < t_end:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            ok_count += 1
            if len(frame_sizes) < 3:
                frame_sizes.append((frame.shape[1], frame.shape[0]))
        else:
            fail_count += 1

    elapsed = time.monotonic() - t_start
    cap.release()

    fps = ok_count / elapsed if elapsed > 0 else 0
    return {
        "fps_raw": round(fps, 2),
        "ok_frames": ok_count,
        "fail_frames": fail_count,
        "duration_s": round(elapsed, 2),
        "frame_size": frame_sizes[0] if frame_sizes else None,
    }


def bench_low_with_diff(
    url: str, duration: float,
    *,
    open_timeout_ms: int,
    read_timeout_ms: int,
    compare_width: int = 320,
    min_laplacian_var: float = 12.0,
    min_gray_std: float = 2.5,
) -> dict:
    """LOW: cap.read() + frame_decode_plausible + prepare_gray + mean_abs_diff."""
    cap = open_cap(url, open_timeout_ms=open_timeout_ms, read_timeout_ms=read_timeout_ms)
    if cap is None:
        return {"error": "не удалось открыть"}

    ok_count = 0
    plausible_count = 0
    diff_count = 0
    prev_gray = None
    t_start = time.monotonic()
    t_end = t_start + duration

    while time.monotonic() < t_end:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            continue
        ok_count += 1
        if not frame_decode_plausible(frame, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std):
            continue
        plausible_count += 1
        gray = prepare_gray(frame, compare_width)
        if prev_gray is not None:
            mean_abs_diff(prev_gray, gray)
            diff_count += 1
        prev_gray = gray

    elapsed = time.monotonic() - t_start
    cap.release()
    return {
        "fps_raw": round(ok_count / elapsed, 2),
        "fps_plausible": round(plausible_count / elapsed, 2),
        "fps_diff": round(diff_count / elapsed, 2),
        "ok_frames": ok_count,
        "plausible_frames": plausible_count,
        "diff_frames": diff_count,
        "duration_s": round(elapsed, 2),
    }




def bench_hi_stream(url: str, duration: float, *, scale: float,
                    open_timeout_ms: int, read_timeout_ms: int) -> dict:
    """Замер эффективного FPS HI-потока через StreamReader."""
    reader = StreamReader(
        url,
        open_timeout_ms=open_timeout_ms,
        read_timeout_ms=read_timeout_ms,
        scale=scale,
    )
    reader.start()
    time.sleep(2.0)  # даём время инициализации

    updates = 0
    last_ts = 0.0
    t_start = time.monotonic()
    t_end = t_start + duration
    frame_sizes: list[tuple[int, int]] = []

    while time.monotonic() < t_end:
        frame, ts = reader.get_latest()
        if ts > last_ts:
            updates += 1
            last_ts = ts
            if frame is not None and len(frame_sizes) < 3:
                frame_sizes.append((frame.shape[1], frame.shape[0]))
        time.sleep(0.01)  # 100Hz poll

    elapsed = time.monotonic() - t_start
    reader.stop()

    fps = updates / elapsed if elapsed > 0 else 0
    return {
        "fps_effective": round(fps, 2),
        "frame_updates": updates,
        "duration_s": round(elapsed, 2),
        "frame_size": frame_sizes[0] if frame_sizes else None,
        "scale": scale,
    }


def bench_yolo(model_path: Path, frame_sizes: list[tuple[int, int]],
               n_iters: int = 50, n_warmup: int = 10) -> dict:
    """Замер времени YOLO inference на разных разрешениях."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime не установлен"}

    if not model_path.is_file():
        return {"error": f"модель не найдена: {model_path}"}

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    results = {}

    for w, h in frame_sizes:
        # Прогрев — одинаковое количество для каждого варианта
        dummy = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        blob = cv2.resize(dummy, (640, 640))[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        for _ in range(n_warmup):
            sess.run(None, {input_name: blob})

        # Замер
        times = []
        for _ in range(n_iters):
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            blob_f = cv2.resize(frame, (640, 640))[:, :, ::-1].astype(np.float32) / 255.0
            blob_f = blob_f.transpose(2, 0, 1)[np.newaxis]
            sess.run(None, {input_name: blob_f})
            times.append(time.perf_counter() - t0)

        arr = np.array(times) * 1000  # → ms
        results[f"{w}x{h}"] = {
            "mean_ms": round(float(arr.mean()), 1),
            "p50_ms": round(float(np.percentile(arr, 50)), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1),
            "max_fps": round(1000.0 / float(arr.mean()), 1),
        }

    return results


def _lo_hi_open(low_url: str, hi_url: str, open_timeout_ms: int, read_timeout_ms: int):
    cap_lo = open_cap(low_url, open_timeout_ms=open_timeout_ms, read_timeout_ms=read_timeout_ms)
    cap_hi = open_cap(hi_url, open_timeout_ms=open_timeout_ms, read_timeout_ms=read_timeout_ms)
    return cap_lo, cap_hi


def _lo_stats(ok_frames: int, plausible: int, triggers: int, elapsed: float,
              loop_times: list) -> dict:
    loop_arr = np.array(loop_times) * 1000 if loop_times else np.array([0.0])
    return {
        "duration_s": round(elapsed, 2),
        "low_fps_raw": round(ok_frames / elapsed, 2),
        "low_fps_plausible": round(plausible / elapsed, 2),
        "low_fps_diff": round(triggers / elapsed, 2),
        "low_ok": ok_frames,
        "low_plausible": plausible,
        "motion_triggers": triggers,
        "trigger_rate_pct": round(100 * triggers / max(plausible - 1, 1), 1),
        "loop_mean_ms": round(float(loop_arr.mean()), 1),
    }


def bench_pipeline_grab_only(
    low_url: str, hi_url: str, duration: float,
    threshold: float, compare_width: int,
    *, open_timeout_ms: int, read_timeout_ms: int,
    min_laplacian_var: float = 12.0, min_gray_std: float = 2.5,
) -> dict:
    """LOW (read+plausible+diff) + HI только grab (без декодирования)."""
    cap_lo, cap_hi = _lo_hi_open(low_url, hi_url, open_timeout_ms, read_timeout_ms)
    if cap_lo is None:
        return {"error": "LOW не открылся"}
    if cap_hi is None:
        return {"error": "HI не открылся"}

    ok_frames = plausible = triggers = hi_grabs = 0
    loop_times: list[float] = []
    prev_gray = None
    t_start = time.monotonic()
    t_end = t_start + duration

    while time.monotonic() < t_end:
        t0 = time.perf_counter()
        cap_hi.grab()
        hi_grabs += 1

        ok, frame = cap_lo.read()
        if not ok or frame is None or frame.size == 0:
            loop_times.append(time.perf_counter() - t0)
            continue
        ok_frames += 1
        if not frame_decode_plausible(frame, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std):
            loop_times.append(time.perf_counter() - t0)
            continue
        plausible += 1
        gray = prepare_gray(frame, compare_width)
        if prev_gray is not None:
            diff = mean_abs_diff(prev_gray, gray)
            if diff > threshold:
                triggers += 1
        prev_gray = gray
        loop_times.append(time.perf_counter() - t0)

    elapsed = time.monotonic() - t_start
    cap_lo.release()
    cap_hi.release()
    r = _lo_stats(ok_frames, plausible, triggers, elapsed, loop_times)
    r["hi_grabs"] = hi_grabs
    r["hi_grabs_per_s"] = round(hi_grabs / elapsed, 2)
    return r


def bench_pipeline_selective_decode(
    low_url: str, hi_url: str, duration: float,
    threshold: float, compare_width: int,
    *, open_timeout_ms: int, read_timeout_ms: int,
    min_laplacian_var: float = 12.0, min_gray_std: float = 2.5,
    hi_decode_max_fps: float = 0.0,
) -> dict:
    """LOW (read+plausible+diff) + HI grab+retrieve+plausible.

    hi_decode_max_fps > 0: retrieve() вызывается не чаще этого числа раз/сек.
    """
    cap_lo, cap_hi = _lo_hi_open(low_url, hi_url, open_timeout_ms, read_timeout_ms)
    if cap_lo is None:
        return {"error": "LOW не открылся"}
    if cap_hi is None:
        return {"error": "HI не открылся"}

    hi_decode_interval = 1.0 / hi_decode_max_fps if hi_decode_max_fps > 0 else 0.0

    ok_frames = plausible = triggers = 0
    hi_decodes = hi_plausible = hi_decode_skipped = 0
    hi_decode_times: list[float] = []
    loop_times: list[float] = []
    prev_gray = None
    last_hi_decode_t = 0.0
    t_start = time.monotonic()
    t_end = t_start + duration

    while time.monotonic() < t_end:
        t0 = time.perf_counter()
        cap_hi.grab()

        ok, frame = cap_lo.read()
        if not ok or frame is None or frame.size == 0:
            loop_times.append(time.perf_counter() - t0)
            continue
        ok_frames += 1
        if not frame_decode_plausible(frame, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std):
            loop_times.append(time.perf_counter() - t0)
            continue
        plausible += 1
        gray = prepare_gray(frame, compare_width)
        if prev_gray is not None:
            diff = mean_abs_diff(prev_gray, gray)
            if diff > threshold:
                triggers += 1
        prev_gray = gray

        now_t = time.perf_counter()
        if hi_decode_interval == 0 or now_t - last_hi_decode_t >= hi_decode_interval:
            t_dec = time.perf_counter()
            ok_hi, hi_frame = cap_hi.retrieve()
            hi_decode_times.append(time.perf_counter() - t_dec)
            hi_decodes += 1
            last_hi_decode_t = now_t
            if ok_hi and hi_frame is not None and hi_frame.size > 0:
                if frame_decode_plausible(hi_frame, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std):
                    hi_plausible += 1
        else:
            hi_decode_skipped += 1

        loop_times.append(time.perf_counter() - t0)

    elapsed = time.monotonic() - t_start
    cap_lo.release()
    cap_hi.release()
    dec_arr = np.array(hi_decode_times) * 1000 if hi_decode_times else np.array([0.0])
    r = _lo_stats(ok_frames, plausible, triggers, elapsed, loop_times)
    r.update({
        "hi_decodes": hi_decodes,
        "hi_plausible": hi_plausible,
        "hi_decode_skipped": hi_decode_skipped,
        "hi_decode_max_fps": hi_decode_max_fps,
        "hi_decode_mean_ms": round(float(dec_arr.mean()), 1),
        "hi_decode_p95_ms": round(float(np.percentile(dec_arr, 95)) if len(dec_arr) > 1 else 0, 1),
    })
    return r


def bench_pipeline_full_yolo(
    low_url: str, hi_url: str, model_path: Path, duration: float,
    threshold: float, compare_width: int,
    *, open_timeout_ms: int, read_timeout_ms: int,
    min_laplacian_var: float = 12.0, min_gray_std: float = 2.5,
    hi_decode_max_fps: float = 0.0,
    cooldown_s: float = 0.0,
) -> dict:
    """LOW (read+plausible+diff) + HI grab+retrieve+plausible + YOLO.

    hi_decode_max_fps > 0: retrieve() не чаще этого числа раз/сек.
    cooldown_s > 0: YOLO не чаще раз в cooldown_s секунд.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime не установлен"}
    if not model_path.is_file():
        return {"error": f"нет модели: {model_path}"}

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    cap_lo, cap_hi = _lo_hi_open(low_url, hi_url, open_timeout_ms, read_timeout_ms)
    if cap_lo is None:
        return {"error": "LOW не открылся"}
    if cap_hi is None:
        return {"error": "HI не открылся"}

    hi_decode_interval = 1.0 / hi_decode_max_fps if hi_decode_max_fps > 0 else 0.0

    ok_frames = plausible = triggers = 0
    hi_decodes = hi_plausible = hi_decode_skipped = 0
    yolo_runs = yolo_skipped = detections_total = 0
    yolo_times: list[float] = []
    loop_times: list[float] = []
    prev_gray = None
    last_yolo_t = 0.0
    last_hi_decode_t = 0.0
    t_start = time.monotonic()
    t_end = t_start + duration

    while time.monotonic() < t_end:
        t0 = time.perf_counter()
        cap_hi.grab()

        ok, frame = cap_lo.read()
        if not ok or frame is None or frame.size == 0:
            loop_times.append(time.perf_counter() - t0)
            continue
        ok_frames += 1
        if not frame_decode_plausible(frame, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std):
            loop_times.append(time.perf_counter() - t0)
            continue
        plausible += 1
        gray = prepare_gray(frame, compare_width)
        if prev_gray is not None:
            diff = mean_abs_diff(prev_gray, gray)
            if diff > threshold:
                triggers += 1
        prev_gray = gray

        # HI decode с ограничением частоты; YOLO с cooldown
        now_t = time.perf_counter()
        if hi_decode_interval == 0 or now_t - last_hi_decode_t >= hi_decode_interval:
            ok_hi, hi_frame = cap_hi.retrieve()
            hi_decodes += 1
            last_hi_decode_t = now_t
            if ok_hi and hi_frame is not None and hi_frame.size > 0:
                if frame_decode_plausible(hi_frame, min_laplacian_var=min_laplacian_var, min_gray_std=min_gray_std):
                    hi_plausible += 1
                    now_t2 = time.perf_counter()
                    if cooldown_s > 0 and now_t2 - last_yolo_t < cooldown_s:
                        yolo_skipped += 1
                    else:
                        last_yolo_t = now_t2
                        t_yolo = time.perf_counter()
                        blob = cv2.resize(hi_frame, (640, 640))[:, :, ::-1].astype(np.float32) / 255.0
                        blob = blob.transpose(2, 0, 1)[np.newaxis]
                        out = sess.run(None, {input_name: blob})[0]
                        yolo_times.append(time.perf_counter() - t_yolo)
                        yolo_runs += 1
                        preds = out[0].T
                        detections_total += int((preds[:, 4] >= 0.35).sum())
        else:
            hi_decode_skipped += 1

        loop_times.append(time.perf_counter() - t0)

    elapsed = time.monotonic() - t_start
    cap_lo.release()
    cap_hi.release()
    yolo_arr = np.array(yolo_times) * 1000 if yolo_times else np.array([0.0])
    r = _lo_stats(ok_frames, plausible, triggers, elapsed, loop_times)
    r.update({
        "hi_decodes": hi_decodes,
        "hi_plausible": hi_plausible,
        "hi_decode_skipped": hi_decode_skipped,
        "hi_decode_max_fps": hi_decode_max_fps,
        "yolo_runs": yolo_runs,
        "yolo_skipped": yolo_skipped,
        "yolo_max_fps": round(1.0 / cooldown_s, 2) if cooldown_s > 0 else 0.0,
        "yolo_mean_ms": round(float(yolo_arr.mean()), 1),
        "yolo_p95_ms": round(float(np.percentile(yolo_arr, 95)) if len(yolo_arr) > 1 else 0, 1),
        "detections_total": detections_total,
    })
    return r


def main() -> int:
    parser = argparse.ArgumentParser(description="Замер FPS пайплайна видеонаблюдения")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--duration", type=float, default=20.0, help="Длительность каждого теста (сек)")
    parser.add_argument("--tcp", action="store_true")
    parser.add_argument("--hi-scale", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--compare-width", type=int, default=320)
    parser.add_argument("--output", type=Path, default=None,
                        help="Каталог для результатов (default: .output/bench/fps/run_...)")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--min-laplacian-var", type=float, default=12.0)
    parser.add_argument("--min-gray-std", type=float, default=2.5)
    parser.add_argument("--yolo-iters", type=int, default=50, help="Число измерительных итераций YOLO")
    parser.add_argument("--yolo-warmup", type=int, default=10, help="Прогрев: число итераций перед замером (для каждого варианта)")
    parser.add_argument("--yolo-max-fps", type=float, default=0.0,
                        help="Шаг [9/9]: макс. частота YOLO (fps). 0=без ограничений, напр. 0.33=раз в 3с")
    parser.add_argument("--hi-decode-max-fps", type=float, default=0.0,
                        help="Шаг [9/9]: макс. частота декодирования HI (fps). 0=без ограничений, напр. 1.0=раз в 1с")
    parser.add_argument("--step-pause-s", type=float, default=15.0,
                        help="Пауза между шагами (сек) — CPU остывает, фоновые процессы успокаиваются")
    parser.add_argument("--skip-pipeline", action="store_true", help="Пропустить тест полного пайплайна")
    args = parser.parse_args()

    if not args.env.is_file():
        print(f"Нет .env: {args.env}", file=sys.stderr)
        return 1

    from dotenv import load_dotenv
    load_dotenv(args.env, override=True)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_capture_options(
        use_tcp=args.tcp, stimeout_us=3_000_000
    )

    cameras = collect_cam_urls()
    active = [(k, v) for k, v in cameras if not skip_url(v)]
    if not active:
        print("Нет камер", file=sys.stderr)
        return 1

    # Берём первую камеру
    vn, low_url = active[0]
    hi_url = resolve_hi_rtsp_url(vn, low_url=low_url, skip_url=skip_url)
    to_ms = dict(open_timeout_ms=10000, read_timeout_ms=10000)

    report: dict = {
        "camera": vn,
        "duration_per_bench": args.duration,
        "yolo_iters": args.yolo_iters,
        "yolo_warmup": args.yolo_warmup,
        "step_pause_s": args.step_pause_s,
        "step9_yolo_max_fps": args.yolo_max_fps,
        "step9_hi_decode_max_fps": args.hi_decode_max_fps,
    }

    print(f"=== FPS Benchmark: {vn} ===\n")

    filt = dict(min_laplacian_var=args.min_laplacian_var, min_gray_std=args.min_gray_std)

    diff_kw = dict(compare_width=args.compare_width, **to_ms, **filt)
    pipe_kw = dict(threshold=args.threshold, compare_width=args.compare_width, **to_ms, **filt)

    def _pause() -> None:
        if args.step_pause_s > 0:
            print(f"  ... пауза {args.step_pause_s:.0f}s (CPU остывает)")
            time.sleep(args.step_pause_s)

    def _step(label: str, fn, *args, **kwargs) -> dict:
        mon = CpuMonitor()
        mon.start()
        r = fn(*args, **kwargs)
        cpu = mon.stop()
        r.update(cpu)
        print(f"      cpu_mean={cpu.get('cpu_mean_pct', '?')}%  cpu_max={cpu.get('cpu_max_pct', '?')}%")
        return r

    yolo_interval_s = 1.0 / args.yolo_max_fps if args.yolo_max_fps > 0 else 0.0
    hi_interval_s   = 1.0 / args.hi_decode_max_fps if args.hi_decode_max_fps > 0 else 0.0

    print(f"[1/9] LOW raw cap.read() ({args.duration}s)...")
    report["low"] = _step("[1/9]", bench_low_stream, low_url, args.duration, **to_ms)
    print(f"      fps_raw={report['low'].get('fps_raw')}  size={report['low'].get('frame_size')}")
    _pause()

    print(f"[2/9] LOW + plausible + diff ({args.duration}s)...")
    report["low_diff"] = _step("[2/9]", bench_low_with_diff, low_url, args.duration, **diff_kw)
    r = report["low_diff"]
    print(f"      fps_raw={r.get('fps_raw')}  fps_plausible={r.get('fps_plausible')}  fps_diff={r.get('fps_diff')}")
    _pause()

    print(f"[3/9] HI raw cap.read() ({args.duration}s)...")
    report["hi"] = _step("[3/9]", bench_low_stream, hi_url, args.duration, **to_ms)
    print(f"      fps_raw={report['hi'].get('fps_raw')}  size={report['hi'].get('frame_size')}")
    _pause()

    print(f"[4/9] HI + plausible + diff ({args.duration}s)...")
    report["hi_diff"] = _step("[4/9]", bench_low_with_diff, hi_url, args.duration, **diff_kw)
    r = report["hi_diff"]
    print(f"      fps_raw={r.get('fps_raw')}  fps_plausible={r.get('fps_plausible')}  fps_diff={r.get('fps_diff')}")
    _pause()

    yolo_sizes = []
    if report["low"].get("frame_size"):
        yolo_sizes.append(report["low"]["frame_size"])
    if report["hi"].get("frame_size"):
        yolo_sizes.append(report["hi"]["frame_size"])
    if not yolo_sizes:
        yolo_sizes = [(640, 360), (1152, 648)]

    print(f"[5/9] YOLO inference ({args.yolo_warmup} warmup + {args.yolo_iters} iter каждый размер)...")
    report["yolo"] = _step("[5/9]", bench_yolo, args.model, yolo_sizes,
                           n_iters=args.yolo_iters, n_warmup=args.yolo_warmup)
    for size, r in report["yolo"].items():
        if isinstance(r, dict) and "mean_ms" in r:
            print(f"      {size}: mean={r['mean_ms']}ms  p95={r['p95_ms']}ms  max_fps={r['max_fps']}")
    _pause()

    if not args.skip_pipeline:
        print(f"[6/9] Пайплайн A: LOW+diff + HI grab-only ({args.duration}s)...")
        report["pipeline_a"] = _step("[6/9]", bench_pipeline_grab_only,
                                     low_url, hi_url, args.duration, **pipe_kw)
        p = report["pipeline_a"]
        print(f"      low_fps={p.get('low_fps_raw')}  triggers={p.get('motion_triggers')}  "
              f"hi_grabs/s={p.get('hi_grabs_per_s')}  loop={p.get('loop_mean_ms')}ms")
        _pause()

        print(f"[7/9] Пайплайн B: LOW+diff + HI decode без ограничений ({args.duration}s)...")
        report["pipeline_b"] = _step("[7/9]", bench_pipeline_selective_decode,
                                     low_url, hi_url, args.duration, **pipe_kw,
                                     hi_decode_max_fps=0.0)
        p = report["pipeline_b"]
        print(f"      low_fps={p.get('low_fps_raw')}  hi_decodes={p.get('hi_decodes')}  "
              f"hi_decode_mean={p.get('hi_decode_mean_ms')}ms  loop={p.get('loop_mean_ms')}ms")
        _pause()

        print(f"[8/9] Пайплайн C: LOW+diff + HI decode + YOLO без ограничений ({args.duration}s)...")
        report["pipeline_c"] = _step("[8/9]", bench_pipeline_full_yolo,
                                     low_url, hi_url, args.model, args.duration, **pipe_kw,
                                     hi_decode_max_fps=0.0, cooldown_s=0.0)
        p = report["pipeline_c"]
        print(f"      low_fps={p.get('low_fps_raw')}  yolo_runs={p.get('yolo_runs')}  "
              f"yolo_mean={p.get('yolo_mean_ms')}ms  loop={p.get('loop_mean_ms')}ms")
        _pause()

        hi_fps_str   = f"{args.hi_decode_max_fps} fps" if args.hi_decode_max_fps > 0 else "unlim"
        yolo_fps_str = f"{args.yolo_max_fps} fps" if args.yolo_max_fps > 0 else "unlim"
        print(f"[9/9] Пайплайн C+лимиты: HI≤{hi_fps_str}  YOLO≤{yolo_fps_str} ({args.duration}s)...")
        report["pipeline_c_cd"] = _step("[9/9]", bench_pipeline_full_yolo,
                                        low_url, hi_url, args.model, args.duration, **pipe_kw,
                                        hi_decode_max_fps=args.hi_decode_max_fps,
                                        cooldown_s=yolo_interval_s)
        p = report["pipeline_c_cd"]
        print(f"      low_fps={p.get('low_fps_raw')}  yolo_runs={p.get('yolo_runs')}  "
              f"yolo_skipped={p.get('yolo_skipped')}  hi_decode_skipped={p.get('hi_decode_skipped')}  "
              f"loop={p.get('loop_mean_ms')}ms")

    # Автосохранение в .output/bench/1_fps/run_*/
    from common.utils.time_msk import ts_for_dir
    out_dir = args.output or DEFAULT_OUTPUT / f"run_{ts_for_dir()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_json or out_dir / "fps_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {json_path}")

    _save_fps_charts(report, out_dir)
    return 0


def _save_fps_charts(report: dict, out_dir: Path) -> None:
    """Строит и сохраняет PNG-графики по результатам fps_bench."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"FPS Benchmark  —  {report.get('camera', '')}  "
                 f"({report.get('duration_per_bench', '?')}s/тест)", fontsize=11)

    # ── График 1 (top-left): FPS по этапам LOW и HI ──────────────────────────
    ax = axes[0][0]
    lo   = report.get("low", {})
    lo_d = report.get("low_diff", {})
    hi   = report.get("hi", {})
    hi_d = report.get("hi_diff", {})

    lo_sz = lo.get("frame_size", (0, 0))
    hi_sz = hi.get("frame_size", (0, 0))

    stage_labels = [
        f"LOW raw\n{lo_sz[0]}×{lo_sz[1]}",
        "LOW\nplausible",
        "LOW\ndiff",
        f"HI raw\n{hi_sz[0]}×{hi_sz[1]}",
        "HI\nplausible",
        "HI\ndiff",
    ]
    stage_values = [
        lo.get("fps_raw", 0),
        lo_d.get("fps_plausible", 0),
        lo_d.get("fps_diff", 0),
        hi.get("fps_raw", 0),
        hi_d.get("fps_plausible", 0),
        hi_d.get("fps_diff", 0),
    ]
    stage_colors = ["#3399ff", "#66bbff", "#99ddff", "#ff8833", "#ffaa66", "#ffcc99"]

    bars = ax.bar(stage_labels, stage_values, color=stage_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, stage_values):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("FPS")
    ax.set_title("Пропускная способность по этапам")
    ax.set_ylim(0, max((v for v in stage_values if v), default=1) * 1.25)
    ax.axhline(12, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(5.5, 12.2, "12fps", fontsize=7, color="gray", ha="right")

    # ── График 2 (top-right): YOLO inference time ─────────────────────────────
    ax = axes[0][1]
    yolo = report.get("yolo", {})
    res_labels, means, p95s = [], [], []
    for res, d in yolo.items():
        if isinstance(d, dict) and "mean_ms" in d:
            res_labels.append(res)
            means.append(d["mean_ms"])
            p95s.append(d["p95_ms"])
    if res_labels:
        x = list(range(len(res_labels)))
        ax.bar([xi - 0.2 for xi in x], means, 0.38, label="mean", color="#44cc66",
               edgecolor="black", linewidth=0.5)
        ax.bar([xi + 0.2 for xi in x], p95s, 0.38, label="p95", color="#cc4444",
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(res_labels, fontsize=8)
        for xi, (m, p) in enumerate(zip(means, p95s)):
            fps_m = round(1000 / m, 1) if m else 0
            fps_p = round(1000 / p, 1) if p else 0
            ax.text(xi - 0.2, m + 1, f"{m:.0f}ms\n({fps_m} fps)", ha="center", fontsize=7)
            ax.text(xi + 0.2, p + 1, f"{p:.0f}ms\n({fps_p} fps)", ha="center", fontsize=7)
        ax.axhline(83, color="orange", linestyle="--", alpha=0.8, linewidth=1)
        ax.text(len(x) - 0.5, 85, "83ms (12fps)", fontsize=7, color="orange")
        ax.set_ylabel("мс")
        ax.set_title("YOLO inference (мс/кадр)")
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)

    # ── График 3 (bottom-left): loop_mean_ms пайплайнов A / B / C / C+cd ───
    ax = axes[1][0]
    pa   = report.get("pipeline_a", {})
    pb   = report.get("pipeline_b", {})
    pc   = report.get("pipeline_c", {})
    pc_cd = report.get("pipeline_c_cd", {})
    pipe_entries = [
        ("A\nHI grab", pa,   "#5599ff"),
        ("B\nHI decode", pb, "#ff9944"),
        ("C\nYOLO", pc,      "#cc44cc"),
    ]
    if pc_cd and not pc_cd.get("error"):
        cd_fps = pc_cd.get("yolo_max_fps", "?")
        pipe_entries.append((f"C+cd\n{cd_fps}fps", pc_cd, "#44aa88"))
    pipe_labels, pipe_vals, pipe_fps, pipe_colors = [], [], [], []
    for label, p, color in pipe_entries:
        lm = p.get("loop_mean_ms")
        if lm:
            pipe_labels.append(label)
            pipe_vals.append(lm)
            pipe_fps.append(round(1000 / lm, 1))
            pipe_colors.append(color)
    if pipe_labels:
        bars = ax.bar(pipe_labels, pipe_vals, color=pipe_colors, edgecolor="black", linewidth=0.5)
        for bar, val, fps in zip(bars, pipe_vals, pipe_fps):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f"{val:.0f}ms\n({fps} fps)", ha="center", va="bottom", fontsize=8)
        ax.axhline(83, color="orange", linestyle="--", alpha=0.8, linewidth=1)
        ax.text(len(pipe_labels) - 0.5, 85, "83ms (12fps)", fontsize=7,
                color="orange", ha="right")
        ax.set_ylabel("мс / итерацию")
        ax.set_title("Pipeline: средняя задержка итерации")
    else:
        ax.set_visible(False)

    # ── График 4 (bottom-center): разбивка loop Pipeline C ───────────────────
    ax = axes[1][1]
    if pa and pb and pc and not pa.get("error") and not pc.get("error"):
        lo_base   = pa.get("loop_mean_ms", 0)
        hi_cost   = max(0, pb.get("loop_mean_ms", 0) - lo_base)
        yolo_cost = max(0, pc.get("loop_mean_ms", 0) - pb.get("loop_mean_ms", 0))

        cats = ["LOW+grab\n(baseline)", "+ HI decode\n+plausible", "+ YOLO"]
        bottoms = [0, lo_base, lo_base + hi_cost]
        heights = [lo_base, hi_cost, yolo_cost]
        colors  = ["#5599ff", "#ff9944", "#cc44cc"]
        labels_leg = [
            f"LOW+grab: {lo_base:.0f}ms",
            f"HI decode: {hi_cost:.0f}ms",
            f"YOLO: {yolo_cost:.0f}ms",
        ]
        for i, (cat, bot, h, col, lbl) in enumerate(
                zip(cats, bottoms, heights, colors, labels_leg)):
            ax.bar(cat, h, bottom=bot, color=col, edgecolor="black",
                   linewidth=0.5, label=lbl)
            if h > 2:
                ax.text(i, bot + h / 2, f"{h:.0f}ms",
                        ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, fontsize=8)
        ax.axhline(83, color="orange", linestyle="--", alpha=0.8, linewidth=1)
        ax.set_ylabel("мс")
        ax.set_title("Вклад компонентов в задержку\n(разница A→B→C)")
        ax.legend(fontsize=7, loc="upper left")
    else:
        ax.set_visible(False)

    # ── График 5 (top-right): загрузка ЦПУ по шагам ─────────────────────────
    ax = axes[0][2]
    cpu_steps = []
    for key, label in [
        ("low",          "LOW\nraw"),
        ("low_diff",     "LOW\n+diff"),
        ("hi",           "HI\nraw"),
        ("hi_diff",      "HI\n+diff"),
        ("yolo",         "YOLO\ninference"),
        ("pipeline_a",   "A\nHI grab"),
        ("pipeline_b",   "B\nHI decode"),
        ("pipeline_c",   "C\nYOLO"),
        ("pipeline_c_cd","C+cd\ncooldown"),
    ]:
        d = report.get(key, {})
        mean_cpu = d.get("cpu_mean_pct")
        max_cpu  = d.get("cpu_max_pct")
        if mean_cpu is not None:
            cpu_steps.append((label, mean_cpu, max_cpu))
    if cpu_steps:
        xlabels = [s[0] for s in cpu_steps]
        means_cpu = [s[1] for s in cpu_steps]
        maxes_cpu = [s[2] for s in cpu_steps]
        x = list(range(len(xlabels)))
        ax.bar([xi - 0.2 for xi in x], means_cpu, 0.38, label="mean", color="#44aadd",
               edgecolor="black", linewidth=0.5)
        ax.bar([xi + 0.2 for xi in x], maxes_cpu, 0.38, label="max", color="#dd4444",
               edgecolor="black", linewidth=0.5)
        for xi, (m, mx) in enumerate(zip(means_cpu, maxes_cpu)):
            ax.text(xi - 0.2, m + 1, f"{m:.0f}%", ha="center", fontsize=7)
            ax.text(xi + 0.2, mx + 1, f"{mx:.0f}%", ha="center", fontsize=7)
        ax.axhline(100, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_ylim(0, 115)
        ax.set_ylabel("% CPU (все ядра)")
        ax.set_title("Загрузка ЦПУ по шагам")
        ax.legend(fontsize=8)
    else:
        ax.set_title("Загрузка ЦПУ\n(нет данных — установите psutil)")
        ax.text(0.5, 0.5, "pip install psutil", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")

    # ── График 6 (bottom-right): CPU: C vs C+cd сравнение ───────────────────
    ax = axes[1][2]
    pc_cpu   = pc.get("cpu_mean_pct") if pc else None
    pc_cd_cpu = pc_cd.get("cpu_mean_pct") if pc_cd else None
    if pc_cpu is not None and pc_cd_cpu is not None:
        cd_fps = pc_cd.get("yolo_max_fps", "?")
        labels_cd = [
            f"C\n(без лимита)",
            f"C+cd\n({cd_fps}fps)",
        ]
        means_pair = [pc.get("cpu_mean_pct", 0), pc_cd.get("cpu_mean_pct", 0)]
        maxes_pair  = [pc.get("cpu_max_pct", 0),  pc_cd.get("cpu_max_pct", 0)]
        fps_pair    = [
            round(1000 / pc.get("loop_mean_ms", 1), 1),
            round(1000 / pc_cd.get("loop_mean_ms", 1), 1),
        ]
        x2 = [0, 1]
        ax.bar([xi - 0.2 for xi in x2], means_pair, 0.38, label="CPU mean",
               color=["#cc44cc", "#44aa88"], edgecolor="black", linewidth=0.5)
        ax.bar([xi + 0.2 for xi in x2], maxes_pair, 0.38, label="CPU max",
               color=["#dd88dd", "#88ccbb"], edgecolor="black", linewidth=0.5)
        for xi, (m, mx, fps) in enumerate(zip(means_pair, maxes_pair, fps_pair)):
            ax.text(xi, max(m, mx) + 2, f"{fps} fps", ha="center", fontsize=8, fontweight="bold")
            ax.text(xi - 0.2, m + 1, f"{m:.0f}%", ha="center", fontsize=7)
            ax.text(xi + 0.2, mx + 1, f"{mx:.0f}%", ha="center", fontsize=7)
        ax.axhline(100, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xticks(x2)
        ax.set_xticklabels(labels_cd, fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_ylabel("% CPU (все ядра)")
        ax.set_title("Cooldown: CPU без ограничения vs с ограничением")
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)

    plt.tight_layout()
    png_path = out_dir / "fps_charts.png"
    plt.savefig(str(png_path), dpi=130)
    plt.close()
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    raise SystemExit(main())
