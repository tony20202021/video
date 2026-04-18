"""
Цикл: читать кадры с каждой доступной RTSP-камеры из .env, сравнивать с предыдущим.
При старте для каждой камеры сохраняется первый удачный кадр (`*__<UTC>_baseline.jpg`) для визуального сравнения.
При средней разнице (после уменьшения и grayscale) выше порога — сохранять кадр в каталог.
HEVC иногда отдаёт битый кадр («серый прямоугольник», в логе FFmpeg POC ref) — перед записью кадр проверяется
(Laplacian variance + std яркости); при провале делаются дополнительные read() до первого годного или сохранение пропускается.

Порог задаётся переменной окружения MOTION_DIFF_THRESHOLD (число, по умолчанию 10.0)
или флагом --threshold. Смысл метрики: mean(|frame - prev|) по пикселям, 0…255.

Период «пульса» MOTION_HEARTBEAT_SEC / --heartbeat-sec: раз в N секунд сохранять кадр с суффиксом _heartbeat.jpg
(даже без движения), чтобы видеть, что цикл жив. По умолчанию 600 с; 0 в .env или в флаге — выключить.

Переменные камер: **CAM_<stem>_URL** (как в verify_cameras.py), напр. CAM_01_9_U_URL.

Обрезка одного «склеенного» кадра (две линзы в одном кадре): в .env задать доли **x,y,w,h** от 0 до 1
(левый верх и размер относительно ширины/высоты кадра). Глобально **MOTION_CROP_REL**, на поток **CAM_<stem>_CROP_REL**
(имя: тот же stem, что в URL: `CAM_01_9_U_CROP_REL` для `CAM_01_9_U_URL`). Примеры: левая половина `0,0,0.5,1`; верхняя `0,0,1,0.5`. Флаг **--crop-rel** — глобально на запуск.

Выход: Ctrl+C — корректное освобождение захватов.

Usage:
    python scripts/cameras/motion_watch.py
    python scripts/cameras/motion_watch.py --threshold 15 --tcp
    python scripts/cameras/motion_watch.py --output .output/motion_watch/my_run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Репозиторий: video/scripts/cameras/motion_watch.py -> video/
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.utils.cam_crop import apply_crop_optional, crop_map_for_cameras, resolve_global_crop
from common.utils.cam_urls import collect_cam_urls as _collect_cam_urls

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT_PARENT = REPO_ROOT / ".output" / "motion_watch"


def _skip_url(url: str) -> bool:
    u = (url or "").strip()
    if not u:
        return True
    if u.startswith("#"):
        return True
    if "<" in u and ">" in u:
        return True
    if not u.lower().startswith("rtsp://"):
        return True
    return False


def _ffmpeg_capture_options(*, use_tcp: bool, stimeout_us: int) -> str:
    parts: list[str] = [f"stimeout;{stimeout_us}"]
    if use_tcp:
        parts.insert(0, "rtsp_transport;tcp")
    return "|".join(parts)


def _stem_from_env_var(var_name: str) -> str:
    return var_name.replace("_URL", "").lower()


def _prepare_gray(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w != width:
        scale = width / float(w)
        nh = max(1, int(round(h * scale)))
        frame = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _mean_abs_diff(prev: np.ndarray, cur: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(prev, cur)))


def _frame_decode_plausible(
    bgr: np.ndarray,
    *,
    min_laplacian_var: float,
    min_gray_std: float,
) -> bool:
    """Отсев типичных артефактов декодера HEVC: однотонный серый кадр без текстуры."""
    if bgr is None or bgr.size == 0:
        return False
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    if float(lap.var()) < min_laplacian_var:
        return False
    if float(g.std()) < min_gray_std:
        return False
    return True


def _read_first_plausible_frame(
    cap: cv2.VideoCapture,
    *,
    max_attempts: int,
    min_laplacian_var: float,
    min_gray_std: float,
) -> np.ndarray | None:
    """До max_attempts чтений — первый кадр, прошедший проверку декодера."""
    for _ in range(max_attempts):
        ok, f = cap.read()
        if (
            ok
            and f is not None
            and f.size > 0
            and _frame_decode_plausible(
                f,
                min_laplacian_var=min_laplacian_var,
                min_gray_std=min_gray_std,
            )
        ):
            return f
    return None


def _pick_frame_to_save_after_motion(
    cap: cv2.VideoCapture,
    first_bgr: np.ndarray,
    *,
    max_extra_reads: int,
    min_laplacian_var: float,
    min_gray_std: float,
) -> np.ndarray | None:
    """Текущий кадр или следующие — первый с нормальной текстурой (не серая заглушка)."""
    if _frame_decode_plausible(
        first_bgr,
        min_laplacian_var=min_laplacian_var,
        min_gray_std=min_gray_std,
    ):
        return first_bgr
    for _ in range(max_extra_reads):
        ok, f = cap.read()
        if not ok or f is None or f.size == 0:
            continue
        if _frame_decode_plausible(
            f,
            min_laplacian_var=min_laplacian_var,
            min_gray_std=min_gray_std,
        ):
            return f
    return None



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Непрерывное сравнение последовательных кадров RTSP, сохранение при изменении"
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV, help="Путь к .env")
    parser.add_argument(
        "--crop-rel",
        type=str,
        default=None,
        metavar="X,Y,W,H",
        help="Глобальная обрезка склеенного кадра, доли 0…1 (перебивает MOTION_CROP_REL из .env)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Порог mean abs diff (0–255); иначе MOTION_DIFF_THRESHOLD из .env или 10",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Каталог для кадров (по умолчанию .output/motion_watch/motion_watch_<UTC>)",
    )
    parser.add_argument("--tcp", action="store_true", help="RTSP через TCP")
    parser.add_argument(
        "--compare-width",
        type=int,
        default=320,
        help="Ширина уменьшения для метрики различия (скорость/шум)",
    )
    parser.add_argument(
        "--open-timeout-ms",
        type=int,
        default=10000,
        help="Таймаут открытия потока (если поддерживается OpenCV)",
    )
    parser.add_argument(
        "--read-timeout-ms",
        type=int,
        default=10000,
        help="Таймаут чтения кадра (если поддерживается)",
    )
    parser.add_argument(
        "--stimeout-us",
        type=int,
        default=8_000_000,
        help="FFmpeg stimeout, мкс",
    )
    parser.add_argument(
        "--min-laplacian-var",
        type=float,
        default=12.0,
        help="Мин. variance(Laplacian) по яркости — ниже считаем битым/пустым кадром (HEVC POC)",
    )
    parser.add_argument(
        "--min-gray-std",
        type=float,
        default=2.5,
        help="Мин. std(яркость) полного кадра — отсекает однотонный серый",
    )
    parser.add_argument(
        "--save-extra-reads",
        type=int,
        default=12,
        help="Сколько доп. read() искать годный кадр при срабатывании движения",
    )
    parser.add_argument(
        "--baseline-attempts",
        type=int,
        default=16,
        help="Попыток read() для базового кадра (первый прошедший проверку)",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=None,
        help="Раз в N сек — *_heartbeat.jpg без движения; 0 = выкл; иначе .env MOTION_HEARTBEAT_SEC или 600",
    )
    args = parser.parse_args()

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
        threshold_from = (
            f"MOTION_DIFF_THRESHOLD={raw_t!r} (.env)" if raw_t else "встроенное по умолчанию"
        )

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
            heartbeat_from = "встроенное по умолчанию 600 с"

    out_dir = args.output
    if out_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUTPUT_PARENT / f"motion_watch_{run_id}"
        output_from = f"авто .output/motion_watch/motion_watch_{run_id}/"
    else:
        output_from = "аргумент --output"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _ffmpeg_capture_options(
        use_tcp=args.tcp,
        stimeout_us=args.stimeout_us,
    )
    ff_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")

    cameras = _collect_cam_urls()
    active: list[tuple[str, str]] = [
        (k, v) for k, v in cameras if not _skip_url(v)
    ]
    if not active:
        print(
            "Нет активных rtsp:// URL (CAM_<stem>_URL без плейсхолдеров).",
            file=sys.stderr,
        )
        return 1

    crop_cli = (args.crop_rel or "").strip() or None
    global_crop, global_crop_from = resolve_global_crop(
        crop_rel_arg=crop_cli,
        motion_crop_env=os.environ.get("MOTION_CROP_REL"),
    )
    if crop_cli and global_crop is None:
        return 1

    crop_by_cam = crop_map_for_cameras(active, global_crop=global_crop)

    caps: dict[str, cv2.VideoCapture] = {}

    print("Параметры запуска (фактические):")
    print(f"  env файл:                    {args.env.resolve()}")
    print(f"  каталог вывода:              {out_dir.resolve()}  ({output_from})")
    print(
        f"  порог движения (mean |Δ|):   {threshold}  [{threshold_from}]"
    )
    print(f"  compare_width:                {args.compare_width}")
    print(f"  пульс (сек):                  {heartbeat_sec}  [{heartbeat_from}]")
    print(f"  RTSP TCP (--tcp):             {args.tcp}")
    print(f"  open_timeout_ms:              {args.open_timeout_ms}")
    print(f"  read_timeout_ms:              {args.read_timeout_ms}")
    print(f"  stimeout_us (FFmpeg):         {args.stimeout_us}")
    print(f"  min_laplacian_var:            {args.min_laplacian_var}")
    print(f"  min_gray_std:                 {args.min_gray_std}")
    print(f"  save_extra_reads:             {args.save_extra_reads}")
    print(f"  baseline_attempts:            {args.baseline_attempts}")
    print(f"  OPENCV_FFMPEG_CAPTURE_OPTIONS: {ff_opts!r}")
    print(f"  OpenCV:                       {cv2.__version__}")
    print(f"  камеры ({len(active)}):       {', '.join(k for k, _ in active)}")
    print(f"  обрезка (глобально):          {global_crop!r}  [{global_crop_from}]")
    for vn, _ in active:
        c = crop_by_cam[vn]
        spec_k = vn.replace("_URL", "_CROP_REL")
        raw_spec = (os.environ.get(spec_k) or "").strip()
        if c is None:
            print(f"    └ {vn}: полный кадр")
        elif raw_spec:
            print(f"    └ {vn}: x,y,w,h={c}  ← {spec_k}")
        else:
            print(f"    └ {vn}: x,y,w,h={c}  ← глобальная обрезка")
    print("Останов: Ctrl+C\n")

    for var_name, url in active:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(args.open_timeout_ms))
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(args.read_timeout_ms))
        except Exception:
            pass
        if not cap.isOpened():
            print(f"  [!] {var_name}: не удалось открыть поток", file=sys.stderr)
            cap.release()
            continue
        caps[var_name] = cap

    if not caps:
        print("Ни одна камера не открылась.", file=sys.stderr)
        return 1

    prev_gray: dict[str, np.ndarray | None] = {k: None for k in caps}

    print("Базовый кадр (старт)…")
    for var_name, cap in caps.items():
        frame0 = _read_first_plausible_frame(
            cap,
            max_attempts=args.baseline_attempts,
            min_laplacian_var=args.min_laplacian_var,
            min_gray_std=args.min_gray_std,
        )
        if frame0 is None:
            print(f"  [!] {var_name}: нет годного кадра для baseline", file=sys.stderr)
            continue
        crop = crop_by_cam[var_name]
        frame0u = apply_crop_optional(frame0, crop)
        stem = _stem_from_env_var(var_name)
        ts0 = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        bname = f"{stem}__{ts0}_baseline.jpg"
        cv2.imwrite(str(out_dir / bname), frame0u)
        prev_gray[var_name] = _prepare_gray(frame0u, args.compare_width)
        print(f"  {bname}")
    print()

    last_heartbeat: dict[str, float] = {k: time.monotonic() for k in caps}

    try:
        while True:
            for var_name, cap in list(caps.items()):
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    continue
                crop = crop_by_cam[var_name]
                frame_u = apply_crop_optional(frame, crop)
                gray = _prepare_gray(frame_u, args.compare_width)
                prev = prev_gray[var_name]
                if prev is None:
                    prev_gray[var_name] = gray
                    continue
                diff = _mean_abs_diff(prev, gray)
                if diff <= threshold:
                    prev_gray[var_name] = gray
                else:
                    to_save = _pick_frame_to_save_after_motion(
                        cap,
                        frame,
                        max_extra_reads=args.save_extra_reads,
                        min_laplacian_var=args.min_laplacian_var,
                        min_gray_std=args.min_gray_std,
                    )
                    if to_save is None:
                        print(
                            f"  [~] {var_name}: движение diff={diff:.2f}, "
                            f"но нет годного кадра после HEVC — пропуск",
                            file=sys.stderr,
                        )
                    else:
                        to_u = apply_crop_optional(to_save, crop)
                        prev_gray[var_name] = _prepare_gray(to_u, args.compare_width)
                        stem = _stem_from_env_var(var_name)
                        ts = datetime.now(timezone.utc).strftime(
                            "%Y%m%d_%H%M%S_%f"
                        )
                        fname = f"{stem}_{ts}.jpg"
                        cv2.imwrite(str(out_dir / fname), to_u)
                        print(f"  сохранено {fname}  (diff={diff:.2f})")

                if heartbeat_sec > 0:
                    now = time.monotonic()
                    if now - last_heartbeat[var_name] >= heartbeat_sec:
                        last_heartbeat[var_name] = now
                        hb = _pick_frame_to_save_after_motion(
                            cap,
                            frame,
                            max_extra_reads=args.save_extra_reads,
                            min_laplacian_var=args.min_laplacian_var,
                            min_gray_std=args.min_gray_std,
                        )
                        if hb is not None:
                            hb_u = apply_crop_optional(hb, crop)
                            stem = _stem_from_env_var(var_name)
                            ts = datetime.now(timezone.utc).strftime(
                                "%Y%m%d_%H%M%S_%f"
                            )
                            hb_name = f"{stem}_{ts}_heartbeat.jpg"
                            cv2.imwrite(str(out_dir / hb_name), hb_u)
                            print(f"  пульс {hb_name}")
                        else:
                            print(
                                f"  [~] {var_name}: пульс — нет годного кадра",
                                file=sys.stderr,
                            )
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nОстанов по Ctrl+C")
    finally:
        for cap in caps.values():
            cap.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
