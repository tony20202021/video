"""
Цикл: читать кадры с каждой доступной RTSP-камеры из .env, сравнивать с предыдущим.
Детекция движения — по **низкому разрешению** (переменные **CAM_<stem>_URL**, обычно stream=1 / субпоток).
Сохранение кадров (движение, baseline, heartbeat) — с **высокого разрешения**, если задан **CAM_<stem>_HI_URL**
(главный поток, stream=0); иначе тот же URL, что и для детекции.

Один и тот же субпоток для двух логических камер (напр. U/D с одного склеенного кадра): открывается **один**
VideoCapture на уникальный RTSP URL — один read() на такт для всей группы; движение считается по двум обрезкам.

При старте для каждой камеры сохраняется baseline с **HI** (или с того же потока, если HI не задан).
HEVC: проверка кадра (Laplacian + std); доп. read() при сохранении.

Порог: **MOTION_DIFF_THRESHOLD** / **--threshold**. Пульс: **MOTION_HEARTBEAT_SEC** / **--heartbeat-sec**.

Обрезка: **MOTION_CROP_REL**, **CAM_<stem>_CROP_REL**, **--crop-rel** — одинаково для низкого и высокого кадра (доли 0…1).

Usage:
    python scripts/cameras/4_motion_watch.py
    python scripts/cameras/4_motion_watch.py --threshold 15 --tcp
    python scripts/cameras/4_motion_watch.py --output .output/motion_watch/my_run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Репозиторий: video/scripts/cameras/4_motion_watch.py -> video/
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.utils.cam_crop import apply_crop_optional, crop_map_for_cameras, resolve_global_crop
from common.utils.cam_urls import collect_cam_urls as _collect_cam_urls
from common.utils.cam_urls import resolve_hi_rtsp_url
from common.utils.motion_utils import redact_url
from common.utils.time_msk import ts_for_dir, ts_for_file

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT_PARENT = REPO_ROOT / ".output" / "4_motion_watch"


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


def _read_hi_save_frame(
    cap: cv2.VideoCapture,
    *,
    max_extra_reads: int,
    min_laplacian_var: float,
    min_gray_std: float,
) -> np.ndarray | None:
    ok, f = cap.read()
    if not ok or f is None or f.size == 0:
        return None
    return _pick_frame_to_save_after_motion(
        cap,
        f,
        max_extra_reads=max_extra_reads,
        min_laplacian_var=min_laplacian_var,
        min_gray_std=min_gray_std,
    )


def _open_capture(
    url: str,
    *,
    open_timeout_ms: int,
    read_timeout_ms: int,
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(open_timeout_ms))
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(read_timeout_ms))
    except Exception:
        pass
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Детекция по субпотоку CAM_*_URL, сохранение с CAM_*_HI_URL при наличии"
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
        help="Ширина уменьшения для метрики различия (по низкому кадру после обрезки)",
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
        run_id = ts_for_dir()
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

    var_low_url: dict[str, str] = {vn: url for vn, url in active}
    var_hi_url: dict[str, str] = {
        vn: resolve_hi_rtsp_url(vn, low_url=url, skip_url=_skip_url)
        for vn, url in active
    }

    unique_urls = set(var_low_url.values()) | set(var_hi_url.values())
    caps_by_url: dict[str, cv2.VideoCapture] = {}
    for url in sorted(unique_urls):
        cap = _open_capture(
            url,
            open_timeout_ms=args.open_timeout_ms,
            read_timeout_ms=args.read_timeout_ms,
        )
        if not cap.isOpened():
            print(f"  [!] не удалось открыть поток: {redact_url(url)[:80]}…", file=sys.stderr)
            cap.release()
            continue
        caps_by_url[url] = cap

    opened_vars: list[tuple[str, str]] = [
        (vn, lu)
        for vn, lu in active
        if lu in caps_by_url and var_hi_url[vn] in caps_by_url
    ]
    if not opened_vars:
        print("Ни одна камера не открылась (проверьте URL и сеть).", file=sys.stderr)
        for c in caps_by_url.values():
            c.release()
        return 1

    vars_by_low: dict[str, list[str]] = defaultdict(list)
    for vn, lu in opened_vars:
        vars_by_low[lu].append(vn)

    prev_gray: dict[str, np.ndarray | None] = {vn: None for vn, _ in opened_vars}
    last_heartbeat: dict[str, float] = {vn: time.monotonic() for vn, _ in opened_vars}

    print("Параметры запуска (фактические):")
    print(f"  env файл:                    {args.env.resolve()}")
    print(f"  каталог вывода:              {out_dir.resolve()}  ({output_from})")
    print(f"  порог движения (mean |Δ|):   {threshold}  [{threshold_from}]")
    print(f"  compare_width (по LOW):       {args.compare_width}")
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
    print(f"  камеры ({len(opened_vars)}):  {', '.join(vn for vn, _ in opened_vars)}")
    print(f"  обрезка (глобально):          {global_crop!r}  [{global_crop_from}]")
    print("  потоки: LOW = CAM_*_URL (детекция), HI = CAM_*_HI_URL или тот же URL (сохранение)")
    low_groups_n = len(vars_by_low)
    print(f"  уникальных LOW RTSP:          {low_groups_n} (один read на группу в цикле)")
    for vn, _ in opened_vars:
        c = crop_by_cam[vn]
        spec_k = vn.replace("_URL", "_CROP_REL")
        raw_spec = (os.environ.get(spec_k) or "").strip()
        low_u = var_low_url[vn]
        hi_u = var_hi_url[vn]
        hi_note = "отдельный HI" if hi_u != low_u else "как LOW"
        cr = (
            f"x,y,w,h={c}  ← {spec_k}"
            if raw_spec
            else (f"x,y,w,h={c}  ← глобальная" if c is not None else "полный кадр")
        )
        print(f"    └ {vn}: LOW … {cr}; HI: {hi_note}")
    print("Останов: Ctrl+C\n")

    print("Базовый кадр (LOW → состояние детектора, HI → файл)…")
    for low_u, var_list in vars_by_low.items():
        cap_l = caps_by_url[low_u]
        frame_l = _read_first_plausible_frame(
            cap_l,
            max_attempts=args.baseline_attempts,
            min_laplacian_var=args.min_laplacian_var,
            min_gray_std=args.min_gray_std,
        )
        if frame_l is None:
            for var_name in var_list:
                print(f"  [!] {var_name}: нет годного LOW для baseline", file=sys.stderr)
            continue
        for var_name in var_list:
            crop = crop_by_cam[var_name]
            fl = apply_crop_optional(frame_l, crop)
            prev_gray[var_name] = _prepare_gray(fl, args.compare_width)

    by_hi_baseline: dict[str, list[str]] = defaultdict(list)
    for var_name, _ in opened_vars:
        by_hi_baseline[var_hi_url[var_name]].append(var_name)

    for hi_u, vlist in by_hi_baseline.items():
        cap_h = caps_by_url[hi_u]
        fh = _read_first_plausible_frame(
            cap_h,
            max_attempts=args.baseline_attempts,
            min_laplacian_var=args.min_laplacian_var,
            min_gray_std=args.min_gray_std,
        )
        if fh is None:
            for var_name in vlist:
                print(f"  [!] {var_name}: нет годного HI для baseline", file=sys.stderr)
            continue
        for var_name in vlist:
            crop = crop_by_cam[var_name]
            im = apply_crop_optional(fh, crop)
            stem = _stem_from_env_var(var_name)
            ts0 = ts_for_file()
            bname = f"{stem}_{ts0}_baseline.jpg"
            cv2.imwrite(str(out_dir / bname), im)
            print(f"  {bname}")
    print()

    try:
        while True:
            for low_u, var_list in vars_by_low.items():
                cap_l = caps_by_url[low_u]
                ok_l, frame_l = cap_l.read()
                if not ok_l or frame_l is None or frame_l.size == 0:
                    continue
                if not _frame_decode_plausible(
                    frame_l,
                    min_laplacian_var=args.min_laplacian_var,
                    min_gray_std=args.min_gray_std,
                ):
                    continue

                pending_motion: list[tuple[str, float, np.ndarray]] = []
                for var_name in var_list:
                    crop = crop_by_cam[var_name]
                    frame_u = apply_crop_optional(frame_l, crop)
                    gray = _prepare_gray(frame_u, args.compare_width)
                    prev = prev_gray[var_name]
                    if prev is None:
                        prev_gray[var_name] = gray
                        continue
                    diff = _mean_abs_diff(prev, gray)
                    if diff <= threshold:
                        prev_gray[var_name] = gray
                    else:
                        pending_motion.append((var_name, diff, gray))

                by_hi: dict[str, list[tuple[str, float, np.ndarray]]] = defaultdict(list)
                for var_name, diff, gray_low in pending_motion:
                    by_hi[var_hi_url[var_name]].append((var_name, diff, gray_low))

                for hi_u, entries in by_hi.items():
                    cap_h = caps_by_url[hi_u]
                    to_save = _read_hi_save_frame(
                        cap_h,
                        max_extra_reads=args.save_extra_reads,
                        min_laplacian_var=args.min_laplacian_var,
                        min_gray_std=args.min_gray_std,
                    )
                    if to_save is None:
                        for var_name, diff, _g in entries:
                            print(
                                f"  [~] {var_name}: движение diff={diff:.2f}, "
                                f"нет годного HI — пропуск",
                                file=sys.stderr,
                            )
                        continue
                    for var_name, diff, gray_low in entries:
                        to_u = apply_crop_optional(to_save, crop_by_cam[var_name])
                        prev_gray[var_name] = gray_low
                        stem = _stem_from_env_var(var_name)
                        ts = ts_for_file()
                        fname = f"{stem}_{ts}.jpg"
                        cv2.imwrite(str(out_dir / fname), to_u)
                        print(f"  сохранено {fname}  (diff={diff:.2f}, HI)")

                for var_name in var_list:
                    if heartbeat_sec <= 0:
                        continue
                    now = time.monotonic()
                    if now - last_heartbeat[var_name] < heartbeat_sec:
                        continue
                    last_heartbeat[var_name] = now
                    hi_u = var_hi_url[var_name]
                    cap_h = caps_by_url[hi_u]
                    hb = _read_hi_save_frame(
                        cap_h,
                        max_extra_reads=args.save_extra_reads,
                        min_laplacian_var=args.min_laplacian_var,
                        min_gray_std=args.min_gray_std,
                    )
                    if hb is not None:
                        hb_u = apply_crop_optional(hb, crop_by_cam[var_name])
                        stem = _stem_from_env_var(var_name)
                        ts = ts_for_file()
                        hb_name = f"{stem}_{ts}_heartbeat.jpg"
                        cv2.imwrite(str(out_dir / hb_name), hb_u)
                        print(f"  пульс {hb_name}")
                    else:
                        print(
                            f"  [~] {var_name}: пульс — нет годного HI",
                            file=sys.stderr,
                        )

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nОстанов по Ctrl+C")
    finally:
        for cap in caps_by_url.values():
            cap.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
