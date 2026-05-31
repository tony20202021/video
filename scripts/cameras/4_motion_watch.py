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

ВНИМАНИЕ: не запускайте скрипты 4 и 5 одновременно против одних и тех же камер.
Оба скрипта читают разные кадры из одного RTSP-буфера → кадры не совпадают (см. задачу 18b).
Для продакшена используйте scripts/agent/agent.py или только 5_motion_people.py.

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
from common.utils.motion_utils import (
    StreamReader,
    ffmpeg_capture_options as _ffmpeg_capture_options,
    mean_abs_diff as _mean_abs_diff,
    open_cap as _open_capture,
    prepare_gray as _prepare_gray,
    read_first_plausible_frame as _read_first_plausible_frame,
    redact_url,
    reopen_cap as _reopen_cap,
    skip_url as _skip_url,
    stem_from_var as _stem_from_env_var,
)
from common.utils.time_msk import ts_for_dir, ts_for_file

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT_PARENT = REPO_ROOT / ".output" / "4_motion_watch"



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
        default=3_000_000,
        help="FFmpeg stimeout, мкс",
    )
    parser.add_argument(
        "--hi-scale", type=float, default=1.0,
        help="Уменьшить HI кадр перед хранением в StreamReader (0.5 = вдвое меньше).",
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

    # Сохраняем параметры запуска сразу при старте
    import json as _json
    from common.utils.time_msk import ts_iso as _ts_iso
    _run_params = {
        "started_at_msk": _ts_iso(),
        "script": "4_motion_watch.py",
        "threshold": args.threshold,
        "heartbeat_sec": None,  # заполним после вычисления
        "tcp": args.tcp,
        "compare_width": args.compare_width,
        "output": str(out_dir),
    }

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

    # LOW потоки — прямые VideoCapture
    caps_by_url: dict[str, cv2.VideoCapture] = {}
    for url in sorted(set(var_low_url.values())):
        cap = _open_capture(url, open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms)
        if not cap.isOpened():
            print(f"  [!] LOW не удалось открыть: {redact_url(url)[:80]}…", file=sys.stderr)
            cap.release()
        else:
            caps_by_url[url] = cap

    # HI потоки — StreamReader (фоновые потоки, не блокируют основной цикл)
    hi_readers: dict[str, StreamReader] = {}
    for url in sorted(set(var_hi_url.values())):
        reader = StreamReader(
            url,
            open_timeout_ms=args.open_timeout_ms,
            read_timeout_ms=args.read_timeout_ms,
            min_laplacian_var=args.min_laplacian_var,
            min_gray_std=args.min_gray_std,
            scale=args.hi_scale,
        )
        reader.start()
        hi_readers[url] = reader

    opened_vars: list[tuple[str, str]] = [
        (vn, lu)
        for vn, lu in active
        if lu in caps_by_url and var_hi_url[vn] in hi_readers
    ]
    if not opened_vars:
        print("Ни одна камера не открылась (проверьте URL и сеть).", file=sys.stderr)
        for c in caps_by_url.values():
            c.release()
        for r in hi_readers.values():
            r.stop()
        return 1

    vars_by_low: dict[str, list[str]] = defaultdict(list)
    for vn, lu in opened_vars:
        vars_by_low[lu].append(vn)

    prev_gray: dict[str, np.ndarray | None] = {vn: None for vn, _ in opened_vars}
    last_heartbeat: dict[str, float] = {vn: time.monotonic() for vn, _ in opened_vars}
    _MAX_LOW_FAILS = 5
    _low_fail: dict[str, int] = defaultdict(int)
    _HI_MAX_AGE = 3.0

    # Дописываем параметры и сохраняем run_params.json
    _run_params["heartbeat_sec"] = heartbeat_sec
    _run_params["threshold"] = threshold
    _run_params["cameras"] = [vn for vn, _ in opened_vars]
    _run_params["crop_global"] = list(global_crop) if global_crop else None
    _run_params["opencv_version"] = cv2.__version__
    (out_dir / "run_params.json").write_text(
        _json.dumps(_run_params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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

    print("Базовый кадр (LOW → детектор, HI → файл)…")
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
            fl = apply_crop_optional(frame_l, crop_by_cam[var_name])
            prev_gray[var_name] = _prepare_gray(fl, args.compare_width)

    by_hi_baseline: dict[str, list[str]] = defaultdict(list)
    for var_name, _ in opened_vars:
        by_hi_baseline[var_hi_url[var_name]].append(var_name)

    for hi_u, vlist in by_hi_baseline.items():
        reader = hi_readers[hi_u]
        deadline = time.monotonic() + 15.0
        fh: np.ndarray | None = None
        while time.monotonic() < deadline:
            fh, _ = reader.get_latest()
            if fh is not None:
                break
            time.sleep(0.2)
        if fh is None:
            for var_name in vlist:
                print(f"  [!] {var_name}: нет HI baseline за 15с", file=sys.stderr)
            continue
        for var_name in vlist:
            im = apply_crop_optional(fh, crop_by_cam[var_name])
            stem = _stem_from_env_var(var_name)
            ts0 = ts_for_file()
            bname = f"{stem}_{ts0}_baseline.jpg"
            cv2.imwrite(str(out_dir / bname), im)
            print(f"  {bname}")
    print()

    try:
        while True:
            for low_u, var_list in vars_by_low.items():
                cap_l = caps_by_url.get(low_u)
                if cap_l is None:
                    if _reopen_cap(low_u, caps_by_url, open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms):
                        print(f"  [R] LOW переподключён: {redact_url(low_u)[:60]}")
                        _low_fail[low_u] = 0
                    continue
                ok_l, frame_l = cap_l.read()
                if not ok_l or frame_l is None or frame_l.size == 0:
                    _low_fail[low_u] += 1
                    if _low_fail[low_u] >= _MAX_LOW_FAILS:
                        print(f"  [!] LOW {_MAX_LOW_FAILS} сбоев — переподключение", file=sys.stderr)
                        _reopen_cap(low_u, caps_by_url, open_timeout_ms=args.open_timeout_ms, read_timeout_ms=args.read_timeout_ms)
                        _low_fail[low_u] = 0
                    continue
                _low_fail[low_u] = 0

                pending_motion: list[tuple[str, float, np.ndarray]] = []
                for var_name in var_list:
                    frame_u = apply_crop_optional(frame_l, crop_by_cam[var_name])
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

                if pending_motion:
                    by_hi: dict[str, list[tuple[str, float, np.ndarray]]] = defaultdict(list)
                    for var_name, diff, gray_low in pending_motion:
                        by_hi[var_hi_url[var_name]].append((var_name, diff, gray_low))

                    for hi_u, entries in by_hi.items():
                        reader = hi_readers.get(hi_u)
                        if reader is None:
                            continue
                        to_save, _ = reader.get_latest()  # не блокирует
                        if to_save is None or reader.age_sec() > _HI_MAX_AGE:
                            for var_name, diff, _ in entries:
                                print(
                                    f"  [~] {var_name}: движение diff={diff:.2f}, "
                                    f"HI недоступен (age={reader.age_sec():.1f}s)",
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
                    reader = hi_readers.get(var_hi_url[var_name])
                    if reader is None:
                        continue
                    hb, _ = reader.get_latest()
                    if hb is None or reader.age_sec() > _HI_MAX_AGE * 2:
                        continue
                    last_heartbeat[var_name] = now  # сбрасываем только при успехе
                    hb_u = apply_crop_optional(hb, crop_by_cam[var_name])
                    stem = _stem_from_env_var(var_name)
                    ts = ts_for_file()
                    hb_name = f"{stem}_{ts}_heartbeat.jpg"
                    cv2.imwrite(str(out_dir / hb_name), hb_u)
                    print(f"  пульс {hb_name}")

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nОстанов по Ctrl+C")
    finally:
        for cap in caps_by_url.values():
            cap.release()
        for reader in hi_readers.values():
            reader.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
