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
from common.utils.atomic import imwrite as _imwrite
from common.utils.cam_crop import apply_crop_optional, crop_map_for_cameras, resolve_global_crop
from common.utils.cam_urls import collect_cam_urls as _collect_cam_urls
from common.utils.camera_run import (
    CapReader as _CapReader,
    CpuMonitor as _CpuMonitor,
    compute_per_frame_log as _compute_per_frame_log,
    parse_img_filename as _parse_img_filename,
    regen_osd_from_images as _regen_osd_from_images,
    save_charts as _save_charts_base,
    save_cpu_csv as _save_cpu_csv,
    save_osd_chart as _save_osd_chart,
    save_pts_chart as _save_pts_chart,
    save_run_stats as _save_run_stats,
)
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
import logging
from common.utils.log_setup import setup_logging, add_file_handler

logger = logging.getLogger(__name__)

DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT_PARENT = REPO_ROOT / ".output" / "pipeline" / "1_motion_diff"

_S4_SAVE_COLORS = {"baseline": "#888888", "diff": "#cc3333", "heartbeat": "#8844bb", "raw": "#dd8800"}
_S4_SAVE_LEVELS = {"baseline": 1, "diff": 2, "heartbeat": 3, "raw": 4}


def _save_charts(frame_log, cpu_log, saves_log, diffs_log, threshold, out_dir):
    _save_charts_base(frame_log, cpu_log, saves_log, diffs_log, threshold, out_dir,
                      title="4_motion_diff_low",
                      event_type="diff",
                      save_colors=_S4_SAVE_COLORS,
                      save_levels=_S4_SAVE_LEVELS)


# ─── Вспомогательные классы ───────────────────────────────────────────────────

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



def _regen_csv_from_images(run_dir: Path) -> None:
    """Восстанавливает saves.csv и diffs.csv из имён сохранённых картинок."""
    import re as _re
    from datetime import datetime as _dt

    images_dir = run_dir / "images"
    if not images_dir.exists():
        logger.info("  [!] images/ не найден — восстановление невозможно")
        return

    def _parse_stem(stem: str):
        """→ (ts_human, ts_msk, cam, img_type) или None."""
        parts = stem.split("_")
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                cam = "_".join(parts[:i])
                ts_msk = "_".join(parts[i:i+4]) if i + 3 < len(parts) else part
                img_type = parts[-1]
                date_p = part
                time_p = parts[i + 1] if i + 1 < len(parts) else "000000"
                ts_human = (f"{date_p[:4]}-{date_p[4:6]}-{date_p[6:]} "
                            f"{time_p[:2]}:{time_p[2:4]}:{time_p[4:6]}")
                return ts_human, ts_msk, cam, img_type
        return None

    # entries: (ts_human, ts_msk, cam, img_type)
    entries = []
    for p in sorted(images_dir.rglob("*.jpg")):
        r = _parse_stem(p.stem)
        if r:
            entries.append(r)

    if not entries:
        logger.info("  [!] Картинок не найдено — восстановление невозможно")
        return

    entries.sort(key=lambda x: x[0])

    def _ts_to_epoch(ts_human: str) -> float:
        try:
            return _dt.strptime(ts_human, "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            return 0.0

    t0 = _ts_to_epoch(entries[0][0])
    saves_rows, diffs_rows = [], []

    for ts_human, ts_msk, cam, img_type in entries:
        mono_s = round(_ts_to_epoch(ts_human) - t0, 3)
        if img_type.startswith("diff"):
            save_type = "diff"
            m = _re.match(r"diff(\d+\.?\d*)", img_type)
            diff_val = float(m.group(1)) if m else 0.0
            diffs_rows.append([mono_s, ts_msk, cam, round(diff_val, 3)])
        elif img_type in ("baseline", "heartbeat"):
            save_type = img_type
        else:
            save_type = img_type
        saves_rows.append([mono_s, ts_msk, cam, save_type])

    if saves_rows:
        with open(run_dir / "saves.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mono_s", "ts_msk", "cam", "type"])
            w.writerows(saves_rows)
        logger.info(f"  saves.csv:  {len(saves_rows)} записей (из картинок)")
    if diffs_rows:
        with open(run_dir / "diffs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mono_s", "ts_msk", "cam", "diff"])
            w.writerows(diffs_rows)
        logger.info(f"  diffs.csv:  {len(diffs_rows)} записей (из картинок)")


def _regen_charts(run_dir: Path) -> None:
    """Перечитывает CSV из существующего run-каталога и перегенерирует графики."""
    import csv as _csv
    import json as _json

    def _load(name: str) -> list:
        p = run_dir / name
        if not p.exists():
            logger.info(f"  [!] {name} не найден — пропущено")
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

    logger.info(f"Перегенерация графиков из: {run_dir}")

    if not (run_dir / "saves.csv").exists():
        logger.info("CSV-файлы не найдены — восстанавливаю из картинок…")
        _regen_csv_from_images(run_dir)

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



# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_logging()
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
        "--csv-save-interval", type=float, default=60.0, metavar="SEC",
        help="Интервал периодического сохранения cpu.csv (сек). 0 = только в конце.",
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
        logger.warning(f"Файл .env не найден: {args.env}")
        return 1

    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("Нужен пакет python-dotenv: pip install python-dotenv")
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

    _base      = args.output or DEFAULT_OUTPUT_PARENT
    _today     = ts_for_file()[:8]
    images_dir = _base / "images"
    meta_dir   = _base / "meta" / _today
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    _run_log_fh = add_file_handler(meta_dir / 'run.log')

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _ffmpeg_capture_options(
        use_tcp=args.tcp, stimeout_us=args.stimeout_us,
    )

    cameras = _collect_cam_urls()
    active: list[tuple[str, str]] = [(k, v) for k, v in cameras if not _skip_url(v)]
    if not active:
        logger.warning("Нет активных rtsp:// URL (CAM_<stem>_URL без плейсхолдеров).")
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
            logger.warning(f"  [!] LOW не удалось открыть: {redact_url(url)[:80]}…")
        else:
            readers[url] = _CapReader(url, cap, args.open_timeout_ms, args.read_timeout_ms)

    opened_vars: list[tuple[str, str]] = [(vn, url) for vn, url in active if url in readers]
    if not opened_vars:
        logger.warning("Ни одна камера не открылась (проверьте URL и сеть).")
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

    cam_dirs: dict[tuple[str, str], Path] = {}

    def _cam_dir(vn: str) -> Path:
        today = ts_for_file()[:8]  # YYYYMMDD in MSK
        key = (today, vn)
        if key not in cam_dirs:
            d = images_dir / today / _stem_from_env_var(vn)
            d.mkdir(parents=True, exist_ok=True)
            (d / "diff").mkdir(exist_ok=True)
            cam_dirs[key] = d
        return cam_dirs[key]

    prev_gray: dict[str, np.ndarray | None] = {vn: None for vn, _ in opened_vars}
    last_good_frame: dict[str, np.ndarray | None] = {vn: None for vn, _ in opened_vars}
    last_heartbeat: dict[str, float] = {vn: time.monotonic() for vn, _ in opened_vars}
    last_save_time: dict[str, float] = {vn: time.monotonic() for vn, _ in opened_vars}
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
    (meta_dir / "run_params.json").write_text(_json.dumps({
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
        "images_dir": str(images_dir),
        "meta_dir":   str(meta_dir),
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
    logger.info(f"Порог:     {threshold}  [{threshold_from}]")
    logger.info(f"Пульс:     {heartbeat_sec} сек  [{heartbeat_from}]")
    logger.info(f"Длит.:     {duration_desc}")
    logger.info(f"Вывод:     images={images_dir}  meta={meta_dir}")
    logger.info(f"Камеры ({len(opened_vars)}): {', '.join(vn for vn, _ in opened_vars)}")
    logger.info(f"Обрезка:   {global_crop!r}  [{global_crop_from}]")
    for vn, _ in opened_vars:
        c = crop_by_cam[vn]
        cr = (f"x,y,w,h={c}" if c is not None else "полный кадр")
        logger.info(f"  └ {vn}: {cr}")
    logger.info("Останов: Ctrl+C")

    t_start = time.monotonic()

    cpu_monitor = _CpuMonitor(interval=max(args.cpu_interval, 0.5))
    cpu_active = args.cpu_interval > 0 and cpu_monitor.start(t_start)
    if args.cpu_interval > 0 and not cpu_active:
        logger.info("  [!] psutil не установлен — мониторинг ЦПУ недоступен (pip install psutil)")

    logger.info("Базовый кадр…")
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
                logger.warning(f"  [!] {vn}: нет годного кадра для baseline")
            continue
        for vn in var_list:
            fl = apply_crop_optional(frame_l, crop_by_cam[vn])
            prev_gray[vn] = _prepare_gray(fl)
            last_good_frame[vn] = fl
            stem = _stem_from_env_var(vn)
            calib0 = rtcp_workers[low_u].get_calib() if rtcp_workers else None
            bname = f"{stem}_{_ts(calib0)}_baseline.jpg"
            _imwrite(_cam_dir(vn) / bname, fl)
            saves_log.append([round(time.monotonic() - t_start, 4), ts_for_file(), vn, "baseline"])
            logger.info(f"  {bname}")
    logger.info('')

    _current_date = ts_for_file()[:8]  # YYYYMMDD MSK при старте
    # cpu.csv ведётся по дню: снапшот CpuMonitor кумулятивный, поэтому фильтруем строки по дате
    # (ts_msk = 'YYYYMMDD_...') → meta/<день>/cpu.csv содержит только свой день.
    _cpu_for_day = lambda rows, day: [r for r in rows if str(r[1])[:8] == day]

    def _flush_day_stats(date_str: str, clear: bool = True) -> None:
        """Сохранить логи за date_str в images_dir/date_str/.

        clear=True (смена суток / конец прогона) — записать и обнулить.
        clear=False (периодически во время прогона) — записать без обнуления,
        чтобы frames.csv/diffs.csv были доступны на диске у ИДУЩЕГО прогона
        (иначе они появлялись только при смене суток или в самом конце)."""
        if not date_str or not (saves_log or frame_log or diffs_log or pts_log):
            return
        day_dir = images_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        for _name, _hdr, _log in [
            ("frames.csv", ["mono_s", "ts_msk", "url_id", "ok", "plausible", "event"], frame_log),
            ("saves.csv",  ["mono_s", "ts_msk", "cam", "type"],                         saves_log),
            ("diffs.csv",  ["mono_s", "ts_msk", "cam", "diff"],                         diffs_log),
            ("pts.csv",    ["mono_s", "ts_msk", "url_id", "pts_ms"],                    pts_log),
        ]:
            if _log:
                with open(day_dir / _name, "w", newline="", encoding="utf-8") as _f:
                    _w = csv.writer(_f)
                    _w.writerow(_hdr)
                    _w.writerows(_log)
                if clear:
                    _log.clear()
        if clear:
            logger.info(f"  [день] {date_str} → статистика → {day_dir}")

    deadline = (time.monotonic() + args.duration) if args.duration > 0 else None
    _last_csv_save = t_start
    _last_stats_save = t_start
    # frames/diffs/pts на диск реже, чем cpu (тяжелее): раз в 5 мин или 5×интервал cpu
    _STATS_SAVE_INTERVAL = max(300.0, args.csv_save_interval * 5)

    # ЧИСТОЕ время ОБРАБОТКИ кадра (gray+diff, без ожидания RTSP-чтения) — скользящее окно;
    # его min/avg/max пишем в строки save/heartbeat как 'счёт/кадр: … мс' (как yolo/classify/identify).
    _proc_ms: list[float] = []
    _PROC_WINDOW = 600          # последние ~50с при ~12 fps

    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                logger.info(f"Длительность {args.duration:.0f} сек истекла — останов.")
                break

            _today = ts_for_file()[:8]
            if _today != _current_date:
                # Смена суток: закрыть день и ЗАВЕСТИ НОВЫЙ meta/<день>/. Раньше meta не роллилась —
                # прогон через полночь писал run.log/cpu.csv в каталог даты старта (каталог за новый
                # день не появлялся), из-за чего статус терял сегодняшние строки.
                _flush_day_stats(_current_date)              # статистика дня → images/<день>/ (+ очистка логов)
                try:                                         # дописать cpu.csv закрытого дня в его meta/
                    _save_cpu_csv(_cpu_for_day(cpu_monitor.snapshot(), _current_date), meta_dir)
                except Exception as _e:
                    logger.warning(f"  [день] cpu.csv закрытого дня не сохранён: {_e}")
                _current_date = _today
                meta_dir = _base / "meta" / _today           # новый каталог дня
                meta_dir.mkdir(parents=True, exist_ok=True)
                try:                                         # перенаправить run.log в новый каталог
                    logging.getLogger().removeHandler(_run_log_fh)
                    _run_log_fh.close()
                except Exception:
                    pass
                _run_log_fh = add_file_handler(meta_dir / 'run.log')
                logger.info(f"  [день] новый meta-каталог: {meta_dir}")

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
                        logger.warning(f"  [!] {_MAX_LOW_IMPLAUSIBLE} битых кадров — переподключение")
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
                        _imwrite(_cam_dir(vn) / "diff" / fname, frame_u)
                        frame_log[-1][5] = "diff"
                        saves_log.append([round(_now - t_start, 4), _ts_str, vn, "diff"])
                        _interval = _now - last_save_time.get(vn, _now)
                        last_save_time[vn] = _now
                        logger.info(f"  {fname}  diff={diff:.2f}  Готово. Время: {_interval:.1f} с.{_compute_per_frame_log(_proc_ms)}")

                # чистое время ОБРАБОТКИ этого кадра (gray+diff, без ожидания чтения) → окно для счёт/кадр
                _proc_ms.append((time.monotonic() - _now) * 1000.0)
                if len(_proc_ms) > _PROC_WINDOW:
                    del _proc_ms[:-_PROC_WINDOW]

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
                    _imwrite(_cam_dir(vn) / hb_name, hb)
                    frame_log[-1][5] = "heartbeat"
                    saves_log.append([round(time.monotonic() - t_start, 4), ts_for_file(), vn, "heartbeat"])
                    _hb_interval = now - last_save_time.get(vn, now)
                    last_save_time[vn] = now
                    logger.info(f"  пульс {hb_name}  Готово. Время: {_hb_interval:.1f} с.{_compute_per_frame_log(_proc_ms)}")

            if args.csv_save_interval > 0:
                _now = time.monotonic()
                if _now - _last_csv_save >= args.csv_save_interval:
                    _save_cpu_csv(_cpu_for_day(cpu_monitor.snapshot(), _current_date), meta_dir)
                    _last_csv_save = _now
                if _now - _last_stats_save >= _STATS_SAVE_INTERVAL:
                    _flush_day_stats(_current_date, clear=False)  # без обнуления
                    _last_stats_save = _now

    except KeyboardInterrupt:
        logger.info("Останов по Ctrl+C")
        logger.info("Сохранение статистики — не нажимайте Ctrl+C повторно…")
    finally:
        for r in readers.values():
            try:
                r.stop()
            except KeyboardInterrupt:
                pass
        for w in rtcp_workers.values():
            try:
                w.stop()
            except KeyboardInterrupt:
                pass
        try:
            cpu_log = cpu_monitor.stop()
        except KeyboardInterrupt:
            cpu_log = cpu_monitor.snapshot()

        logger.info("Сохранение результатов…")

        if frame_log:
            with open(meta_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "url_id", "ok", "plausible", "event"])
                writer.writerows(frame_log)
            logger.info(f"  frames.csv: {len(frame_log)} строк")

        if saves_log:
            with open(meta_dir / "saves.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cam", "type"])
                writer.writerows(saves_log)
            logger.info(f"  saves.csv:  {len(saves_log)} записей")

        if diffs_log:
            with open(meta_dir / "diffs.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "cam", "diff"])
                writer.writerows(diffs_log)
            logger.info(f"  diffs.csv:  {len(diffs_log)} записей")

        if pts_log:
            with open(meta_dir / "pts.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["mono_s", "ts_msk", "url_id", "pts_ms"])
                writer.writerows(pts_log)
            logger.info(f"  pts.csv:    {len(pts_log)} записей")

        _cpu_cur = _cpu_for_day(cpu_log, _current_date)   # финальный meta = каталог последнего дня
        _save_cpu_csv(_cpu_cur, meta_dir)

        _save_charts(frame_log, _cpu_cur, saves_log, diffs_log, threshold, meta_dir)
        _save_pts_chart(pts_log, _cpu_cur, meta_dir)
        _save_run_stats(frame_log, pts_log, saves_log, diffs_log, meta_dir)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
