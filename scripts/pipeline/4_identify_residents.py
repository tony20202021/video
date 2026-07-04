"""
Офлайн идентификация жителей из кропов 6_2 (Модель 2 — PersonIdentifier).

Принимает выходные каталоги скрипта 6_2, находит кропы в classified/resident/,
прогоняет PersonIdentifier. GroupClassifier не используется.

Входные данные:
  - Каталог прогона 3_classify_groups:  .output/pipeline/3_classify_groups/run_<ts>

Выход:
  .output/cameras/6_3_identify_residents_files/run_<ts>/
    <структура из 6_2>/
      classified/
        <person_id>/       — идентифицированный житель
        unknown_resident/  — уверенность ниже identify_conf
    identifications.csv
    timeline_chart.png
    cpu_chart.png
    run_stats.json
    run.log

Usage:
    python scripts/pipeline/4_identify_residents.py .output/pipeline/3_classify_groups/run_20260629_210753_msk
"""

from __future__ import annotations

import argparse
import csv
import json as _json
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.camera_run import (
    CpuMonitor as _CpuMonitor,
    save_cpu_csv as _save_cpu_csv,
    draw_cpu_on_ax as _draw_cpu_on_ax,
)
from common.utils.time_msk import ts_for_dir
import logging
from common.utils.log_setup import setup_logging, add_file_handler

logger = logging.getLogger(__name__)

MSK = timezone(timedelta(hours=3))

DEFAULT_OUTPUT = REPO_ROOT / ".output" / "pipeline" / "4_identify_residents"
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
UNKNOWN_CLASS  = "unknown_resident"




def _find_resident_crops(run_dir: Path) -> list[tuple[Path, Path]]:
    """Возвращает [(crop_path, rel_cam_dir)] из classified/resident/ в 6_2 run.

    rel_cam_dir — путь до каталога камеры относительно run_dir.
    """
    results: list[tuple[Path, Path]] = []
    try:
        for resident_dir in sorted(run_dir.rglob("classified/resident")):
            if not resident_dir.is_dir():
                continue
            cam_dir = resident_dir.parent.parent  # .../cam/classified/resident → .../cam
            try:
                rel_cam = cam_dir.relative_to(run_dir)
            except ValueError:
                continue
            for crop in sorted(resident_dir.iterdir()):
                if crop.suffix.lower() == ".jpg":
                    results.append((crop, rel_cam))
    except OSError:
        pass
    return results


# ─── ML ──────────────────────────────────────────────────────────────────────

def _load_identifier(model_path: Path | None, config_path: Path):
    """Загружает PersonIdentifier из явного пути или из config.yaml."""
    if model_path is not None:
        path = model_path
    else:
        # Пробуем взять из config.yaml
        if not config_path.is_file():
            return None
        try:
            import yaml
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            identify_path = cfg.get("models", {}).get("identify")
            if not identify_path:
                logger.warning("  [ML] config.yaml: нет models.identify")
                return None
            path = Path(identify_path)
        except Exception as e:
            logger.warning(f"  [ML] Ошибка чтения {config_path}: {e}")
            return None

    try:
        from ml.identify import PersonIdentifier
        ident = PersonIdentifier()
        if not ident.load(path):
            logger.warning(f"  [ML] Не удалось загрузить: {path}")
            return None
        return ident
    except Exception as e:
        logger.warning(f"  [ML] Ошибка инициализации PersonIdentifier: {e}")
        return None


def _identify(ident, bgr_crop, *, identify_conf: float):
    """Возвращает (person_id, id_conf, out_class).

    out_class — person_id если уверенность >= identify_conf, иначе unknown_resident.
    """
    if ident is None:
        return None, 0.0, UNKNOWN_CLASS
    person_id, id_conf = ident.identify(bgr_crop, threshold=identify_conf)
    out_class = person_id if person_id else UNKNOWN_CLASS
    return person_id, id_conf, out_class


# ─── Timestamp from crop filename ────────────────────────────────────────────

def _crop_ts_epoch(crop_path: Path) -> float | None:
    stem = crop_path.stem
    for part in stem.split("_"):
        if len(part) == 8 and part.isdigit():
            idx = stem.index(part)
            rest = stem[idx:]
            sub = rest.split("_")
            if len(sub) >= 2 and len(sub[1]) == 6 and sub[1].isdigit():
                try:
                    d, t = sub[0], sub[1]
                    dt = datetime(int(d[:4]), int(d[4:6]), int(d[6:]),
                                  int(t[:2]), int(t[2:4]), int(t[4:6]),
                                  tzinfo=MSK)
                    return dt.timestamp()
                except ValueError:
                    pass
    return None


# ─── Charts ──────────────────────────────────────────────────────────────────

def _save_timeline_chart(id_log: list[dict], out_path: Path) -> None:
    if not id_log:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    by_person: dict[str, list[float]] = {}
    for row in id_log:
        if not row.get("ts_epoch"):
            continue
        entity = row.get("out_class", UNKNOWN_CLASS)
        by_person.setdefault(entity, []).append(row["ts_epoch"])

    if not by_person:
        return

    fig, ax = plt.subplots(1, 1, figsize=(16, max(len(by_person) * 0.8 + 2, 4)))
    fig.suptitle("6_3 — идентификация жителей", fontsize=11)

    persons = sorted(by_person)
    y_map = {p: i + 1 for i, p in enumerate(persons)}
    colors = plt.cm.tab20.colors  # type: ignore[attr-defined]

    for i, person in enumerate(persons):
        times = by_person[person]
        color = colors[i % len(colors)]
        ax.scatter(times, [y_map[person]] * len(times),
                   color=color, s=25, alpha=0.8, label=f"{person} ({len(times)})")

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()), fontsize=7)
    ax.set_xlabel("время (epoch)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110)
    plt.close()
    logger.info(f"  timeline_chart.png → {out_path}")


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

    ax = axes[0][0]
    _draw_cpu_on_ax(ax, cpu_log, title="6_3 — загрузка ЦПУ")
    if not has_timing:
        ax.set_xlabel("время от старта, с")

    if has_timing:
        ax2  = axes[1][0]
        ts   = [r[0] for r in timing_log]
        inf  = [r[1] for r in timing_log]
        avg  = sum(inf) / len(inf) if inf else 0
        ax2.bar(ts, inf, width=0.3, color="#338833", alpha=0.7,
                label=f"identify  avg {avg:.0f}ms")
        ax2.scatter(ts, inf, s=6, color="#338833", zorder=5)
        ax2.set_ylabel("мс / кроп")
        ax2.set_xlabel("время от старта, с")
        ax2.set_title("ML: время идентификации кропа")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=110)
    plt.close()
    logger.info(f"  cpu_chart.png → {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Офлайн идентификация жителей из кропов 6_2 (Модель 2)"
    )
    parser.add_argument("input_dir", type=Path,
                        help="Конкретный run-каталог 3_classify_groups")
    parser.add_argument("--model",          type=Path, default=None,
                        help="Явный путь к ONNX PersonIdentifier (иначе из config.yaml)")
    parser.add_argument("--config",         type=Path, default=DEFAULT_CONFIG,
                        help="config.yaml с путями к ML-моделям")
    parser.add_argument("--identify-conf",  type=float, default=0.70, metavar="CONF",
                        help="Порог PersonIdentifier (default: 0.70)")
    parser.add_argument("--output",         type=Path, default=None)
    parser.add_argument("--cpu-interval",   type=float, default=2.0)
    args = parser.parse_args()

    # ── Single-instance guard ──────────────────────────────────────────────
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
            and not any("conda" in (c or "").lower() for c in (p.info.get("cmdline") or []))
        ]
        if _others:
            logger.warning("[!] %s уже запущен (PID: %s). Завершение.", _this_name, _others)
            return 1
    except ImportError:
        pass

    ident = _load_identifier(args.model, args.config)
    if ident is not None:
        status = f"готов, классов: {ident.person_count}" if ident.ready else "не загружен"
        logger.info(f"  [ML] PersonIdentifier: {status}")
    else:
        logger.info(f"  [ML] Идентификатор не загружен — все кропы → {UNKNOWN_CLASS}/")

    out_dir = args.output or (DEFAULT_OUTPUT / f"run_{ts_for_dir()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_dir.is_dir():
        logger.warning(f"[!] Не найдено: {args.input_dir}")
        return 1
    run_pairs = [(args.input_dir, "")]


    add_file_handler(out_dir / 'run.log')

    t_start     = time.monotonic()
    cpu_monitor = _CpuMonitor(interval=max(args.cpu_interval, 0.5))
    cpu_active  = args.cpu_interval > 0 and cpu_monitor.start(t_start)

    timing_log: list[list] = []   # [mono_s, identify_ms]
    id_log:     list[dict] = []   # одна запись на кроп

    _periodic_stop = threading.Event()

    def _periodic_cpu_save() -> None:
        while not _periodic_stop.wait(timeout=60.0):
            _snap = cpu_monitor.snapshot()
            if not _snap:
                continue
            try:
                _save_cpu_csv(_snap, out_dir)
                _save_cpu_chart(_snap, out_dir / "cpu_chart.png",
                                list(timing_log) or None)
            except Exception:
                pass

    _periodic_thread = threading.Thread(target=_periodic_cpu_save,
                                        daemon=True, name="cpu-periodic")
    if cpu_active:
        _periodic_thread.start()

    run_params = {
        "script":        "6_3_identify_residents_files",
        "inputs":        [str(args.input_dir)],
        "model":         str(args.model) if args.model else None,
        "config":        str(args.config),
        "ml_active":     ident is not None,
        "identify_conf": args.identify_conf,
        "cpu_interval":  args.cpu_interval,
        "out_dir":       str(out_dir),
    }
    (out_dir / "run_params.json").write_text(
        _json.dumps(run_params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"ML:        {'PersonIdentifier готов' if ident and ident.ready else 'не загружен (unknown_resident/)'}")
    logger.info(f"Порог M2:  {args.identify_conf}")
    logger.info(f"Вывод:     {out_dir}")
    logger.info(f"Прогонов:  {len(run_pairs)}")
    for rd, ps in run_pairs:
        prefix = f"{ps}/" if ps else ""
        resident_crops = _find_resident_crops(rd)
        logger.info(f"  {prefix}{rd.name}  [{len(resident_crops)} кропов-жителей]")
    logger.info('')

    grand_total = 0
    grand_identified = defaultdict(int)

    import cv2

    for run_dir, parent_stem in run_pairs:
        run_name = run_dir.name
        _ps = parent_stem or run_dir.parent.name
        label = f"{_ps}/{run_name}"
        logger.info(f"── {label} ──────────────────────────────────────")

        resident_crops = _find_resident_crops(run_dir)
        if not resident_crops:
            logger.info(f"  [!] Нет кропов жителей в {run_dir}")
            continue

        logger.info(f"  Кропов жителей: {len(resident_crops)}")

        for crop_path, rel_cam in resident_crops:
            bgr = cv2.imread(str(crop_path))
            if bgr is None:
                continue

            t0 = time.monotonic()
            person_id, id_conf, out_class = _identify(
                ident, bgr, identify_conf=args.identify_conf
            )
            identify_ms = (time.monotonic() - t0) * 1000
            timing_log.append([round(t0 - t_start, 3), round(identify_ms, 1)])

            dest = out_dir / _ps / run_name / rel_cam / "classified" / out_class
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(crop_path, dest / crop_path.name)

            ts_ep = _crop_ts_epoch(crop_path)
            id_log.append({
                "mono_s":    round(t0 - t_start, 3),
                "ts_epoch":  ts_ep,
                "rel_path":  str(rel_cam),
                "crop":      crop_path.name,
                "person_id": person_id or "",
                "id_conf":   round(id_conf, 3),
                "out_class": out_class,
            })
            grand_identified[out_class] += 1
            grand_total += 1

        logger.info(f"  → обработано: {len(resident_crops)}")
        logger.info('')

    _periodic_stop.set()
    cpu_log = cpu_monitor.stop()

    if id_log:
        csv_path = out_dir / "identifications.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "mono_s", "ts_epoch", "rel_path", "crop",
                "person_id", "id_conf", "out_class",
            ])
            w.writeheader()
            w.writerows(id_log)
        logger.info(f"identifications.csv: {len(id_log)} записей → {csv_path}")
    else:
        logger.info("Идентификаций не найдено.")

    _save_cpu_csv(cpu_log, out_dir)

    if timing_log:
        with open(out_dir / "identify_timing.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mono_s", "identify_ms"])
            w.writerows(timing_log)

    _save_timeline_chart(id_log, out_dir / "timeline_chart.png")
    _save_cpu_chart(cpu_log, out_dir / "cpu_chart.png", timing_log or None)

    stats = {
        "input_runs":           [f"{ps}/{rd.name}" if ps else str(rd) for rd, ps in run_pairs],
        "crops_total":          grand_total,
        "identified_by_person": dict(grand_identified),
        "ml_active":            ident is not None,
        "duration_sec":         round(time.monotonic() - t_start, 1),
    }
    (out_dir / "run_stats.json").write_text(
        _json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


    logger.info(f"\nГотово. Время: {stats['duration_sec']} с.  Вывод: {out_dir}")
    summary = "  ".join(f"{p}: {n}" for p, n in sorted(grand_identified.items()))
    if summary:
        logger.info(f"Итог: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
