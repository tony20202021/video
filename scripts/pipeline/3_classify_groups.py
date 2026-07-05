"""
Офлайн классификация кропов по группам (Модель 1 — GroupClassifier).

Принимает выходные каталоги скрипта 5_2 (кропы уже вырезаны YOLO),
прогоняет GroupClassifier. YOLO и PersonIdentifier не используются.

Входные данные:
  - Каталог прогона 5_2:   .output/cameras/5_2_yolo_boxes_files/run_<ts>
  - Родительский каталог:  .output/cameras/5_2_yolo_boxes_files/  (все run_*)
  Можно смешивать.

Выход:
  .output/cameras/6_2_classify_groups_files/run_<ts>/
    <parent_stem>/
      <5_2_run_name>/
        <sub_run>/
          <cam>/
            classified/
              resident/
              courier/
              delivery/
              utilities/
              other/
              uncertain/     — уверенность ниже classify_conf
              unknown/       — модель не загружена
    classifications.csv
    timeline_chart.png
    cpu_chart.png
    run_stats.json
    run.log

Usage:
    python scripts/cameras/6_2_classify_groups_files.py .output/cameras/5_2_yolo_boxes_files
    python scripts/cameras/6_2_classify_groups_files.py run_dir1 run_dir2
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

from common.utils.atomic import copy as _copy
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

DEFAULT_OUTPUT = REPO_ROOT / ".output" / "pipeline" / "3_classify_groups"
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

_CLASS_COLORS = {
    "1_resident":  "#228833",
    "2_delivery":  "#0055cc",
    "3_utilities": "#770077",
    "4_guest":     "#cc7700",
    "uncertain":   "#dddddd",
    "unknown":     "#aaaaaa",
}




def _find_crops(run_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Возвращает {sub_run_name: {cam_name: [crop_path, ...]}}.

    Перебирает все jpg-файлы в run_dir рекурсивно без фильтров по именам каталогов.
    Группировка: первый уровень вложенности → sub_run, второй → cam.
    """
    result: dict[str, dict[str, list[Path]]] = {}
    try:
        for img in sorted(run_dir.rglob("*.jpg")):
            rel = img.relative_to(run_dir)
            parts = rel.parts
            if len(parts) >= 3:
                sub_run, cam = parts[0], parts[1]
            elif len(parts) == 2:
                sub_run, cam = parts[0], "_"
            else:
                sub_run, cam = "_", "_"
            result.setdefault(sub_run, {}).setdefault(cam, []).append(img)
    except OSError:
        pass
    return result


# ─── ML ──────────────────────────────────────────────────────────────────────

def _load_classifier(config_path: Path):
    """Загружает GroupClassifier из config.yaml."""
    if not config_path.is_file():
        return None
    try:
        import yaml
    except ImportError:
        logger.warning("  [ML] Нужен pyyaml: pip install pyyaml")
        return None
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"  [ML] Ошибка чтения {config_path}: {e}")
        return None
    models = cfg.get("models", {})
    classify_path = models.get("classify")
    if not classify_path:
        logger.warning("  [ML] config.yaml: нет models.classify")
        return None
    try:
        from ml.classify import GroupClassifier
        clf = GroupClassifier()
        if not clf.load(Path(classify_path)):
            logger.warning(f"  [ML] Не удалось загрузить: {classify_path}")
            return None
        return clf
    except Exception as e:
        logger.warning(f"  [ML] Ошибка инициализации: {e}")
        return None


def _classify(clf, bgr_crop, *, classify_conf: float):
    """Возвращает (group_class, group_conf, out_class).

    out_class — имя подкаталога: group_class если уверенность >= classify_conf,
    иначе 'uncertain'.
    """
    if clf is None:
        return "unknown", 0.0, "unknown"
    group_class, group_conf, _ = clf.classify(bgr_crop)
    out_class = group_class if group_conf >= classify_conf else "uncertain"
    return group_class, group_conf, out_class


# ─── Timestamp from crop filename ────────────────────────────────────────────

def _crop_ts_epoch(crop_path: Path) -> float | None:
    """Парсит дату-время из имени файла кропа (формат 5_2).

    Пример: cam_01_9_d_20260628_143742_559207_msk_diff5.1_p1of1_conf0.49.jpg
    """
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

def _save_timeline_chart(class_log: list[dict], out_path: Path) -> None:
    if not class_log:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    by_key: dict[str, dict[str, list[float]]] = {}
    for row in class_log:
        if not row.get("ts_epoch"):
            continue
        key = f"{row['sub_run']}/{row['cam']}"
        entity = row.get("out_class", "unknown")
        by_key.setdefault(key, {}).setdefault(entity, []).append(row["ts_epoch"])

    if not by_key:
        return

    keys = sorted(by_key)
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(16, max(n * 2.5, 4)), squeeze=False)
    fig.suptitle("6_2 — классификация по группам", fontsize=11)

    for i, key in enumerate(keys):
        ax = axes[i][0]
        entities = sorted(by_key[key])
        y_map = {e: j + 1 for j, e in enumerate(entities)}
        for entity, times in by_key[key].items():
            color = _CLASS_COLORS.get(entity, "#228833")
            ax.scatter(times, [y_map[entity]] * len(times),
                       color=color, s=25, alpha=0.8, label=f"{entity} ({len(times)})")
        ax.set_title(key, fontsize=8)
        ax.set_yticks(list(y_map.values()))
        ax.set_yticklabels(list(y_map.keys()), fontsize=7)
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[-1][0].set_xlabel("время (epoch)")
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
    _draw_cpu_on_ax(ax, cpu_log, title="6_2 — загрузка ЦПУ")
    if not has_timing:
        ax.set_xlabel("время от старта, с")

    if has_timing:
        ax2  = axes[1][0]
        ts   = [r[0] for r in timing_log]
        inf  = [r[1] for r in timing_log]
        avg  = sum(inf) / len(inf) if inf else 0
        ax2.bar(ts, inf, width=0.3, color="#2255cc", alpha=0.7,
                label=f"classify  avg {avg:.0f}ms")
        ax2.scatter(ts, inf, s=6, color="#2255cc", zorder=5)
        ax2.set_ylabel("мс / кроп")
        ax2.set_xlabel("время от старта, с")
        ax2.set_title("ML: время классификации кропа")
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
        description="Офлайн классификация кропов по группам (Модель 1)"
    )
    parser.add_argument("input_dir", type=Path,
                        help="Конкретный run-каталог 2_yolo_boxes_files")
    parser.add_argument("--config",        type=Path, default=DEFAULT_CONFIG,
                        help="config.yaml с путями к ML-моделям")
    parser.add_argument("--classify-conf", type=float, default=0.65, metavar="CONF",
                        help="Порог GroupClassifier (default: 0.65)")
    parser.add_argument("--output",        type=Path, default=None)
    parser.add_argument("--cpu-interval",  type=float, default=2.0)
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

    clf = _load_classifier(args.config)
    if clf is not None:
        logger.info(f"  [ML] GroupClassifier: {'готов' if clf.ready else 'не загружен'}")
    else:
        logger.info("  [ML] Классификатор не загружен — все кропы → unknown/")

    if not args.input_dir.is_dir():
        logger.warning(f"[!] Не найдено: {args.input_dir}")
        return 1
    run_pairs = [(args.input_dir, "")]

    out_dir = args.output or (DEFAULT_OUTPUT / f"run_{ts_for_dir()}")
    out_dir.mkdir(parents=True, exist_ok=True)


    add_file_handler(out_dir / 'run.log')

    t_start     = time.monotonic()
    cpu_monitor = _CpuMonitor(interval=max(args.cpu_interval, 0.5))
    cpu_active  = args.cpu_interval > 0 and cpu_monitor.start(t_start)

    timing_log: list[list] = []   # [mono_s, classify_ms]
    class_log:  list[dict] = []   # одна запись на кроп

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
        "script":        "6_2_classify_groups_files",
        "inputs":        [str(args.input_dir)],
        "config":        str(args.config),
        "ml_active":     clf is not None,
        "classify_conf": args.classify_conf,
        "cpu_interval":  args.cpu_interval,
        "out_dir":       str(out_dir),
    }
    (out_dir / "run_params.json").write_text(
        _json.dumps(run_params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"ML:        {'GroupClassifier готов' if clf and clf.ready else 'не загружен (unknown/)'}")
    logger.info(f"Порог M1:  {args.classify_conf}")
    logger.info(f"Вывод:     {out_dir}")
    logger.info(f"Прогонов:  {len(run_pairs)}")
    for rd, ps in run_pairs:
        prefix = f"{ps}/" if ps else ""
        crops_by_subrun = _find_crops(rd)
        total = sum(len(cs) for cams in crops_by_subrun.values() for cs in cams.values())
        logger.info(f"  {prefix}{rd.name}  [{total} кропов]")
    logger.info('')

    grand_total = 0
    grand_classified = defaultdict(int)

    import cv2

    for run_dir, parent_stem in run_pairs:
        run_name = run_dir.name
        _ps = parent_stem or run_dir.parent.name
        label = f"{_ps}/{run_name}"
        logger.info(f"── {label} ──────────────────────────────────────")

        crops_by_subrun = _find_crops(run_dir)
        if not crops_by_subrun:
            logger.info(f"  [!] Нет кропов в {run_dir}")
            continue

        for sub_run_name, cams in sorted(crops_by_subrun.items()):
            logger.info(f"  {sub_run_name}")
            for cam_name, crop_paths in sorted(cams.items()):
                classified_base = (
                    out_dir / run_name / sub_run_name / cam_name / "classified"
                )
                n_cam = 0
                logger.info(f"    {cam_name}: {len(crop_paths)} кропов")

                for crop_path in crop_paths:
                    bgr = cv2.imread(str(crop_path))
                    if bgr is None:
                        continue

                    t0 = time.monotonic()
                    group, group_conf, out_class = _classify(
                        clf, bgr,
                        classify_conf=args.classify_conf,
                    )
                    classify_ms = (time.monotonic() - t0) * 1000
                    timing_log.append([round(t0 - t_start, 3), round(classify_ms, 1)])

                    dest = classified_base / out_class
                    dest.mkdir(parents=True, exist_ok=True)
                    _copy(crop_path, dest / crop_path.name)

                    ts_ep = _crop_ts_epoch(crop_path)
                    class_log.append({
                        "mono_s":     round(t0 - t_start, 3),
                        "ts_epoch":   ts_ep,
                        "run_name":   run_name,
                        "sub_run":    sub_run_name,
                        "cam":        cam_name,
                        "crop":       crop_path.name,
                        "group":      group,
                        "group_conf": round(group_conf, 3),
                        "out_class":  out_class,
                    })
                    grand_classified[out_class] += 1
                    n_cam += 1

                grand_total += n_cam
                logger.info(f"    → классифицировано: {n_cam}")

        logger.info('')

    _periodic_stop.set()
    cpu_log = cpu_monitor.stop()

    if class_log:
        csv_path = out_dir / "classifications.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "mono_s", "ts_epoch", "run_name", "sub_run", "cam", "crop",
                "group", "group_conf", "out_class",
            ])
            w.writeheader()
            w.writerows(class_log)
        logger.info(f"classifications.csv: {len(class_log)} записей → {csv_path}")
    else:
        logger.info("Классификаций не найдено.")

    _save_cpu_csv(cpu_log, out_dir)

    if timing_log:
        with open(out_dir / "classify_timing.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mono_s", "classify_ms"])
            w.writerows(timing_log)

    _save_timeline_chart(class_log, out_dir / "timeline_chart.png")
    _save_cpu_chart(cpu_log, out_dir / "cpu_chart.png", timing_log or None)

    stats = {
        "input_runs":          [f"{ps}/{rd.name}" if ps else str(rd) for rd, ps in run_pairs],
        "crops_total":         grand_total,
        "classified_by_class": dict(grand_classified),
        "ml_active":           clf is not None,
        "duration_sec":        round(time.monotonic() - t_start, 1),
    }
    (out_dir / "run_stats.json").write_text(
        _json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


    logger.info(f"Готово. Время: {stats['duration_sec']} с.  Вывод: {out_dir}")
    summary = "  ".join(f"{cls}: {n}" for cls, n in sorted(grand_classified.items()))
    if summary:
        logger.info(f"Итог: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
