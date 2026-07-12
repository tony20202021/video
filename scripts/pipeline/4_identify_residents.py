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

from common.utils.classes import EXTRA_DATASET_DIRS, GROUP_CLASSES, RESIDENT_CLASS

GUEST_CLASS = "4_guest"
# Классы, кропы которых подаются на идентификацию
_IDENTIFY_CLASSES = (RESIDENT_CLASS, GUEST_CLASS)
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
UNKNOWN_CLASS  = "unknown_resident"




def _has_identify_crops(run_dir: Path) -> bool:
    """Проверяет наличие кропов для идентификации (resident + guest) в run_dir.

    Форматы:
      1. Старый (camera-run): run_dir/.../classified/<class>/*.jpg
      2. Новый (inference):   run_dir/<class>/*.jpg
    """
    try:
        for cls in _IDENTIFY_CLASSES:
            for pattern in (f"classified/{cls}", cls):
                for d in run_dir.rglob(pattern) if "/" in pattern else [run_dir / pattern]:
                    if d.is_dir() and any(f.suffix.lower() == ".jpg"
                                          for f in d.iterdir() if f.is_file()):
                        return True
    except OSError:
        pass
    return False


# Оставляем старое имя как алиас для совместимости
_has_resident_crops = _has_identify_crops


def _find_identify_crops(run_dir: Path) -> tuple[list[tuple[Path, Path]], bool]:
    """Возвращает ([(crop_path, rel_cam_dir)], flat_format).

    flat_format=True  → плоская структура inference (run_dir/<class>/*.jpg)
    flat_format=False → camera-run структура (run_dir/.../classified/<class>/*.jpg)
    """
    results: list[tuple[Path, Path]] = []
    seen: set[Path] = set()

    # Старый camera-run формат: classified/<class>/
    patterns = []
    for cls in _IDENTIFY_CLASSES:
        patterns += [f"classified/{cls}"]
    patterns += ["classified/resident"]  # устаревшее имя

    try:
        for pattern in patterns:
            for cls_dir in sorted(run_dir.rglob(pattern)):
                if not cls_dir.is_dir() or cls_dir in seen:
                    continue
                seen.add(cls_dir)
                cam_dir = cls_dir.parent.parent
                try:
                    rel_cam = cam_dir.relative_to(run_dir)
                except ValueError:
                    continue
                for crop in sorted(cls_dir.iterdir()):
                    if crop.suffix.lower() == ".jpg":
                        results.append((crop, rel_cam))
    except OSError:
        pass

    if results:
        return results, False

    # Новый плоский формат inference: run_dir/<class>/*.jpg
    try:
        for cls_name in list(_IDENTIFY_CLASSES) + ["resident"]:
            cls_dir = run_dir / cls_name
            if not cls_dir.is_dir() or cls_dir in seen:
                continue
            seen.add(cls_dir)
            for crop in sorted(cls_dir.iterdir()):
                if crop.is_file() and crop.suffix.lower() == ".jpg":
                    results.append((crop, Path(".")))
    except OSError:
        pass

    return results, True


def _find_resident_crops(run_dir: Path) -> list[tuple[Path, Path]]:
    """Обратная совместимость: возвращает только список кропов без flat-флага."""
    crops, _ = _find_identify_crops(run_dir)
    return crops


# ─── ML ──────────────────────────────────────────────────────────────────────

def _load_identifier(model_path: Path | None = None):
    """Загружает PersonIdentifier из явного пути или из IDENTIFY_MODEL (.env)."""
    if model_path is not None:
        path = model_path
    else:
        identify_path = os.environ.get("IDENTIFY_MODEL", "").strip()
        if not identify_path:
            logger.warning("  [ML] IDENTIFY_MODEL не задан в .env")
            return None
        path = Path(identify_path)
        if not path.is_absolute():
            path = REPO_ROOT / path

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
    """Возвращает (person_id, id_conf, out_class, probs_dict).

    out_class — person_id если уверенность >= identify_conf, иначе unknown_resident.
    """
    if ident is None:
        return None, 0.0, UNKNOWN_CLASS, {}
    person_id, id_conf, probs = ident.identify_with_probs(bgr_crop, threshold=identify_conf)
    out_class = person_id if person_id else UNKNOWN_CLASS
    return person_id, id_conf, out_class, probs


# ─── Per-date output files ───────────────────────────────────────────────────

def _save_date_outputs(run_id_log: list[dict], inference_date_dir: Path) -> None:
    """Сохраняет выходные файлы М2 в директорию inference.

    identifications.csv → inference_date_dir  (вероятности по всем классам М2)
    labels.json         → inference_date_dir  (для label_ui)
    """
    if not run_id_log:
        return

    inference_date_dir.mkdir(parents=True, exist_ok=True)

    # identifications.csv — в директорию М2 inference, накапливается между запусками
    prob_cols = sorted(k for k in run_id_log[0] if k.startswith("p_"))
    fieldnames = (
        ["mono_s", "ts_epoch", "run_name", "cam", "crop",
         "person_id", "id_conf", "conf_2nd", "margin"]
        + prob_cols
        + ["out_class"]
    )
    csv_path = inference_date_dir / "identifications.csv"
    existing_crops: set[str] = set()
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    existing_crops.add(row["crop"])
        except Exception:
            pass
    new_rows = [r for r in run_id_log if r["crop"] not in existing_crops]
    write_header = not csv_path.exists() or not existing_crops
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(new_rows)
    total_csv = len(existing_crops) + len(new_rows)
    logger.info(f"identifications.csv → {csv_path}  (+{len(new_rows)}, итого ~{total_csv})")


    # labels.json in inference output dir — built from subdir structure for label_ui review
    if inference_date_dir and inference_date_dir.is_dir():
        inf_labels: dict[str, str] = {}
        for person_dir in sorted(inference_date_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            label = "" if person_dir.name == UNKNOWN_CLASS else person_dir.name
            for f in sorted(person_dir.glob("*.jpg")):
                inf_labels[str(f)] = label
        inf_labels_path = inference_date_dir / "labels.json"
        inf_labels_path.write_text(
            _json.dumps({"version": 1, "labels": inf_labels}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"labels.json          → {inf_labels_path}  ({len(inf_labels)} записей)")


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
                        help="Явный путь к ONNX PersonIdentifier (иначе из IDENTIFY_MODEL в .env)")
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

    ident = _load_identifier(args.model)
    if ident is not None:
        status = f"готов, классов: {ident.person_count}" if ident.ready else "не загружен"
        logger.info(f"  [ML] PersonIdentifier: {status}")
    else:
        logger.info(f"  [ML] Идентификатор не загружен — все кропы → {UNKNOWN_CLASS}/")

    if not args.input_dir.is_dir():
        logger.warning(f"[!] Не найдено: {args.input_dir}")
        return 1
    run_pairs = [(args.input_dir, "")]

    # Определяем формат входных данных
    _, flat_format = _find_identify_crops(args.input_dir)

    # Пути вывода зависят от формата
    _base_out = args.output or DEFAULT_OUTPUT
    _run_name = args.input_dir.name   # дата (20260709) или имя run-dir
    _run_ts   = ts_for_dir()
    if flat_format:
        # Новый формат: images/<date>/<person_id>/ + meta/<date>/<run_ts>/
        images_root = _base_out / "images"
        meta_dir    = _base_out / "meta" / _run_name / _run_ts
        out_dir     = meta_dir   # логи, CSV, charts — сюда
    else:
        # Старый формат: всё в out_dir/run_<ts>/
        out_dir     = _base_out / f"run_{_run_ts}"
        images_root = out_dir
        meta_dir    = out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

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
        "identify_model": os.environ.get("IDENTIFY_MODEL", ""),
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
    logger.info(f"Формат:    {'плоский (inference)' if flat_format else 'camera-run'}")
    logger.info(f"Прогонов:  {len(run_pairs)}")
    for rd, ps in run_pairs:
        prefix = f"{ps}/" if ps else ""
        crops, _ = _find_identify_crops(rd)
        logger.info(f"  {prefix}{rd.name}  [{len(crops)} кропов]")
    logger.info('')

    grand_total = 0
    grand_skipped = 0
    grand_identified = defaultdict(int)

    import cv2

    for run_dir, parent_stem in run_pairs:
        run_name = run_dir.name
        run_id_log: list[dict] = []
        _ps = parent_stem or run_dir.parent.name
        label = f"{_ps}/{run_name}"
        logger.info(f"── {label} ──────────────────────────────────────")

        crops, _ = _find_identify_crops(run_dir)
        if not crops:
            logger.info(f"  [!] Нет кропов в {run_dir}")
            continue

        logger.info(f"  Кропов: {len(crops)}")

        for crop_path, rel_cam in crops:
            t0 = time.monotonic()

            if flat_format:
                # Новый формат: images/<date>/<person_id>/
                # Проверяем, не идентифицирован ли уже (в любом person_id подкаталоге)
                date_out = images_root / run_name
                if date_out.is_dir() and any(
                    (sub / crop_path.name).exists()
                    for sub in date_out.iterdir() if sub.is_dir()
                ):
                    grand_skipped += 1
                    continue
            else:
                rel_cam_dest = out_dir / _ps / run_name / rel_cam / "classified"
                if any(
                    (rel_cam_dest / cls / crop_path.name).exists()
                    for cls in rel_cam_dest.iterdir() if rel_cam_dest.is_dir() and cls.is_dir()
                ) if rel_cam_dest.is_dir() else False:
                    grand_skipped += 1
                    continue

            bgr = cv2.imread(str(crop_path))
            if bgr is None:
                continue

            person_id, id_conf, out_class, probs = _identify(
                ident, bgr, identify_conf=args.identify_conf
            )
            identify_ms = (time.monotonic() - t0) * 1000
            timing_log.append([round(t0 - t_start, 3), round(identify_ms, 1)])

            if flat_format:
                dest = images_root / run_name / out_class
            else:
                dest = images_root / _ps / run_name / rel_cam / "classified" / out_class
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(crop_path, dest / crop_path.name)

            ts_ep = _crop_ts_epoch(crop_path)
            sorted_probs = sorted(probs.values(), reverse=True)
            conf_2nd = round(sorted_probs[1], 4) if len(sorted_probs) > 1 else 0.0
            entry = {
                "mono_s":    round(t0 - t_start, 3),
                "ts_epoch":  ts_ep,
                "run_name":  run_name,
                "cam":       str(rel_cam),
                "crop":      crop_path.name,
                "crop_abs":  str(crop_path),
                "person_id": person_id or "",
                "id_conf":   round(id_conf, 4),
                "conf_2nd":  conf_2nd,
                "margin":    round(id_conf - conf_2nd, 4),
                "out_class": out_class,
            }
            entry.update({f"p_{cls}": p for cls, p in probs.items()})
            run_id_log.append(entry)
            id_log.append(entry)
            grand_identified[out_class] += 1
            grand_total += 1

        logger.info(f"  → обработано: {len(crops) - grand_skipped}  пропущено (уже есть): {grand_skipped}")
        logger.info('')

        # Save identifications.csv and labels.json into the input date directory
        if run_id_log:
            inf_date_dir = (images_root / run_name) if flat_format else None
            if inf_date_dir is not None:
                _save_date_outputs(run_id_log, inf_date_dir)

    _periodic_stop.set()
    cpu_log = cpu_monitor.stop()

    if id_log:
        csv_path = meta_dir / "identifications.csv"
        meta_dir.mkdir(parents=True, exist_ok=True)
        # Build fieldnames dynamically (prob columns depend on model classes)
        prob_cols = [k for k in id_log[0] if k.startswith("p_")]
        fieldnames = ["mono_s", "ts_epoch", "run_name", "cam", "crop",
                      "person_id", "id_conf", "conf_2nd", "margin"] + prob_cols + ["out_class"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(id_log)
        logger.info(f"identifications.csv: {len(id_log)} записей → {csv_path}")
    else:
        logger.info("Идентификаций не найдено.")

    _save_cpu_csv(cpu_log, meta_dir)

    if timing_log:
        with open(meta_dir / "identify_timing.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mono_s", "identify_ms"])
            w.writerows(timing_log)

    _save_timeline_chart(id_log, meta_dir / "timeline_chart.png")
    _save_cpu_chart(cpu_log, meta_dir / "cpu_chart.png", timing_log or None)

    stats = {
        "input_runs":           [f"{ps}/{rd.name}" if ps else str(rd) for rd, ps in run_pairs],
        "crops_total":          grand_total,
        "identified_by_person": dict(grand_identified),
        "ml_active":            ident is not None,
        "duration_sec":         round(time.monotonic() - t_start, 1),
    }
    (meta_dir / "run_stats.json").write_text(
        _json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"Готово. Время: {stats['duration_sec']} с.  Вывод: {_base_out}")
    summary = "  ".join(f"{p}: {n}" for p, n in sorted(grand_identified.items()))
    if summary:
        logger.info(f"Итог: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
