"""
Стратегии 1,2,3,5,8: отбор кропов из inference/images/YYYYMMDD/ в новый датасет.

  1 — ошибки: labels.json класс ≠ подкаталог (ручная переразметка)
  2 — uncertain: файл размечен вручную в labels.json
  3 — серая зона: conf < CLASSIFY_CONF_HIGH, есть в labels.json
  5 — псевдо-метки: conf ≥ CLASSIFY_CONF_HIGH, модель права (без ручной проверки)
  8 — margin: conf_top1 − conf_top2 < MARGIN_THRESH, есть в labels.json

Usage:
    python scripts/train/dataset_v2_from_inference.py \\
        .data/groups/v1/inference/images/20260705 \\
        --output .data/groups/v2/dataset

    python scripts/train/dataset_v2_from_inference.py \\
        .data/groups/v1/inference/images/20260705 \\
        --output .data/groups/v2/dataset \\
        --strategies 1,2 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from math import prod
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.utils.log_setup import setup_logging
from common.utils.classes import GROUP_CLASSES

logger = logging.getLogger(__name__)

_VALID_CLASSES = set(GROUP_CLASSES)

MSK = timezone(timedelta(hours=3))
UNCERTAIN_DIR = "uncertain"

_SOURCES_FIELDS = [
    "timestamp", "strategy", "date", "filename",
    "src_path", "dst_path", "true_class", "pred_class",
    "conf", "conf_2nd", "margin",
]


# ── .env helpers ─────────────────────────────────────────────────────────────

def _ef_float(key: str, default: float) -> float:
    val = os.environ.get(key, "")
    try:
        return float(val) if val else default
    except ValueError:
        return default


# ── File helpers ──────────────────────────────────────────────────────────────

def _load_labels(labels_path: Path) -> dict[str, str]:
    if not labels_path.is_file():
        return {}
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    return data.get("labels", {})


def _is_multilabel(labels_path: Path) -> bool:
    """labels.json — формат multi-label (v2 {img:[classes]})? Тогда стратегии здесь неприменимы."""
    if not labels_path.is_file():
        return False
    try:
        data = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    labels = data.get("labels", data)
    return data.get("version") == 2 or (
        isinstance(labels, dict) and any(isinstance(v, list) for v in labels.values()))


def _load_csv(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _auto_find_csv(date_dir: Path) -> Path | None:
    # Накопительный CSV в images/YYYYMMDD/ (предпочтительно)
    cumulative = date_dir / "classifications.csv"
    if cumulative.is_file():
        return cumulative
    # Последний прогон в meta/YYYYMMDD/*/
    inference_root = date_dir.parent.parent
    meta_dir = inference_root / "meta" / date_dir.name
    if meta_dir.is_dir():
        for rd in sorted(meta_dir.iterdir(), reverse=True):
            candidate = rd / "classifications.csv"
            if candidate.is_file():
                return candidate
    return None


# ── Temporal deduplication ────────────────────────────────────────────────────

def _parse_cam_dt(filename: str) -> tuple[str, datetime] | None:
    """Парсит камеру и datetime из имени файла кропа."""
    stem = Path(filename).stem
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit() and i + 1 < len(parts):
            nt = parts[i + 1]
            if len(nt) == 6 and nt.isdigit():
                try:
                    cam = "_".join(parts[:i])
                    dt = datetime(
                        int(p[:4]), int(p[4:6]), int(p[6:]),
                        int(nt[:2]), int(nt[2:4]), int(nt[4:6]),
                        tzinfo=MSK,
                    )
                    return cam, dt
                except ValueError:
                    pass
    return None


def _dedup_temporal(candidates: list[dict], window_min: int) -> list[dict]:
    """Для каждого (камера, временной_бакет) оставляет первый кандидат."""
    seen: set[tuple] = set()
    result = []
    for c in candidates:
        parsed = _parse_cam_dt(c["filename"])
        if parsed is None:
            result.append(c)
            continue
        cam, dt = parsed
        bucket = (dt.hour * 60 + dt.minute) // window_min
        key = (cam, dt.date(), bucket)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


# ── Quota ─────────────────────────────────────────────────────────────────────

def _apply_quota(candidates: list[dict], max_per_class: int) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    result = []
    for c in candidates:
        cls = c["true_class"]
        if counts[cls] < max_per_class:
            result.append(c)
            counts[cls] += 1
    return result


def _count_class_sizes(output_dir: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for cls in _VALID_CLASSES:
        d = output_dir / cls
        sizes[cls] = len(list(d.glob("*.jpg"))) if d.is_dir() else 0
    return sizes


def _compute_target(sizes: dict[str, int], mode: str) -> float:
    vals = [v for v in sizes.values() if v > 0]
    if not vals:
        return 0.0
    if mode == "max":
        return float(max(sizes.values()))
    elif mode == "mean":
        return sum(sizes.values()) / len(sizes)
    elif mode == "median":
        s = sorted(sizes.values())
        n = len(s)
        return (s[n // 2 - 1] + s[n // 2]) / 2.0 if n % 2 == 0 else float(s[n // 2])
    else:  # geomean
        return prod(vals) ** (1.0 / len(vals))


def _split_errors_with_quota(
    candidates: list[dict],
    output_dir: Path,
    conf_high: float,
    target_mode: str,
) -> tuple[list[dict], float]:
    """Split strat-1 into strat-7 (confident, always include) + strat-1 (deficit quota).

    Returns (new_candidates, target_size).
    """
    non_errors = [c for c in candidates if c["strategy"] != 1]
    errors = [c for c in candidates if c["strategy"] == 1]
    if not errors:
        return candidates, 0.0

    current = _count_class_sizes(output_dir)
    target = _compute_target(current, target_mode)

    # If dataset is empty there's nothing to balance against — keep everything
    if target == 0.0:
        result = list(non_errors)
        for c in errors:
            c = dict(c)
            try:
                if float(c["conf"]) >= conf_high:
                    c["strategy"] = 7
            except (TypeError, ValueError):
                pass
            result.append(c)
        return result, target

    confident: list[dict] = []
    uncertain: list[dict] = []
    for c in errors:
        try:
            c_conf = float(c["conf"])
        except (TypeError, ValueError):
            c_conf = None
        if c_conf is not None and c_conf >= conf_high:
            c = dict(c)
            c["strategy"] = 7
            confident.append(c)
        else:
            c = dict(c)
            # score: prefer high-conf errors in underrepresented classes
            c_conf_f = c_conf if c_conf is not None else 0.5
            cur = max(1.0, float(current.get(c["true_class"], 0)))
            c["_score"] = c_conf_f * (target / cur)
            uncertain.append(c)

    uncertain.sort(key=lambda x: -x["_score"])

    quota_used: dict[str, int] = defaultdict(int)
    selected: list[dict] = []
    for c in uncertain:
        cls = c["true_class"]
        deficit = max(0, int(target) - current.get(cls, 0))
        if deficit > 0 and quota_used[cls] < deficit:
            selected.append(c)
            quota_used[cls] += 1

    for c in selected:
        c.pop("_score", None)

    return non_errors + confident + selected, target


# ── Strategy collectors ───────────────────────────────────────────────────────

def _collect_from_labels(
    labels: dict[str, str],
    strategies: set[int],
) -> list[dict]:
    """Стратегии 1 и 2: итерируем labels.json."""
    result = []
    for abs_path_str, true_class in labels.items():
        if true_class not in _VALID_CLASSES:
            continue
        p = Path(abs_path_str)
        if not p.exists():
            continue
        actual_cls = p.parent.name
        if actual_cls == UNCERTAIN_DIR:
            if 2 in strategies:
                result.append({
                    "strategy": 2, "src": p, "filename": p.name,
                    "true_class": true_class, "pred_class": UNCERTAIN_DIR,
                    "conf": "", "conf_2nd": "", "margin": "",
                })
        elif actual_cls != true_class:
            if 1 in strategies:
                result.append({
                    "strategy": 1, "src": p, "filename": p.name,
                    "true_class": true_class, "pred_class": actual_cls,
                    "conf": "", "conf_2nd": "", "margin": "",
                })
    return result


def _resolve_src(date_dir: Path, out_class: str, filename: str) -> Path | None:
    """Ищет файл кропа: сначала напрямую, потом в unique/ подкаталоге."""
    for candidate in (
        date_dir / out_class / filename,
        date_dir / "unique" / out_class / filename,
    ):
        if candidate.exists():
            return candidate.resolve()
    return None


def _collect_from_csv(
    date_dir: Path,
    csv_rows: list[dict],
    labels: dict[str, str],
    *,
    conf_low: float,
    conf_high: float,
    margin_thresh: float,
    strategies: set[int],
) -> list[dict]:
    """Стратегии 3, 5, 8: читаем classifications.csv."""
    result = []
    for row in csv_rows:
        try:
            conf = float(row["group_conf"])
        except (KeyError, ValueError):
            continue

        filename  = row.get("crop", "")
        out_class = row.get("out_class", "")
        if not filename or not out_class:
            continue

        src = _resolve_src(date_dir, out_class, filename)
        if src is None:
            continue

        abs_str = str(src)
        true_in_json = labels.get(abs_str)

        try:
            conf_2nd = float(row["conf_2nd"]) if row.get("conf_2nd") else None
            margin   = float(row["margin"])   if row.get("margin")   else None
        except ValueError:
            conf_2nd = margin = None

        # Стратегия 3: серая зона (conf < conf_high), вручную размечено
        # Нижняя граница убрана: покрываем также случай conf < conf_low (стрт 4 gap)
        if (3 in strategies and conf < conf_high
                and true_in_json is not None and true_in_json in _VALID_CLASSES):
            result.append({
                "strategy": 3, "src": src, "filename": filename,
                "true_class": true_in_json, "pred_class": out_class,
                "conf": conf, "conf_2nd": conf_2nd or "", "margin": margin or "",
            })

        # Стратегия 5: псевдо-метки, модель уверена и права
        if 5 in strategies and conf >= conf_high:
            if true_in_json is None or true_in_json == out_class:
                result.append({
                    "strategy": 5, "src": src, "filename": filename,
                    "true_class": out_class, "pred_class": out_class,
                    "conf": conf, "conf_2nd": conf_2nd or "", "margin": margin or "",
                })

        # Стратегия 8: margin sampling (только если данные есть в CSV)
        if 8 in strategies and margin is not None and margin < margin_thresh:
            if true_in_json is not None and true_in_json in _VALID_CLASSES:
                result.append({
                    "strategy": 8, "src": src, "filename": filename,
                    "true_class": true_in_json, "pred_class": out_class,
                    "conf": conf, "conf_2nd": conf_2nd or "", "margin": margin,
                })

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Стратегии 1,2,3,5,8: отбор кропов из inference/images/YYYYMMDD/"
    )
    parser.add_argument("date_dir", type=Path,
                        help="Каталог inference/images/YYYYMMDD/")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Выходной каталог датасета (напр. .data/groups/v2/dataset)")
    parser.add_argument("--labels",  type=Path, default=None,
                        help="labels.json (по умолчанию: date_dir/labels.json)")
    parser.add_argument("--csv",     type=Path, default=None,
                        help="classifications.csv (по умолчанию: автопоиск)")
    parser.add_argument("--conf-low",     type=float,
                        default=_ef_float("CLASSIFY_CONF", 0.65))
    parser.add_argument("--conf-high",    type=float,
                        default=_ef_float("CLASSIFY_CONF_HIGH", 0.85))
    parser.add_argument("--margin-thresh", type=float,
                        default=_ef_float("MARGIN_THRESH", 0.20))
    parser.add_argument("--dedup-min",    type=int, default=15,
                        help="Окно временной дедупликации для стратегии 5, минут (0=выкл)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Квота: максимум файлов на класс")
    parser.add_argument("--strategies",   default="1,2,3,5,8",
                        help="Стратегии через запятую (default: 1,2,3,5,8)")
    parser.add_argument("--target-mode", default="geomean",
                        choices=["geomean", "median", "mean", "max"],
                        help="Целевой размер класса для квоты ошибок (default: geomean)")
    parser.add_argument("--no-error-quota", action="store_true",
                        help="Не применять квоту дефицита для стратегии 1 (включить все ошибки)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Не копировать, только показать статистику")
    args = parser.parse_args()

    date_dir = args.date_dir.resolve()
    if not date_dir.is_dir():
        logger.error("Не найден: %s", date_dir)
        return 1

    strategies = {int(s.strip()) for s in args.strategies.split(",") if s.strip()}
    date_name  = date_dir.name

    labels_path = args.labels or (date_dir / "labels.json")
    csv_path    = args.csv    or _auto_find_csv(date_dir)

    # Стратегии отбора (ошибка = класс≠подкаталог, псевдо-метки и т.п.) — single-label.
    # На multi-label данных (v4) они неприменимы: класс не выводится из одного подкаталога.
    if _is_multilabel(labels_path):
        logger.error(
            "labels.json — multi-label (v2). Стратегии отбора этого скрипта пока single-label.\n"
            "  Для v4 обучайте прямо на размеченном каталоге "
            "(single/<class>/ + multi/ + labels.json): 3_train_groups --data <dataset|inference-date>."
        )
        return 2

    labels   = _load_labels(labels_path)
    csv_rows = _load_csv(csv_path) if csv_path else []

    if 8 in strategies and not any(r.get("margin") for r in csv_rows):
        logger.warning("  Стратегия 8: колонка margin не найдена в CSV — пропускается.")
        strategies.discard(8)

    logger.info("=== dataset_v2_from_inference: %s ===", date_name)
    logger.info("  Стратегии:    %s", sorted(strategies))
    logger.info("  labels.json:  %s  (%d записей)", labels_path, len(labels))
    logger.info("  CSV:          %s  (%d строк)", csv_path or "—", len(csv_rows))
    logger.info("  conf_low=%.2f  conf_high=%.2f  margin_thresh=%.2f",
                args.conf_low, args.conf_high, args.margin_thresh)
    logger.info("  Вывод:        %s", args.output)
    logger.info("")

    # ── Collect ───────────────────────────────────────────────────────────────
    # Индекс conf по абсолютному пути — для обогащения стрт 1 данными из CSV
    csv_conf: dict[str, tuple] = {}
    for row in csv_rows:
        try:
            c_val = float(row["group_conf"])
        except (KeyError, ValueError):
            continue
        fn = row.get("crop", "")
        oc = row.get("out_class", "")
        if not fn or not oc:
            continue
        src = _resolve_src(date_dir, oc, fn)
        if src:
            try:
                c2 = float(row["conf_2nd"]) if row.get("conf_2nd") else None
                mg = float(row["margin"])   if row.get("margin")   else None
            except ValueError:
                c2 = mg = None
            csv_conf[str(src)] = (c_val, c2, mg)

    candidates: list[dict] = []
    candidates += _collect_from_labels(labels, strategies)
    candidates += _collect_from_csv(
        date_dir, csv_rows, labels,
        conf_low=args.conf_low, conf_high=args.conf_high,
        margin_thresh=args.margin_thresh, strategies=strategies,
    )

    # Деdup по пути: один файл → одна стратегия (с наименьшим номером)
    seen: dict[str, dict] = {}
    for c in sorted(candidates, key=lambda x: x["strategy"]):
        key = str(c["src"])
        if key not in seen:
            seen[key] = c
    candidates = list(seen.values())

    # Обогащаем стрт 1 данными conf из CSV (labels-only кандидаты не имеют conf)
    if 1 in strategies and csv_conf:
        enriched = []
        for c in candidates:
            if c["strategy"] == 1 and c["conf"] == "":
                key = str(c["src"])
                if key in csv_conf:
                    c_val, c2, mg = csv_conf[key]
                    c = dict(c)
                    c["conf"] = c_val
                    if c2 is not None:
                        c["conf_2nd"] = c2
                    if mg is not None:
                        c["margin"] = mg
            enriched.append(c)
        candidates = enriched

    # Временна́я дедупликация для стратегии 5
    s5    = [c for c in candidates if c["strategy"] == 5]
    other = [c for c in candidates if c["strategy"] != 5]
    if s5 and args.dedup_min > 0:
        before = len(s5)
        s5 = _dedup_temporal(s5, args.dedup_min)
        logger.info("  Стратегия 5: дедупликация %d → %d (окно %dмин)", before, len(s5), args.dedup_min)
    candidates = other + s5

    # Квота ошибок: стрт 1 → стрт 7 (уверенные) + стрт 1 (дефицит-квота)
    if 1 in strategies and not args.no_error_quota:
        before_err = len([c for c in candidates if c["strategy"] == 1])
        out_resolved = args.output.resolve()
        candidates, target_val = _split_errors_with_quota(
            candidates, out_resolved, args.conf_high, args.target_mode
        )
        after_s1 = len([c for c in candidates if c["strategy"] == 1])
        after_s7 = len([c for c in candidates if c["strategy"] == 7])
        logger.info(
            "  Квота ошибок (target=%s=%.0f): стрт 7: %d  стрт 1: %d  (было 1: %d)",
            args.target_mode, target_val, after_s7, after_s1, before_err,
        )

    # Квота на класс (сортируем по приоритету перед применением)
    if args.max_per_class:
        candidates.sort(key=lambda x: x["strategy"])
        before = len(candidates)
        candidates = _apply_quota(candidates, args.max_per_class)
        logger.info("  Квота %d/класс: %d → %d", args.max_per_class, before, len(candidates))

    # ── Stats ─────────────────────────────────────────────────────────────────
    by_strategy: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in candidates:
        by_strategy[c["strategy"]][c["true_class"]] += 1

    for s in sorted(by_strategy):
        total = sum(by_strategy[s].values())
        breakdown = "  ".join(f"{cls}:{n}" for cls, n in sorted(by_strategy[s].items()))
        logger.info("  Стратегия %d: %d  [%s]", s, total, breakdown)
    logger.info("  Итого: %d файлов", len(candidates))

    if args.dry_run or not candidates:
        if args.dry_run:
            logger.info("  (dry-run: файлы не скопированы)")
        return 0

    # ── Copy ──────────────────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    sources_path = args.output / "sources.csv"
    write_header = not sources_path.is_file()
    now_ts = datetime.now(MSK).strftime("%Y%m%d_%H%M%S_msk")
    copied = skipped = 0

    with open(sources_path, "a", newline="", encoding="utf-8") as sf:
        sw = csv.DictWriter(sf, fieldnames=_SOURCES_FIELDS)
        if write_header:
            sw.writeheader()
        for c in candidates:
            cls_dir = args.output / c["true_class"]
            cls_dir.mkdir(exist_ok=True)
            dst = cls_dir / c["filename"]
            if not dst.exists():
                shutil.copy2(str(c["src"]), str(dst))
                copied += 1
            else:
                skipped += 1
            sw.writerow({
                "timestamp":  now_ts,
                "strategy":   c["strategy"],
                "date":       date_name,
                "filename":   c["filename"],
                "src_path":   str(c["src"]),
                "dst_path":   str(dst),
                "true_class": c["true_class"],
                "pred_class": c["pred_class"],
                "conf":       c["conf"],
                "conf_2nd":   c["conf_2nd"],
                "margin":     c["margin"],
            })

    logger.info("  Скопировано: %d  Пропущено (уже есть): %d", copied, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
