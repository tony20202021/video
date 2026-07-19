#!/usr/bin/env bash
# Досев labels.json по результатам инференса (3_classify_groups) — multi-label (v4).
#
# ВНИМАНИЕ: обычно НЕ нужен — 3_classify_groups.py уже пишет labels.json (v2) при инференсе,
# а 2_label_ui ведёт его при ручной разметке. Скрипт лишь ДОСЕВАЕТ одиночные метки из
# single/<class>/ для файлов, которых ещё нет в labels.json, и СОХРАНЯЕТ существующие
# (в т.ч. multi/) — формат v2 {img: [classes]}, ключи относительно images/<date>/.
#
#   inference/images/YYYYMMDD/
#     single/1_resident/crop1.jpg  →  labels["single/1_resident/crop1.jpg"] = ["1_resident"]
#     multi/crop2.jpg              →  сохраняется как есть из labels.json (набор классов)
#     labels.json (v2)             ← создаётся/дополняется этим скриптом
#
# Usage:
#   ./sh/train/groups/1_3_dataset_groups_inference_labels.sh
#   ./sh/train/groups/1_3_dataset_groups_inference_labels.sh --poll-sec 30
#   ./sh/train/groups/1_3_dataset_groups_inference_labels.sh --once

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"

_ef() { local k="$1" d="$2"; local v; v=$(grep -E "^\s*${k}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r'); echo "${v:-$d}"; }
GROUPS_VER="$(_ef GROUPS_VER v3)"

INFERENCE_IMAGES="$REPO/.data/groups/$GROUPS_VER/inference/images"

POLL_SEC=60
ONCE=0

export PYTHONIOENCODING=utf-8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --poll-sec) POLL_SEC="$2"; shift 2 ;;
        --once)     ONCE=1; shift ;;
        *) echo "[!] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

_ts() { date '+%H:%M:%S'; }
SCRIPT_NAME="$(basename "$0" .sh)"

# ── Досев labels.json для одного каталога с датой (v4 multi-label) ────────────
#
# Читает существующий labels.json (v2, сохраняет multi/), досеивает одиночные метки
# из single/<class>/ для незнакомых файлов, пишет v2. Пропускает, если не изменилось.
#
_update_date_dir() {
    local date_dir="$1"
    "$PYTHON" - "$date_dir" "$REPO" <<'PYEOF'
import sys
from pathlib import Path

date_dir = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo / "src"))
from common.utils import multilabel as ml
from common.utils.classes import GROUP_CLASSES

labels_path = date_dir / "labels.json"
labels = ml.load_labels(labels_path)          # {rel: [classes]} — сохраняем существующее (incl. multi/)

single_root = date_dir / "single"
if single_root.is_dir():
    for cls_dir in sorted(single_root.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in GROUP_CLASSES:
            continue
        for f in sorted(cls_dir.glob("*.jpg")):
            rel = f.relative_to(date_dir).as_posix()
            labels.setdefault(rel, [cls_dir.name])

before = labels_path.read_text(encoding="utf-8") if labels_path.is_file() else ""
ml.save_labels(labels_path, labels, task="classify")
after = labels_path.read_text(encoding="utf-8")
if after != before:
    print(f"labels.json: {len(labels)} записей → {labels_path}")
PYEOF
}

# ── Watch-цикл ────────────────────────────────────────────────────────────────

echo "=== 1_3_dataset_groups_inference_labels ==="
echo "  Inference images: $INFERENCE_IMAGES"
echo "  poll: ${POLL_SEC}s"
echo ""
echo "[labels] Ctrl+C для остановки"
echo ""

while true; do
    # Собираем все каталоги с датами, в которых есть jpg
    mapfile -t date_dirs < <(
        find "$INFERENCE_IMAGES" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort
    )

    found=0
    for date_dir in "${date_dirs[@]+"${date_dirs[@]}"}"; do
        n=$(find "$date_dir" -name "*.jpg" -type f 2>/dev/null | wc -l)
        [[ "$n" -eq 0 ]] && continue

        found=1
        out=$(_update_date_dir "$date_dir")
        if [[ -n "$out" ]]; then
            echo "$(_ts)  INFO      $(basename "$date_dir"): $out"
        else
            echo "$(_ts)  INFO      ($SCRIPT_NAME) $date_dir: labels.json актуален ($n файлов) — ожидание ${POLL_SEC}s…"
        fi
    done

    if [[ "$found" -eq 0 ]]; then
        echo "$(_ts)  INFO      ($SCRIPT_NAME) Файлов нет в $INFERENCE_IMAGES — ожидание ${POLL_SEC}s…"
    fi

    if [[ "$ONCE" -eq 1 ]]; then
        break
    fi

    sleep "$POLL_SEC"
done
