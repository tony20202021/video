#!/usr/bin/env bash
# Автоматическое создание labels.json по результатам инференса (3_classify_groups).
#
# Следит за .data/groups/v1/inference/images/ — для каждого каталога с датой
# создаёт/обновляет labels.json внутри него на основе структуры подкаталогов-классов:
#
#   inference/images/YYYYMMDD/
#     1_resident/crop1.jpg   →  labels["…/crop1.jpg"] = "1_resident"
#     2_delivery/crop2.jpg   →  labels["…/crop2.jpg"] = "2_delivery"
#     unknown/crop3.jpg      →  labels["…/crop3.jpg"] = "unknown"
#     labels.json            ← создаётся/обновляется этим скриптом
#
# Формат labels.json идентичен выводу 2_label_ui и совместим с:
#   ./sh/train/1_dataset_groups.sh apply --labels inference/images/YYYYMMDD/labels.json
#
# Usage:
#   ./sh/train/1_3_dataset_groups_inference_labels.sh
#   ./sh/train/1_3_dataset_groups_inference_labels.sh --poll-sec 30
#   ./sh/train/1_3_dataset_groups_inference_labels.sh --once

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"

INFERENCE_IMAGES="$REPO/.data/groups/v1/inference/images"

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

# ── Обновление labels.json для одного каталога с датой ────────────────────────
#
# Сканирует все *.jpg в подкаталогах первого уровня (имя подкаталога = класс),
# строит {абс_путь: класс} и записывает labels.json.
# Пропускает, если содержимое не изменилось.
#
_update_date_dir() {
    local date_dir="$1"
    "$PYTHON" - "$date_dir" <<'PYEOF'
import json, sys
from pathlib import Path

date_dir = Path(sys.argv[1]).resolve()
labels: dict[str, str] = {}

for jpg in sorted(date_dir.rglob("*.jpg")):
    rel = jpg.relative_to(date_dir)
    parts = rel.parts
    if len(parts) >= 2:          # class_dir/filename.jpg
        cls = parts[0]
        labels[str(jpg)] = cls

labels_path = date_dir / "labels.json"

if labels_path.is_file():
    try:
        existing = json.loads(labels_path.read_text(encoding="utf-8"))
        if existing.get("labels") == labels:
            sys.exit(0)          # нет изменений
    except (json.JSONDecodeError, KeyError):
        pass                     # повреждённый JSON — перезапишем

labels_path.write_text(
    json.dumps({"version": 1, "labels": labels}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
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
            echo "$(_ts)  INFO      $date_dir: labels.json актуален ($n файлов) — ожидание ${POLL_SEC}s…"
        fi
    done

    if [[ "$found" -eq 0 ]]; then
        echo "$(_ts)  INFO      Файлов нет в $INFERENCE_IMAGES — ожидание ${POLL_SEC}s…"
    fi

    if [[ "$ONCE" -eq 1 ]]; then
        break
    fi

    sleep "$POLL_SEC"
done
