#!/usr/bin/env bash
# Мастер-скрипт: собирает новый датасет из всех источников.
#
# Вызывает:
#   3_dataset_from_prev.sh    — один раз для старого датасета (стратегии 4,7)
#   3_dataset_from_inference.sh — циклом по всем датам инференса (стратегии 1,2,3,5,8)
#   3_dataset_fill_minor.sh   — после всех стратегий: дополняет классы где new < prev
#
# Usage:
#   ./sh/train/3_dataset_build.sh --output .data/groups/v2/dataset
#   ./sh/train/3_dataset_build.sh \
#       --prev     .data/groups/v1/dataset \
#       --inference .data/groups/v1/inference/images \
#       --output   .data/groups/v2/dataset \
#       --from-date 20260701          # только даты >= этой
#       --dry-run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

FROM_INFERENCE="$REPO/sh/train/3_dataset_from_inference.sh"
FROM_PREV="$REPO/sh/train/3_dataset_from_prev.sh"
FILL_MINOR="$REPO/sh/train/3_dataset_fill_minor.sh"

export PYTHONIOENCODING=utf-8

# Дефолты
PREV_DATASET="$REPO/.data/groups/v1/dataset"
INFERENCE_IMAGES="$REPO/.data/groups/v1/inference/images"
OUTPUT="$REPO/.data/groups/v2/dataset"
FROM_DATE=""
DRY_RUN=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prev)       PREV_DATASET="$2";      shift 2 ;;
        --inference)  INFERENCE_IMAGES="$2";  shift 2 ;;
        --output|-o)  OUTPUT="$2";            shift 2 ;;
        --from-date)  FROM_DATE="$2";         shift 2 ;;
        --dry-run)    DRY_RUN="--dry-run"; EXTRA_ARGS+=("--dry-run"); shift ;;
        --strategies) EXTRA_ARGS+=("--strategies" "$2"); shift 2 ;;
        --max-per-class) EXTRA_ARGS+=("--max-per-class" "$2"); shift 2 ;;
        *) echo "[!] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT" ]]; then
    echo "[!] --output обязателен" >&2
    exit 1
fi

_ts() { date '+%H:%M:%S'; }

echo "=== 3_dataset_build ==="
echo "  prev dataset:      $PREV_DATASET"
echo "  inference images:  $INFERENCE_IMAGES"
echo "  output:            $OUTPUT"
[[ -n "$FROM_DATE" ]] && echo "  from-date:         $FROM_DATE"
echo ""

# ── Стратегии 4,7: предыдущий датасет ────────────────────────────────────────
if [[ -d "$PREV_DATASET" ]]; then
    echo "$(_ts)  INFO      Стратегии 4,7: $PREV_DATASET"
    "$FROM_PREV" "$PREV_DATASET" --output "$OUTPUT" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    echo ""
else
    echo "$(_ts)  WARN      Предыдущий датасет не найден: $PREV_DATASET — пропускаем"
fi

# ── Стратегии 1,2,3,5,8: каталоги инференса по датам ─────────────────────────
if [[ ! -d "$INFERENCE_IMAGES" ]]; then
    echo "$(_ts)  WARN      Каталог инференса не найден: $INFERENCE_IMAGES — пропускаем"
    exit 0
fi

mapfile -t date_dirs < <(find "$INFERENCE_IMAGES" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#date_dirs[@]} -eq 0 ]]; then
    echo "$(_ts)  WARN      Нет каталогов с датами в $INFERENCE_IMAGES"
    exit 0
fi

for date_dir in "${date_dirs[@]}"; do
    date_name="$(basename "$date_dir")"

    # Фильтр по --from-date
    if [[ -n "$FROM_DATE" && "$date_name" < "$FROM_DATE" ]]; then
        echo "$(_ts)  INFO      Пропускаем $date_name (раньше --from-date $FROM_DATE)"
        continue
    fi

    n=$(find "$date_dir" -name "*.jpg" -type f 2>/dev/null | wc -l)
    if [[ "$n" -eq 0 ]]; then
        echo "$(_ts)  INFO      $date_name: пусто, пропускаем"
        continue
    fi

    echo "$(_ts)  INFO      Стратегии 1,2,3,5,8: $date_name ($n файлов)"
    "$FROM_INFERENCE" "$date_dir" --output "$OUTPUT" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    echo ""
done

# ── Стратегия F: дополнить минорные классы из предыдущего датасета ────────────
if [[ -d "$PREV_DATASET" ]]; then
    echo "$(_ts)  INFO      Стратегия F: дополнение минорных классов из $PREV_DATASET"
    "$FILL_MINOR" "$PREV_DATASET" --output "$OUTPUT" ${DRY_RUN:+"$DRY_RUN"}
    echo ""
fi

echo "$(_ts)  INFO      Готово. Датасет: $OUTPUT"
