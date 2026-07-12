#!/usr/bin/env bash
# Мастер-скрипт: собирает новый датасет из всех источников.
#
# Вызывает:
#   3_dataset_run_inference.sh  — для каждого --dataset (прогон модели, полная унификация)
#   3_dataset_from_prev.sh      — для каждого --prev датасета (только стратегии 4,7)
#   3_dataset_from_inference.sh — для каждого --inference каталога (стратегии 1,2,3,5,7,8)
#   3_dataset_fill_minor.sh     — после всех стратегий (стратегия F)
#
# --dataset  полная унификация: прогоняет модель на датасете, создаёт inference-style
#            каталог <dataset>/../inference/images/dataset/, затем обрабатывает его
#            со всеми стратегиями 1,2,3,5,7,8. Заменяет --prev для максимального охвата.
#
# --prev     устаревший вариант: только стратегии 4,7. Оставлен для совместимости.
#
# --prev, --dataset, --inference можно указывать несколько раз.
#
# Usage (полная унификация):
#   ./sh/train/groups/3_dataset_build.sh \
#       --dataset   .data/groups/v1/dataset \
#       --dataset   .data/groups/v2/dataset \
#       --inference .data/groups/v1/inference/images \
#       --inference .data/groups/v2/inference/images \
#       --output    .data/groups/v3/dataset \
#       --from-date 20260708 \
#       --always-fill \
#       --dry-run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

FROM_INFERENCE="$REPO/sh/train/groups/3_dataset_from_inference.sh"
FROM_PREV="$REPO/sh/train/groups/3_dataset_from_prev.sh"
RUN_INFERENCE="$REPO/sh/train/groups/3_dataset_run_inference.sh"
FILL_MINOR="$REPO/sh/train/groups/3_dataset_fill_minor.sh"

export PYTHONIOENCODING=utf-8

DATASET_DIRS=()    # --dataset: полная унификация (все стратегии)
PREV_DATASETS=()   # --prev:    только стратегии 4,7 (устаревший)
INFERENCE_DIRS=()
OUTPUT=""
FROM_DATE=""
DRY_RUN=""
ALWAYS_FILL=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET_DIRS+=("$2");   shift 2 ;;
        --prev)          PREV_DATASETS+=("$2");  shift 2 ;;
        --inference)     INFERENCE_DIRS+=("$2"); shift 2 ;;
        --output|-o)     OUTPUT="$2";            shift 2 ;;
        --from-date)     FROM_DATE="$2";         shift 2 ;;
        --dry-run)       DRY_RUN="--dry-run"; EXTRA_ARGS+=("--dry-run"); shift ;;
        --always-fill)   ALWAYS_FILL="--always-fill"; shift ;;
        --strategies)    EXTRA_ARGS+=("--strategies" "$2"); shift 2 ;;
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
for d in "${DATASET_DIRS[@]+"${DATASET_DIRS[@]}"}"; do
    echo "  dataset (unified): $d"
done
for p in "${PREV_DATASETS[@]+"${PREV_DATASETS[@]}"}"; do
    echo "  prev (strats 4,7): $p"
done
for i in "${INFERENCE_DIRS[@]+"${INFERENCE_DIRS[@]}"}"; do
    echo "  inference images:  $i"
done
echo "  output:            $OUTPUT"
[[ -n "$FROM_DATE" ]] && echo "  from-date:         $FROM_DATE"
echo ""

# ── Полная унификация: прогон модели на датасете → все стратегии 1,2,3,5,7,8 ─
for ds in "${DATASET_DIRS[@]+"${DATASET_DIRS[@]}"}"; do
    if [[ ! -d "$ds" ]]; then
        echo "$(_ts)  WARN      Датасет не найден: $ds — пропускаем"
        continue
    fi
    ds_inf_out="$(cd "$ds/.." && pwd)/inference/images/dataset"
    echo "$(_ts)  INFO      Инференс модели на датасете: $ds"
    echo "$(_ts)  INFO        → $ds_inf_out"
    "$RUN_INFERENCE" "$ds" --output "$ds_inf_out" ${DRY_RUN:+"$DRY_RUN"}
    echo ""
    echo "$(_ts)  INFO      Стратегии 1,2,3,5,7,8: $ds_inf_out"
    # dedup-min 0: датасет не видео-поток, временная дедупликация не нужна
    "$FROM_INFERENCE" "$ds_inf_out" --output "$OUTPUT" \
        --dedup-min 0 "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    echo ""
done

# ── Стратегии 4,7: предыдущие датасеты (устаревший вариант) ──────────────────
for prev in "${PREV_DATASETS[@]+"${PREV_DATASETS[@]}"}"; do
    if [[ -d "$prev" ]]; then
        echo "$(_ts)  INFO      Стратегии 4,7 (--prev): $prev"
        "$FROM_PREV" "$prev" --output "$OUTPUT" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        echo ""
    else
        echo "$(_ts)  WARN      Предыдущий датасет не найден: $prev — пропускаем"
    fi
done

# ── Стратегии 1,2,3,5,8: все inference-каталоги по датам ─────────────────────
_run_inference_dir() {
    local inf_root="$1"
    local with_from_date="$2"   # "yes" или "no"

    if [[ ! -d "$inf_root" ]]; then
        echo "$(_ts)  WARN      Каталог инференса не найден: $inf_root — пропускаем"
        return
    fi

    mapfile -t date_dirs < <(find "$inf_root" -mindepth 1 -maxdepth 1 -type d | sort)
    if [[ ${#date_dirs[@]} -eq 0 ]]; then
        echo "$(_ts)  WARN      Нет каталогов с датами в $inf_root"
        return
    fi

    for date_dir in "${date_dirs[@]}"; do
        date_name="$(basename "$date_dir")"
        if [[ "$with_from_date" == "yes" && -n "$FROM_DATE" && "$date_name" < "$FROM_DATE" ]]; then
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
}

# Первый --inference уважает --from-date, остальные — без фильтра
first=1
for inf in "${INFERENCE_DIRS[@]+"${INFERENCE_DIRS[@]}"}"; do
    if [[ "$first" -eq 1 ]]; then
        _run_inference_dir "$inf" "yes"
        first=0
    else
        _run_inference_dir "$inf" "no"
    fi
done

# ── Стратегия F: fill_minor из предыдущих датасетов ──────────────────────────
for ds in "${DATASET_DIRS[@]+"${DATASET_DIRS[@]}"}"; do
    if [[ -d "$ds" ]]; then
        echo "$(_ts)  INFO      Стратегия F: дополнение из --dataset $ds"
        "$FILL_MINOR" "$ds" --output "$OUTPUT" ${DRY_RUN:+"$DRY_RUN"} ${ALWAYS_FILL:+"$ALWAYS_FILL"}
        echo ""
    fi
done
for prev in "${PREV_DATASETS[@]+"${PREV_DATASETS[@]}"}"; do
    if [[ -d "$prev" ]]; then
        echo "$(_ts)  INFO      Стратегия F: дополнение из --prev $prev"
        "$FILL_MINOR" "$prev" --output "$OUTPUT" ${DRY_RUN:+"$DRY_RUN"} ${ALWAYS_FILL:+"$ALWAYS_FILL"}
        echo ""
    fi
done

echo "$(_ts)  INFO      Готово. Датасет: $OUTPUT"
