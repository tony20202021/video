#!/usr/bin/env bash
# Непрерывная проверка кропов из 2_yolo_boxes_files на дубли с датасетом.
#
# Поведение по умолчанию:
#   - Дубли (уже есть в датасете) — удалять
#   - Уникальные — перемещать в .data/groups/v2/new/ (с сохранением структуры подкаталогов)
#   - Работает непрерывно (watch-режим)
#
# Usage:
#   ./sh/train/groups/1_1_dataset_groups_check.sh
#   ./sh/train/groups/1_1_dataset_groups_check.sh --no-delete-doubles   # дубли не удалять
#   ./sh/train/groups/1_1_dataset_groups_check.sh --no-delete-after     # не чистить пустые каталоги
#   ./sh/train/groups/1_1_dataset_groups_check.sh --poll-sec 30

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/dataset_groups.py"

INPUT_DIR="$REPO/.output/pipeline/2_yolo_boxes_files"
DATASET="$REPO/.data/groups/v2/dataset"
DST_NEW="$REPO/.data/groups/v2/new"

POLL_SEC=10
DELETE_DOUBLES=1   # удалять дубли сразу
DELETE_AFTER=1     # чистить пустые каталоги после обработки

export PYTHONIOENCODING=utf-8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-delete-doubles) DELETE_DOUBLES=0; shift ;;
        --no-delete-after)   DELETE_AFTER=0;   shift ;;
        --poll-sec)          POLL_SEC="$2"; shift 2 ;;
        *) echo "[!] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$DST_NEW"

echo "=== 1_1_dataset_groups_check ==="
echo "  Input:          $INPUT_DIR"
echo "  Dataset:        $DATASET"
echo "  New → :         $DST_NEW"
echo "  delete-doubles=$DELETE_DOUBLES  delete-after=$DELETE_AFTER  poll=${POLL_SEC}s"
echo ""
echo "[check] Ctrl+C для остановки"
echo ""

# ── Вспомогательные функции ────────────────────────────────────────────────────

_count_crops() {
    find "$INPUT_DIR" -name "*.jpg" -type f \
        ! -path "*/unique/*" \
        ! -path "*/double/*" 2>/dev/null | wc -l
}

_ts() { date '+%H:%M:%S'; }
SCRIPT_NAME="$(basename "$0" .sh)"

_move_unique() {
    local unique_dir="$INPUT_DIR/unique"
    [[ ! -d "$unique_dir" ]] && return 0

    local moved=0
    while IFS= read -r -d '' f; do
        local rel="${f#$unique_dir/}"
        local dst="$DST_NEW/$rel"
        mkdir -p "$(dirname "$dst")"
        mv "$f" "$dst"
        echo "$(_ts)  INFO      [→ new] $rel"
        (( moved++ )) || true
    done < <(find "$unique_dir" -name "*.jpg" -type f -print0 2>/dev/null)

    # Убираем пустой unique/
    find "$unique_dir" -type d -empty -delete 2>/dev/null || true
    [[ -d "$unique_dir" ]] && rmdir "$unique_dir" 2>/dev/null || true

    echo "$(_ts)  INFO      Уникальных: $moved"
}

_handle_doubles() {
    local double_dir="$INPUT_DIR/double"
    [[ ! -d "$double_dir" ]] && return 0

    local count
    count=$(find "$double_dir" -name "*.jpg" -type f 2>/dev/null | wc -l)

    if [[ "$DELETE_DOUBLES" -eq 1 ]]; then
        rm -rf "$double_dir"
        echo "$(_ts)  INFO      Дублей удалено: $count"
    else
        echo "$(_ts)  INFO      Дублей (оставлено в double/): $count"
    fi
}

_prune_empty_dirs() {
    [[ "$DELETE_AFTER" -eq 0 ]] && return 0
    find "$INPUT_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
}

# ── Watch-цикл ────────────────────────────────────────────────────────────────

while true; do
    n=$(_count_crops)

    if [[ "$n" -eq 0 ]]; then
        echo "$(_ts)  INFO      ($SCRIPT_NAME) Кропов нет в $INPUT_DIR — ожидание ${POLL_SEC}s…"
        sleep "$POLL_SEC"
        continue
    fi

    ts="$(date '+%H:%M:%S')"
    echo "[$ts] Найдено кропов: $n — запуск check…"

    "$PYTHON" "$SCRIPT" check \
        --src     "$INPUT_DIR" \
        --dataset "$DATASET"

    _move_unique
    _handle_doubles
    _prune_empty_dirs
    echo ""
done
