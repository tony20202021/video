#!/usr/bin/env bash
# Управление датасетом групп (Модель 1)
#
# Usage:
#   ./sh/train/groups/1_dataset_groups.sh build
#   ./sh/train/groups/1_dataset_groups.sh build --labels .output/train/2_label_ui/labels.json
#   ./sh/train/groups/1_dataset_groups.sh apply --labels .data/groups/v2/new/labels.json --dataset .data/groups/v2
#   ./sh/train/groups/1_dataset_groups.sh apply --labels .data/groups/v2/new/labels.json --dataset .data/groups/v2 --move
#   ./sh/train/groups/1_dataset_groups.sh check --src .data/groups/v2/new --dataset .data/groups/v2/dataset
#   ./sh/train/groups/1_dataset_groups.sh add --src .output/pipeline/2_yolo_boxes_files/run_xxx --dataset .data/groups/v2
#   ./sh/train/groups/1_dataset_groups.sh status --dataset .data/groups/v2

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV="conda_video"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
SCRIPT="$REPO/scripts/train/dataset_groups.py"

export PYTHONIOENCODING=utf-8

if [[ $# -eq 0 ]]; then
    CMD="apply"
else
    CMD="$1"
    shift
fi

case "$CMD" in
    build|apply|check|add|status) ;;
    *) echo "[!] Unknown command: $CMD (expected: build, apply, check, add, status)" >&2; exit 1 ;;
esac

_ef() { local k="$1" d="$2"; local v; v=$(grep -E "^\s*${k}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r'); echo "${v:-$d}"; }
GROUPS_VER="$(_ef GROUPS_VER v3)"

LABELS="$REPO/.data/groups/$GROUPS_VER/new/labels.json"
VERSION=""
OUT=""
MOVE=1
SRC="$REPO/.data/groups/$GROUPS_VER/new"
DATASET="$REPO/.data/groups/$GROUPS_VER/dataset"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --labels)  LABELS="$2";  shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --out)     OUT="$2";     shift 2 ;;
        --move)    MOVE=1;       shift ;;
        --src)     SRC="$2";     shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "[!] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

_abs() { local p="$1"; [[ "$p" != /* ]] && p="$REPO/$p"; echo "$p"; }

args=("$CMD")

case "$CMD" in
    build)
        [[ -n "$LABELS"  ]] && args+=("--labels"  "$(_abs "$LABELS")")
        [[ -n "$VERSION" ]] && args+=("--version" "$VERSION")
        [[ -n "$OUT"     ]] && args+=("--out"     "$(_abs "$OUT")")
        ;;
    apply)
        if [[ -z "$LABELS" ]];  then echo "[!] --labels required" >&2;  exit 1; fi
        if [[ -z "$DATASET" ]]; then echo "[!] --dataset required" >&2; exit 1; fi
        args+=("--labels" "$(_abs "$LABELS")" "--dataset" "$(_abs "$DATASET")")
        [[ "$MOVE" -eq 1 ]] && args+=("--move")
        ;;
    check)
        if [[ -z "$SRC" ]];     then echo "[!] --src required" >&2;     exit 1; fi
        if [[ -z "$DATASET" ]]; then echo "[!] --dataset required" >&2; exit 1; fi
        args+=("--src" "$(_abs "$SRC")" "--dataset" "$(_abs "$DATASET")")
        ;;
    add)
        if [[ -z "$SRC" ]];     then echo "[!] --src required" >&2;     exit 1; fi
        if [[ -z "$DATASET" ]]; then echo "[!] --dataset required" >&2; exit 1; fi
        args+=("--src" "$(_abs "$SRC")" "--dataset" "$(_abs "$DATASET")")
        ;;
    status)
        if [[ -z "$DATASET" ]]; then echo "[!] --dataset required" >&2; exit 1; fi
        args+=("--dataset" "$(_abs "$DATASET")")
        ;;
esac

echo "=== dataset_groups: $CMD ==="
echo ""

exec "$PYTHON" "$SCRIPT" "${args[@]}"
