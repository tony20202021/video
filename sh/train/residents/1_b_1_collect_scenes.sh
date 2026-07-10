#!/usr/bin/env bash
# Сбор кропов жителей по сценам для полу-автоматической разметки.
#
# Шаг 1 из 4 в пайплайне полу-авторазметки:
#   1. [этот скрипт] Собрать кадры без diff-фильтра → разбить на сцены
#   2. Разметить по 1 кадру на сцену в .data/residents/$RESIDENTS_VER/scene_pool/representatives/
#   3. ./5_propagate_scenes.sh → авторазметка внутри сцен
#   4. ./2_filter_residents.sh --min-diff N → финальная дедупликация
#
# Выход (.data/residents/$RESIDENTS_VER/scene_pool/):
#   .data/residents/<RESIDENTS_VER>/scene_pool/
#     *.jpg                 — все кандидаты (без diff-фильтра)
#     scenes.json           — разбивка на сцены + представители
#     representatives/      — по 1 (самый резкий) кадр на сцену для разметки
#
# Usage:
#   ./sh/train/residents/4_collect_scenes.sh
#   ./sh/train/residents/4_collect_scenes.sh --scene-gap 60 --scene-diff 40
#   ./sh/train/residents/4_collect_scenes.sh --dry-run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/conda_video/bin/python"
SCRIPT="$REPO/scripts/train/collect_scenes.py"

export PYTHONIOENCODING=utf-8

_ef() {
    local val
    val=$(grep -E "^\s*${1}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${val:-$2}"
}

RESIDENTS_VER="$(_ef RESIDENTS_VER v1)"
OUT="$REPO/.data/residents/$RESIDENTS_VER/scene_pool"
SCENE_GAP=120
SCENE_DIFF=50
INTERVAL=2
MIN_CONF=0.35
MIN_BLUR=40
MAX_PERSONS=1
EXTRA_ARGS=()

# ── Входные каталоги ──────────────────────────────────────────────────────────
# Добавлять/убирать по мере накопления новых дат.
# Формат: --src <путь>  (по одному на каждый class-каталог)
SOURCES=(
    # датасет v1
    --src "$REPO/.data/groups/v1/dataset/1_resident"        # 3927
    --src "$REPO/.data/groups/v1/dataset/4_guest"           # 179
    # инференс v1
    --src "$REPO/.data/groups/v1/inference/images/20260705/1_resident"  # 987
    --src "$REPO/.data/groups/v1/inference/images/20260705/4_guest"     # 340
    --src "$REPO/.data/groups/v1/inference/images/20260706/1_resident"  # 1105
    --src "$REPO/.data/groups/v1/inference/images/20260706/4_guest"     # 367
    --src "$REPO/.data/groups/v1/inference/images/20260707/1_resident"  # 623
    --src "$REPO/.data/groups/v1/inference/images/20260707/4_guest"     # 211
    # инференс v2
    --src "$REPO/.data/groups/v2/inference/images/20260708/1_resident"  # 982
    --src "$REPO/.data/groups/v2/inference/images/20260708/4_guest"     # 7
    --src "$REPO/.data/groups/v2/inference/images/20260709/1_resident"  # 1715
    --src "$REPO/.data/groups/v2/inference/images/20260709/4_guest"     # 79
    # --src "$REPO/.data/groups/v2/inference/images/20260710/1_resident"  # 14  (не размечен)
)
# ─────────────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)           OUT="$2";       shift 2 ;;
        --scene-gap)     SCENE_GAP="$2"; shift 2 ;;
        --scene-diff)    SCENE_DIFF="$2"; shift 2 ;;
        --interval)      INTERVAL="$2";  shift 2 ;;
        --min-conf)      MIN_CONF="$2";  shift 2 ;;
        --min-blur)      MIN_BLUR="$2";  shift 2 ;;
        --max-persons)   MAX_PERSONS="$2"; shift 2 ;;
        --dry-run)       EXTRA_ARGS+=("--dry-run"); shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== 1_b_1_collect_scenes ==="
echo "  Выход:       $OUT"
echo "  scene-gap:   ${SCENE_GAP}s | scene-diff: $SCENE_DIFF"
echo "  interval:    ${INTERVAL}s | min-conf: $MIN_CONF | min-blur: $MIN_BLUR"
echo ""

"$PYTHON" "$SCRIPT" \
    --out          "$OUT" \
    --scene-gap    "$SCENE_GAP" \
    --scene-diff   "$SCENE_DIFF" \
    --interval     "$INTERVAL" \
    --min-conf     "$MIN_CONF" \
    --min-blur     "$MIN_BLUR" \
    --max-persons  "$MAX_PERSONS" \
    "${SOURCES[@]}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo ""
echo "Следующий шаг:"
echo "  Разметьте representatives/ (по 1 кадру на сцену) → labels.json"
echo "  Затем: ./sh/train/residents/5_propagate_scenes.sh"
