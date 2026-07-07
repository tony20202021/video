#!/usr/bin/env bash
# Непрерывная дедупликация внутри .data/groups/v1/new/:
# находит файлы с одинаковым именем (из разных подкаталогов), удаляет лишние,
# оставляя первый по алфавиту пути.
#
# Usage:
#   ./sh/train/1_2_dataset_groups_check_new.sh
#   ./sh/train/1_2_dataset_groups_check_new.sh --no-delete   # только показать, не удалять
#   ./sh/train/1_2_dataset_groups_check_new.sh --poll-sec 30
#   ./sh/train/1_2_dataset_groups_check_new.sh --once        # один прогон и выход

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

NEW_DIR="$REPO/.data/groups/v1/new"

POLL_SEC=30
DELETE=1   # удалять дубли
ONCE=0     # один прогон и выход

export PYTHONIOENCODING=utf-8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-delete) DELETE=0; shift ;;
        --poll-sec)  POLL_SEC="$2"; shift 2 ;;
        --once)      ONCE=1; shift ;;
        *) echo "[!] Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d "$NEW_DIR" ]]; then
    echo "[!] Каталог не найден: $NEW_DIR" >&2
    exit 1
fi

_ts() { date '+%H:%M:%S'; }
SCRIPT_NAME="$(basename "$0" .sh)"

echo "=== 1_2_dataset_groups_check_new ==="
echo "  New dir:  $NEW_DIR"
echo "  delete=$DELETE  poll=${POLL_SEC}s"
echo ""
echo "[dedup] Ctrl+C для остановки"
echo ""

# ── Дедупликация ───────────────────────────────────────────────────────────────
#
# Алгоритм:
#   1. Сканируем все .jpg в new/ рекурсивно, сортируем по пути (алфавитно).
#   2. Для каждого имени файла: первый путь — оставляем, остальные — дубли.
#   3. Дубли удаляем (или только сообщаем при --no-delete).

_dedup() {
    local n_files n_dupes=0 n_kept=0

    n_files=$(find "$NEW_DIR" -name "*.jpg" -type f 2>/dev/null | wc -l)
    if [[ "$n_files" -eq 0 ]]; then
        return 0
    fi

    declare -A seen
    while IFS= read -r f; do
        local name
        name=$(basename "$f")
        if [[ -v seen[$name] ]]; then
            # дубль
            local rel="${f#$NEW_DIR/}"
            if [[ "$DELETE" -eq 1 ]]; then
                rm -f "$f"
                echo "$(_ts)  INFO      [✕ дубль удалён] $rel"
            else
                echo "$(_ts)  INFO      [= дубль] $rel  (оригинал: ${seen[$name]#$NEW_DIR/})"
            fi
            (( n_dupes++ )) || true
        else
            seen[$name]="$f"
            (( n_kept++ )) || true
        fi
    done < <(find "$NEW_DIR" -name "*.jpg" -type f 2>/dev/null | sort)

    unset seen

    if [[ $n_dupes -gt 0 || $n_files -gt 0 ]]; then
        if [[ "$DELETE" -eq 1 ]]; then
            echo "$(_ts)  INFO      Файлов: $n_files  уникальных: $n_kept  дублей удалено: $n_dupes"
        else
            echo "$(_ts)  INFO      Файлов: $n_files  уникальных: $n_kept  дублей: $n_dupes"
        fi
    fi
}

# ── Watch-цикл ────────────────────────────────────────────────────────────────

while true; do
    n=$(find "$NEW_DIR" -name "*.jpg" -type f 2>/dev/null | wc -l)

    if [[ "$n" -gt 0 ]]; then
        echo "$(_ts)  INFO      Найдено файлов в new/: $n — проверка дублей…"
        _dedup
        echo ""
    else
        echo "$(_ts)  INFO      ($SCRIPT_NAME) Файлов нет в $NEW_DIR — ожидание ${POLL_SEC}s…"
    fi

    if [[ "$ONCE" -eq 1 ]]; then
        break
    fi

    sleep "$POLL_SEC"
done
