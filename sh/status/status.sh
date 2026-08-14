#!/usr/bin/env bash
# Мастер-скрипт статуса: Linux (локально) + Windows-клиенты
# Итоговый .md содержит две таблицы по всем машинам:
#   1. Ресурсы серверов (CPU / RAM / Disk / статус скриптов)
#   2. Linux-сервисы с колонкой 10м-статистики
#
# Usage:
#   ./sh/status/status.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

TS_SERVER=""
TS_SERVER_USER=""
TS_SERVER_REPO=""
TS_CAMERAS_3=""
TS_DEVELOP=""
TS_CAMERAS_1=""
TS_CAMERAS_3_USER=""
TS_DEVELOP_USER=""
TS_CAMERAS_1_USER=""
WIN_CAMERAS_3_LABEL="CAMERAS_3"
WIN_DEVELOP_LABEL="DEVELOP"
WIN_CAMERAS_1_LABEL="CAMERAS_1"
# Путь к репозиторию на каждой Windows-машине (переопределяется в .env: TS_*_REPO).
# Разный на разных машинах — отсюда строится путь к status_win.ps1.
TS_CAMERAS_3_REPO='C:\_Work\video'
TS_DEVELOP_REPO='E:\_Home\Tony\pet projects\video'
TS_CAMERAS_1_REPO='C:\Work\video'

if [[ -f "$REPO/.env" ]]; then
    while IFS= read -r _line; do
        [[ -z "$_line" || "$_line" =~ ^# ]] && continue
        _val="${_line#*=}"; _val="${_val%%#*}"; _val="${_val%"${_val##*[! ]}"}"
        _val="${_val#\"}"; _val="${_val%\"}"   # снять обрамляющие кавычки (пути с пробелами)
        case "$_line" in
            TS_SERVER=*)            TS_SERVER="$_val" ;;
            TS_SERVER_USER=*)       TS_SERVER_USER="$_val" ;;
            TS_SERVER_REPO=*)       TS_SERVER_REPO="$_val" ;;
            TS_CAMERAS_3=*)         TS_CAMERAS_3="$_val" ;;
            TS_DEVELOP=*)           TS_DEVELOP="$_val" ;;
            TS_CAMERAS_1=*)         TS_CAMERAS_1="$_val" ;;
            TS_CAMERAS_3_USER=*)    TS_CAMERAS_3_USER="$_val" ;;
            TS_DEVELOP_USER=*)      TS_DEVELOP_USER="$_val" ;;
            TS_CAMERAS_1_USER=*)    TS_CAMERAS_1_USER="$_val" ;;
            TS_CAMERAS_3_LABEL=*)   WIN_CAMERAS_3_LABEL="$_val" ;;
            TS_DEVELOP_LABEL=*)     WIN_DEVELOP_LABEL="$_val" ;;
            TS_CAMERAS_1_LABEL=*)   WIN_CAMERAS_1_LABEL="$_val" ;;
            TS_CAMERAS_3_REPO=*)    TS_CAMERAS_3_REPO="$_val" ;;
            TS_DEVELOP_REPO=*)      TS_DEVELOP_REPO="$_val" ;;
            TS_CAMERAS_1_REPO=*)    TS_CAMERAS_1_REPO="$_val" ;;
        esac
    done < "$REPO/.env"
fi

# Linux-сервер: дефолты для SSH-опроса (если запускаем НЕ на самом сервере)
TS_SERVER_USER="${TS_SERVER_USER:-$(whoami)}"
TS_SERVER_REPO="${TS_SERVER_REPO:-$REPO}"

# Путь к status_win.ps1 на каждой машине = <repo>\sh\status\status_win.ps1
WIN_SCRIPT_CAMERAS_3="$TS_CAMERAS_3_REPO\sh\status\status_win.ps1"
WIN_SCRIPT_DEVELOP="$TS_DEVELOP_REPO\sh\status\status_win.ps1"
WIN_SCRIPT_CAMERAS_1="$TS_CAMERAS_1_REPO\sh\status\status_win.ps1"

OUT_DIR="$REPO/.output/status/$(date +%Y-%m-%d)"
mkdir -p "$OUT_DIR"
TS_FILE=$(date +"%Y%m%d_%H%M%S")
TS_HUMAN=$(date +"%Y-%m-%d %H:%M:%S")
MD_OUT="$OUT_DIR/${TS_FILE}.md"
VER=$(cat "$REPO/VERSION" 2>/dev/null | tr -d '\r\n')

sep() { echo "════════════════════════════════════════════════════════"; }

# ── вспомогательная функция: сбор системных метрик ────────────────────────────
# Печатает одну строку: "cpu%|ram_used/ram_total MB|disk_used/disk_total GB"
_linux_sys_stats() {
    python3 "$REPO/scripts/utils/sys_stats.py" 2>/dev/null
}

# ── выполняемся ли мы НА этом узле? (IP среди локальных адресов) ───────────────
_is_self() {
    local ip="$1"
    [[ -z "$ip" ]] && return 1
    {
        hostname -I 2>/dev/null
        command -v ip >/dev/null 2>&1 && ip -4 -o addr show 2>/dev/null | grep -oE 'inet [0-9.]+' | awk '{print $2}'
        command -v tailscale >/dev/null 2>&1 && tailscale ip -4 2>/dev/null
    } | tr ' ' '\n' | grep -qx "$ip"
}

# ── парсинг метрик из вывода status_win.ps1 ───────────────────────────────────
_parse_win_sys() {
    local win_out="$1"
    local cpu ram disk work_path scripts
    # Убрать Windows \r — иначе awk/grep получают "OK\r" вместо "OK"
    local w
    w=$(echo "$win_out" | tr -d '\r')

    cpu=$(echo "$w" | grep "  CPU:" | grep -oE '[0-9]+%' | head -1)
    ram=$(echo "$w" | grep "  RAM:" | grep -oE '[0-9.]+/[0-9.]+ GB' | head -1)

    # Диски из строки "Disks: C: 105/465 GB<br>D: 25/25 GB" (разделитель <br> из ps1)
    disk=$(echo "$w" | grep "^  Disks:" | sed 's/^  Disks: //' | sed 's/[[:space:]]*$//')
    if [[ -z "$disk" ]]; then
        local du df
        du=$(echo "$w" | grep "Disk C:" | grep -oE '[0-9]+ GB used' | grep -oE '[0-9]+' | head -1)
        df=$(echo "$w" | grep "Disk C:" | grep -oE '[0-9]+ GB free' | grep -oE '[0-9]+' | head -1)
        [[ -n "$du" && -n "$df" ]] && disk="C: ${du}/$((du + df)) GB" || disk="—"
    fi

    # Рабочий каталог — из строки "Work:  C:\_Work\video  [C: 105/465 GB]"
    work_path=$(echo "$w" | grep "^  Work:" | sed 's/^  Work:[[:space:]]*//' | sed 's/[[:space:]]*\[.*//' | sed 's/[[:space:]]*$//')

    local m_status s_status
    m_status=$(echo "$w" | grep -E '^[[:space:]]+1_motion_diff[[:space:]]' | awk '{print ($2=="OK")?"OK":"✗"}' | head -1)
    s_status=$(echo "$w" | grep -E '^[[:space:]]+2_send[[:space:]]'        | awk '{print ($2=="OK")?"OK":"✗"}' | head -1)
    scripts="motion_diff ${m_status:-?}<br>send ${s_status:-?}"

    echo "${cpu:-?}|${ram:----}|${disk}|${work_path}|${scripts}"
}

# ── Linux ─────────────────────────────────────────────────────────────────────
sep
# md Linux-таблицы пишем в ДЕТЕРМИНИРОВАННЫЙ temp (--md-out), не в timestamped .output/status
# (--no-save). Так status.sh надёжно читает строки и удаляет файл — без парсинга пути из ANSI-вывода.
LINUX_MD="$(mktemp "${TMPDIR:-/tmp}/vstatus.XXXXXX.md")"
trap 'rm -f "$LINUX_MD" 2>/dev/null' EXIT    # temp удаляется всегда, даже при раннем выходе
if [[ -z "$TS_SERVER" ]] || _is_self "$TS_SERVER"; then
    # мы НА сервере (или TS_SERVER не задан) — локально
    LINUX_LABEL="$(hostname)"; linux_repo="$REPO"
    linux_out=$(bash "$REPO/sh/status/status_linux.sh" --no-save --md-out "$LINUX_MD" 2>&1)
    linux_sys=$(_linux_sys_stats)
else
    # запуск с другой машины — опрашиваем сервер по SSH
    LINUX_LABEL="$TS_SERVER"; linux_repo="$TS_SERVER_REPO"
    _ssh_srv() {
        ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
            "$TS_SERVER_USER@$TS_SERVER" "$@" 2>&1
    }
    _rmd="/tmp/vstatus_remote_$$.md"
    linux_out=$(_ssh_srv "bash '$TS_SERVER_REPO/sh/status/status_linux.sh' --no-save --md-out '$_rmd'")
    _ssh_srv "cat '$_rmd' 2>/dev/null; rm -f '$_rmd'" | tr -d '\r' > "$LINUX_MD"
    linux_sys=$(_ssh_srv "python3 '$TS_SERVER_REPO/scripts/utils/sys_stats.py'" | tr -d '\r')
fi
echo "  LINUX  ($LINUX_LABEL)"
sep
echo "$linux_out" | grep -v "^  → "
IFS='|' read -r linux_cpu linux_ram linux_disk linux_svcs <<< "$linux_sys"

# ── Windows (SSH) ─────────────────────────────────────────────────────────────
_ssh_win() {
    local host="$1" user="$2" script="${3:-$WIN_SCRIPT_CAMERAS_3}"
    ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "$user@$host" \
        "powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"$script\"" 2>&1 \
        | tr -d '\r' | LC_ALL=C grep -a "" \
        || true
}

# ── вспомогательная функция: строки WIN_SVC → markdown-строки таблицы ──────────
# WIN_SVC формат: # WIN_SVC|name|server|status|input|output|last_log|last_idle|stats_10m
_win_svc_md_rows() {
    local win_out="$1" label="${2:-}"   # label — алиас машины из .env (вместо hostname)
    echo "$win_out" | tr -d $'\r' | { grep "^# WIN_SVC|" || true; } | while IFS='|' read -r _ name server status input output last_log last_idle stats_10m; do
        local s_icon
        [[ "$status" == "OK" ]] && s_icon="✓" || s_icon="✗"
        echo "| ${label:-$server} | \`${name}\` | ${s_icon} ${status} | ${input} | ${output} | ${last_log} | ${last_idle} | ${stats_10m} |"
    done
}

# CAMERAS_3
echo ""
sep
echo "  WINDOWS  ($WIN_CAMERAS_3_LABEL / $TS_CAMERAS_3)"
sep
# дефолты + предпроверка доступности (как у CAMERAS_1/DEVELOP): иначе текст ошибки SSH
# непустой → парсится в мусор '?'. Только достучавшись — опрашиваем status_win.ps1.
cam3_cpu="—"; cam3_ram="—"; cam3_disk="—"; cam3_repo="—"; cam3_scripts="недоступна"
cam3_out=""
if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "$TS_CAMERAS_3_USER@$TS_CAMERAS_3" "echo ok" &>/dev/null 2>&1; then
    cam3_out=$(_ssh_win "$TS_CAMERAS_3" "$TS_CAMERAS_3_USER" "$WIN_SCRIPT_CAMERAS_3")
    if [[ -n "$cam3_out" ]]; then
        echo "$cam3_out" | grep -v "^# WIN_SVC|"
        IFS='|' read -r cam3_cpu cam3_ram cam3_disk cam3_repo cam3_scripts <<< "$(_parse_win_sys "$cam3_out")"
    fi
else
    echo "  [недоступна — $TS_CAMERAS_3]"
fi

# DEVELOP
dev_cpu="—"; dev_ram="—"; dev_disk="—"; dev_repo="—"; dev_scripts="недоступна"
dev_out=""
echo ""
sep
echo "  WINDOWS  ($WIN_DEVELOP_LABEL / $TS_DEVELOP)"
sep
if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "$TS_DEVELOP_USER@$TS_DEVELOP" "echo ok" &>/dev/null 2>&1; then
    dev_out=$(_ssh_win "$TS_DEVELOP" "$TS_DEVELOP_USER" "$WIN_SCRIPT_DEVELOP")
    if [[ -n "$dev_out" ]]; then
        echo "$dev_out" | grep -v "^# WIN_SVC|"
        IFS='|' read -r dev_cpu dev_ram dev_disk dev_repo dev_scripts <<< "$(_parse_win_sys "$dev_out")"
    fi
else
    echo "  [недоступна или SSH не настроен]"
fi

# CAMERAS_1
cam1_cpu="—"; cam1_ram="—"; cam1_disk="—"; cam1_repo="—"; cam1_scripts="недоступна"
cam1_out=""
if [[ -n "$TS_CAMERAS_1" ]]; then
    echo ""
    sep
    echo "  WINDOWS  ($WIN_CAMERAS_1_LABEL / $TS_CAMERAS_1)"
    sep
    if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
            "$TS_CAMERAS_1_USER@$TS_CAMERAS_1" "echo ok" &>/dev/null 2>&1; then
        cam1_out=$(_ssh_win "$TS_CAMERAS_1" "$TS_CAMERAS_1_USER" "$WIN_SCRIPT_CAMERAS_1")
        if [[ -n "$cam1_out" ]]; then
            echo "$cam1_out" | grep -v "^# WIN_SVC|"
            IFS='|' read -r cam1_cpu cam1_ram cam1_disk cam1_repo cam1_scripts <<< "$(_parse_win_sys "$cam1_out")"
        fi
    else
        echo "  [недоступна или SSH не настроен]"
    fi
fi

# ── Собираем .md ──────────────────────────────────────────────────────────────

{
    echo "# Pipeline Status"
    echo ""
    echo "_${TS_HUMAN}  ·  v${VER:-?}_"
    echo ""
    echo "## Ресурсы серверов"
    echo ""
    echo "| Машина | CPU | RAM | Диск | Каталог | Статус |"
    echo "|---|---|---|---|---|---|"
    echo "| Linux (${LINUX_LABEL}) | ${linux_cpu} | ${linux_ram} | ${linux_disk} | ${linux_repo} | ${linux_svcs} |"
    echo "| ${WIN_CAMERAS_3_LABEL} | ${cam3_cpu} | ${cam3_ram} | ${cam3_disk} | ${cam3_repo} | ${cam3_scripts} |"
    echo "| ${WIN_DEVELOP_LABEL} | ${dev_cpu} | ${dev_ram} | ${dev_disk} | ${dev_repo} | ${dev_scripts} |"
    [[ -n "$TS_CAMERAS_1" ]] && echo "| ${WIN_CAMERAS_1_LABEL} | ${cam1_cpu} | ${cam1_ram} | ${cam1_disk} | ${cam1_repo} | ${cam1_scripts} |"
    echo ""
    echo "## Сервисы"
    echo ""
    # Заголовок + разделитель (первые 2 строки из Linux .md)
    if [[ -n "$LINUX_MD" && -f "$LINUX_MD" ]]; then
        { grep "^|" "$LINUX_MD" || true; } | head -2
    fi
    # Windows-сервисы ПЕРВЫЕ (они первые в пайплайне: камеры → детекция → отправка)
    _win_svc_md_rows "$cam3_out" "$WIN_CAMERAS_3_LABEL"
    [[ -n "$dev_out" ]] && _win_svc_md_rows "$dev_out" "$WIN_DEVELOP_LABEL"
    [[ -n "$cam1_out" ]] && _win_svc_md_rows "$cam1_out" "$WIN_CAMERAS_1_LABEL"
    # Linux-сервисы (пропускаем header/separator — уже выведены)
    if [[ -n "$LINUX_MD" && -f "$LINUX_MD" ]]; then
        { grep "^|" "$LINUX_MD" || true; } | tail -n +3
    fi
    echo ""
} > "$MD_OUT"

# Удалить промежуточный файл pipeline_status.py
[[ -n "$LINUX_MD" && -f "$LINUX_MD" ]] && rm -f "$LINUX_MD"

# ── Графики стадий → OUT_DIR (best-effort: сбой НЕ ломает статус) ──────────────
# Имена с префиксом ${TS_FILE} — группируются по прогону рядом с ${TS_FILE}.md.
CHART_PREFIX="$OUT_DIR/${TS_FILE}"
_charts=()
# conda-python: plot_meta_charts тянет camera_run → cv2 + matplotlib (системный python3 их не имеет)
_PY="$HOME/miniconda3/envs/conda_video/bin/python"; [[ -x "$_PY" ]] || _PY=python3
# Сервер: YOLO — CPU/утилизация/FPS/длительности (единственная стадия с cpu.csv+run.log)
_ymeta=$(ls -d "$REPO/.output/pipeline/2_yolo_boxes_files/meta/"2026* 2>/dev/null | tail -1 || true)
if [[ -n "${_ymeta:-}" ]] && \
   "$_PY" "$REPO/scripts/utils/plot_meta_charts.py" "$_ymeta" --out "${CHART_PREFIX}_linux_yolo.png" >/dev/null 2>&1; then
    _charts+=("${TS_FILE}_linux_yolo.png")
fi
# Камера CAMERAS_3: charts.png (cpu/fps) + pts_chart.png (дрейф PTS). Рендер ПЕРЕНЕСЁН НА СЕРВЕР —
# камеру не грузим matplotlib (motion_diff пишет только CSV при MOTION_RENDER_CHARTS=0). Тянем лёгкие CSV
# по scp и строим графики здесь через 1_motion_diff.py --regen-from (best-effort: сбой НЕ ломает статус).
# ВАЖНО: периодический флеш кладёт frames/diffs/pts/saves.csv в images/<день>/, а cpu.csv — в meta/<день>/.
# Каталог дня — по дате сервера; при рассинхроне scp не найдёт (graceful).
if [[ -n "${cam3_out:-}" && -n "$TS_CAMERAS_3" ]]; then
    _cday="$(date +%Y%m%d)"
    _cimg="${TS_CAMERAS_3_REPO//\\//}/.output/pipeline/1_motion_diff/images/${_cday}"
    _cmeta="${TS_CAMERAS_3_REPO//\\//}/.output/pipeline/1_motion_diff/meta/${_cday}"
    _ctmp="$(mktemp -d)"
    for _f in frames diffs pts saves; do
        scp -o ConnectTimeout=10 -o BatchMode=yes \
            "$TS_CAMERAS_3_USER@$TS_CAMERAS_3:$_cimg/$_f.csv" "$_ctmp/" >/dev/null 2>&1 || true
    done
    scp -o ConnectTimeout=10 -o BatchMode=yes \
        "$TS_CAMERAS_3_USER@$TS_CAMERAS_3:$_cmeta/cpu.csv" \
        "$TS_CAMERAS_3_USER@$TS_CAMERAS_3:$_cmeta/run_params.json" "$_ctmp/" >/dev/null 2>&1 || true
    if [[ -f "$_ctmp/frames.csv" && -f "$_ctmp/saves.csv" ]] && \
       "$_PY" "$REPO/scripts/pipeline/1_motion_diff.py" --regen-from "$_ctmp" >/dev/null 2>&1; then
        [[ -f "$_ctmp/charts.png" ]]    && cp "$_ctmp/charts.png"    "${CHART_PREFIX}_cam3_cpu_fps.png"  && _charts+=("${TS_FILE}_cam3_cpu_fps.png")
        [[ -f "$_ctmp/pts_chart.png" ]] && cp "$_ctmp/pts_chart.png" "${CHART_PREFIX}_cam3_pts_drift.png" && _charts+=("${TS_FILE}_cam3_pts_drift.png")
    fi
    rm -rf "$_ctmp"
fi

echo ""
echo "  → $MD_OUT"
if (( ${#_charts[@]} )); then
    echo "  Графики → $OUT_DIR :"
    for _c in "${_charts[@]}"; do echo "    • $_c"; done
fi
echo ""
