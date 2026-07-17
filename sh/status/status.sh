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
        case "$_line" in
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

# Путь к status_win.ps1 на каждой машине = <repo>\sh\status\status_win.ps1
WIN_SCRIPT_CAMERAS_3="$TS_CAMERAS_3_REPO\sh\status\status_win.ps1"
WIN_SCRIPT_DEVELOP="$TS_DEVELOP_REPO\sh\status\status_win.ps1"
WIN_SCRIPT_CAMERAS_1="$TS_CAMERAS_1_REPO\sh\status\status_win.ps1"

OUT_DIR="$REPO/.output/status/$(date +%Y-%m-%d)"
mkdir -p "$OUT_DIR"
TS_FILE=$(date +"%Y%m%d_%H%M%S")
TS_HUMAN=$(date +"%Y-%m-%d %H:%M:%S")
MD_OUT="$OUT_DIR/${TS_FILE}.md"

sep() { echo "════════════════════════════════════════════════════════"; }

# ── вспомогательная функция: сбор системных метрик ────────────────────────────
# Печатает одну строку: "cpu%|ram_used/ram_total MB|disk_used/disk_total GB"
_linux_sys_stats() {
    python3 - <<'PYEOF'
import subprocess, re
# CPU
cpu = "?"
try:
    r = subprocess.run(["top", "-bn1"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        m = re.search(r"([\d.]+)\s*id", line)
        if m:
            cpu = str(round(100 - float(m.group(1))))
            break
except Exception:
    pass
# RAM
ram = "?"
try:
    r = subprocess.run(["free", "-m"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("Mem:"):
            p = line.split()
            ram = f"{round(int(p[2])/1024,1)}/{round(int(p[1])/1024,1)} GB"
            break
except Exception:
    pass
# Disk
disk = "?"
try:
    import os
    r = subprocess.run(["df", "-BG", os.getcwd()], capture_output=True, text=True)
    p = r.stdout.splitlines()[1].split()
    used = int(p[2].rstrip("G"))
    total = int(p[1].rstrip("G"))
    disk = f"{used}/{total} GB"
except Exception:
    pass
# Active services — list each by name
svc_lines = []
for svc in ["video-transfer", "video-yolo", "video-classify", "video-identify"]:
    try:
        r = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True)
        icon = "✓" if r.stdout.strip() == "active" else "✗"
    except Exception:
        icon = "?"
    svc_lines.append(f"{icon} {svc}")
print(f"{cpu}%|{ram}|{disk}|{'<br>'.join(svc_lines)}")
PYEOF
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
echo "  LINUX  ($(hostname))"
sep

linux_out=$(bash "$REPO/sh/status/status_linux.sh" 2>&1)
echo "$linux_out" | grep -v "^  → "

linux_md=$(echo "$linux_out" | grep -o '/[^ ]*\.md' | tail -1)
IFS='|' read -r linux_cpu linux_ram linux_disk linux_svcs <<< "$(_linux_sys_stats)"

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
# WIN_SVC формат: # WIN_SVC|name|server|status|input|output|file|last_log|last_idle|stats_10m
_win_svc_md_rows() {
    local win_out="$1"
    echo "$win_out" | tr -d $'\r' | grep "^# WIN_SVC|" | while IFS='|' read -r _ name server status input output file last_log last_idle stats_10m; do
        local s_icon
        [[ "$status" == "OK" ]] && s_icon="✓" || s_icon="✗"
        echo "| ${server} | \`${name}\` | ${s_icon} ${status} | ${input} | ${output} | ${file} | ${last_log} | ${last_idle} | ${stats_10m} |"
    done
}

# CAMERAS_3
echo ""
sep
echo "  WINDOWS  ($WIN_CAMERAS_3_LABEL / $TS_CAMERAS_3)"
sep
cam3_out=$(_ssh_win "$TS_CAMERAS_3" "$TS_CAMERAS_3_USER" "$WIN_SCRIPT_CAMERAS_3")
if [[ -n "$cam3_out" ]]; then
    echo "$cam3_out" | grep -v "^# WIN_SVC|"
    IFS='|' read -r cam3_cpu cam3_ram cam3_disk cam3_repo cam3_scripts <<< "$(_parse_win_sys "$cam3_out")"
else
    echo "  [недоступна — $TS_CAMERAS_3]"
    cam3_cpu="—"; cam3_ram="—"; cam3_disk="—"; cam3_repo="—"; cam3_scripts="недоступна"
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
    echo "_${TS_HUMAN}_"
    echo ""
    echo "## Ресурсы серверов"
    echo ""
    echo "| Машина | CPU | RAM | Диск | Каталог | Статус |"
    echo "|---|---|---|---|---|---|"
    echo "| Linux ($(hostname)) | ${linux_cpu} | ${linux_ram} | ${linux_disk} | ${REPO} | ${linux_svcs} |"
    echo "| ${WIN_CAMERAS_3_LABEL} | ${cam3_cpu} | ${cam3_ram} | ${cam3_disk} | ${cam3_repo} | ${cam3_scripts} |"
    echo "| ${WIN_DEVELOP_LABEL} | ${dev_cpu} | ${dev_ram} | ${dev_disk} | ${dev_repo} | ${dev_scripts} |"
    [[ -n "$TS_CAMERAS_1" ]] && echo "| ${WIN_CAMERAS_1_LABEL} | ${cam1_cpu} | ${cam1_ram} | ${cam1_disk} | ${cam1_repo} | ${cam1_scripts} |"
    echo ""
    echo "## Сервисы"
    echo ""
    # Заголовок + разделитель (первые 2 строки из Linux .md)
    if [[ -n "$linux_md" && -f "$linux_md" ]]; then
        grep "^|" "$linux_md" | head -2
    fi
    # Windows-сервисы ПЕРВЫЕ (они первые в пайплайне: камеры → детекция → отправка)
    _win_svc_md_rows "$cam3_out"
    [[ -n "$dev_out" ]] && _win_svc_md_rows "$dev_out"
    [[ -n "$cam1_out" ]] && _win_svc_md_rows "$cam1_out"
    # Linux-сервисы (пропускаем header/separator — уже выведены)
    if [[ -n "$linux_md" && -f "$linux_md" ]]; then
        grep "^|" "$linux_md" | tail -n +3
    fi
    echo ""
} > "$MD_OUT"

# Удалить промежуточный файл pipeline_status.py
[[ -n "$linux_md" && -f "$linux_md" ]] && rm -f "$linux_md"

echo ""
echo "  → $MD_OUT"
echo ""
