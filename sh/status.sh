#!/usr/bin/env bash
# Мастер-скрипт статуса: Linux (локально) + Windows-клиенты
# Итоговый .md содержит две таблицы по всем машинам:
#   1. Ресурсы серверов (CPU / RAM / Disk / статус скриптов)
#   2. Linux-сервисы с колонкой 10м-статистики
#
# Usage:
#   ./sh/status.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

TS_WIN_ENTRY=""
TS_WIN_HOME=""
TS_WIN_EVELINA=""
WIN_ENTRY_LABEL="win-entry"
WIN_HOME_LABEL="win-home"
WIN_EVELINA_LABEL="win-evelina"
WIN_SCRIPT='C:\_Work\video\sh\status_win.ps1'
WIN_SCRIPT_EVELINA='C:\Work\video\sh\status_win.ps1'

if [[ -f "$REPO/.env" ]]; then
    while IFS= read -r _line; do
        [[ -z "$_line" || "$_line" =~ ^# ]] && continue
        _val="${_line#*=}"; _val="${_val%%#*}"; _val="${_val%"${_val##*[! ]}"}"
        case "$_line" in
            TS_WIN_ENTRY=*)        TS_WIN_ENTRY="$_val" ;;
            TS_WIN_HOME=*)         TS_WIN_HOME="$_val" ;;
            TS_WIN_EVELINA=*)      TS_WIN_EVELINA="$_val" ;;
            TS_WIN_ENTRY_LABEL=*)  WIN_ENTRY_LABEL="$_val" ;;
            TS_WIN_HOME_LABEL=*)   WIN_HOME_LABEL="$_val" ;;
            TS_WIN_EVELINA_LABEL=*) WIN_EVELINA_LABEL="$_val" ;;
        esac
    done < "$REPO/.env"
fi

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
# Active services
active = 0
for svc in ["video-transfer", "video-yolo", "video-classify", "video-identify"]:
    try:
        r = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True)
        if r.stdout.strip() == "active":
            active += 1
    except Exception:
        pass
print(f"{cpu}%|{ram}|{disk}|{active}/4 active")
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
    scripts="motion_diff ${m_status:-?} / send ${s_status:-?}"

    echo "${cpu:-?}|${ram:----}|${disk}|${work_path}|${scripts}"
}

# ── Linux ─────────────────────────────────────────────────────────────────────
sep
echo "  LINUX  ($(hostname))"
sep

linux_out=$(bash "$REPO/sh/status_linux.sh" 2>&1)
echo "$linux_out" | grep -v "^  → "

linux_md=$(echo "$linux_out" | grep -o '/[^ ]*\.md' | tail -1)
IFS='|' read -r linux_cpu linux_ram linux_disk linux_svcs <<< "$(_linux_sys_stats)"

# ── Windows (SSH) ─────────────────────────────────────────────────────────────
_ssh_win() {
    local host="$1" user="$2" script="${3:-$WIN_SCRIPT}"
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
        local s_icon file_md
        [[ "$status" == "OK" ]] && s_icon="✓" || s_icon="✗"
        # Дата и время в 2 строки: "2026-07-15 07:41" → "2026-07-15<br>07:41"
        file_md=$(echo "$file" | sed 's/ \([0-9][0-9]:[0-9][0-9]\)$/<br>\1/')
        echo "| ${server} | \`${name}\` | ${s_icon} ${status} | ${input} | ${output} | ${file_md} | ${last_log} | ${last_idle} | ${stats_10m} |"
    done
}

# win-entry
echo ""
sep
echo "  WINDOWS  ($WIN_ENTRY_LABEL / $TS_WIN_ENTRY)"
sep
julie2_out=$(_ssh_win "$TS_WIN_ENTRY" "julia")
if [[ -n "$julie2_out" ]]; then
    echo "$julie2_out" | grep -v "^# WIN_SVC|"
    IFS='|' read -r julie2_cpu julie2_ram julie2_disk julie2_repo julie2_scripts <<< "$(_parse_win_sys "$julie2_out")"
else
    echo "  [недоступна — $TS_WIN_ENTRY]"
    julie2_cpu="—"; julie2_ram="—"; julie2_disk="—"; julie2_repo="—"; julie2_scripts="недоступна"
fi

# win-home
tony8_cpu="—"; tony8_ram="—"; tony8_disk="—"; tony8_repo="—"; tony8_scripts="недоступна"
tony8_out=""
echo ""
sep
echo "  WINDOWS  ($WIN_HOME_LABEL / $TS_WIN_HOME)"
sep
if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        "tony@$TS_WIN_HOME" "echo ok" &>/dev/null 2>&1; then
    tony8_out=$(_ssh_win "$TS_WIN_HOME" "tony")
    if [[ -n "$tony8_out" ]]; then
        echo "$tony8_out" | grep -v "^# WIN_SVC|"
        IFS='|' read -r tony8_cpu tony8_ram tony8_disk tony8_repo tony8_scripts <<< "$(_parse_win_sys "$tony8_out")"
    fi
else
    echo "  [недоступна или SSH не настроен]"
fi

# win-evelina
evelina_cpu="—"; evelina_ram="—"; evelina_disk="—"; evelina_repo="—"; evelina_scripts="недоступна"
evelina_out=""
if [[ -n "$TS_WIN_EVELINA" ]]; then
    echo ""
    sep
    echo "  WINDOWS  ($WIN_EVELINA_LABEL / $TS_WIN_EVELINA)"
    sep
    if ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
            "evelina@$TS_WIN_EVELINA" "echo ok" &>/dev/null 2>&1; then
        evelina_out=$(_ssh_win "$TS_WIN_EVELINA" "evelina" "$WIN_SCRIPT_EVELINA")
        if [[ -n "$evelina_out" ]]; then
            echo "$evelina_out" | grep -v "^# WIN_SVC|"
            IFS='|' read -r evelina_cpu evelina_ram evelina_disk evelina_repo evelina_scripts <<< "$(_parse_win_sys "$evelina_out")"
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
    echo "| ${WIN_ENTRY_LABEL} | ${julie2_cpu} | ${julie2_ram} | ${julie2_disk} | ${julie2_repo} | ${julie2_scripts} |"
    echo "| ${WIN_HOME_LABEL} | ${tony8_cpu} | ${tony8_ram} | ${tony8_disk} | ${tony8_repo} | ${tony8_scripts} |"
    [[ -n "$TS_WIN_EVELINA" ]] && echo "| ${WIN_EVELINA_LABEL} | ${evelina_cpu} | ${evelina_ram} | ${evelina_disk} | ${evelina_repo} | ${evelina_scripts} |"
    echo ""
    echo "## Сервисы"
    echo ""
    # Заголовок + разделитель (первые 2 строки из Linux .md)
    if [[ -n "$linux_md" && -f "$linux_md" ]]; then
        grep "^|" "$linux_md" | head -2
    fi
    # Windows-сервисы ПЕРВЫЕ (они первые в пайплайне: камеры → детекция → отправка)
    _win_svc_md_rows "$julie2_out"
    [[ -n "$tony8_out" ]] && _win_svc_md_rows "$tony8_out"
    [[ -n "$evelina_out" ]] && _win_svc_md_rows "$evelina_out"
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
