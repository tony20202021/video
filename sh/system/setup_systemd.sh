#!/usr/bin/env bash
# Генерация и установка systemd-сервисов для video surveillance.
#
# Использование:
#   ./sh/system/setup_systemd.sh                    # установить и запустить основные сервисы
#   ./sh/system/setup_systemd.sh --with-data        # + data-pipeline (watch-скрипты обучения)
#   ./sh/system/setup_systemd.sh --remove           # удалить все сервисы
#   ./sh/system/setup_systemd.sh --status           # статус всех сервисов
#   ./sh/system/setup_systemd.sh --logs video-classify           # последние 100 строк
#   ./sh/system/setup_systemd.sh --logs video-classify --follow  # следить за логами
#
# Основные сервисы:
#   video-transfer   — Transfer Server (приём файлов с Windows)
#   video-yolo       — YOLO детекция кропов (watch-режим)
#   video-classify   — Классификация групп, Модель 1 (watch-режим)
#   video-smooth     — Темпоральное сглаживание классов Модели 1 (watch-режим)
#   video-identify   — Идентификация жителей, Модель 2 (watch-режим)
#
# Опциональные (--with-data, только во время обучения):
#   video-data-check   — проверка дублей кропов с датасетом
#   video-data-dedup   — дедупликация внутри new/
#   video-data-labels  — авто-labels из инференса

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SYSTEMD_DIR="/etc/systemd/system"

# ── Аргументы ─────────────────────────────────────────────────────────────────

REMOVE=0
STATUS=0
WITH_DATA=0
LOGS_SVC=""
LOGS_FOLLOW=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remove)    REMOVE=1;       shift ;;
        --status)    STATUS=1;       shift ;;
        --with-data) WITH_DATA=1;    shift ;;
        --logs)      LOGS_SVC="$2";  shift 2 ;;
        --follow|-f) LOGS_FOLLOW=1;  shift ;;
        *) echo "[!] Неизвестный аргумент: $1" >&2; exit 1 ;;
    esac
done

# ── Конфигурация ──────────────────────────────────────────────────────────────

_ef() {
    local k="$1" d="$2"
    local v; v=$(grep -E "^\s*${k}\s*=" "$REPO/.env" 2>/dev/null | tail -1 | sed 's/.*=[[:space:]]*//' | tr -d $'\r')
    echo "${v:-$d}"
}

CONDA_ENV="$(_ef CONDA_ENV conda_video)"
PYTHON="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
USER_NAME="$(whoami)"

CORE_SERVICES=(
    video-transfer
    video-yolo
    video-classify
    video-smooth
    video-identify
    video-smooth-identity
)

DATA_SERVICES=(
    video-data-check
    video-data-dedup
    video-data-labels
)

ALL_SERVICES=("${CORE_SERVICES[@]}")
[[ "$WITH_DATA" -eq 1 ]] && ALL_SERVICES+=("${DATA_SERVICES[@]}")

# ── Статус ────────────────────────────────────────────────────────────────────

if [[ "$STATUS" -eq 1 ]]; then
    echo "=== Статус сервисов video ==="
    for svc in "${CORE_SERVICES[@]}" "${DATA_SERVICES[@]}"; do
        state=$(systemctl is-active   "$svc" 2>/dev/null || echo "inactive")
        enabled=$(systemctl is-enabled "$svc" 2>/dev/null || echo "disabled")
        printf "  %-24s  %-12s  %s\n" "$svc" "$state" "$enabled"
    done
    exit 0
fi

# ── Логи ──────────────────────────────────────────────────────────────────────

if [[ -n "$LOGS_SVC" ]]; then
    if [[ "$LOGS_FOLLOW" -eq 1 ]]; then
        exec journalctl -u "$LOGS_SVC" -f
    else
        exec journalctl -u "$LOGS_SVC" -n 100 --no-pager
    fi
fi

# ── Удаление ──────────────────────────────────────────────────────────────────

if [[ "$REMOVE" -eq 1 ]]; then
    echo "=== Удаление сервисов video ==="
    for svc in "${CORE_SERVICES[@]}" "${DATA_SERVICES[@]}"; do
        f="$SYSTEMD_DIR/$svc.service"
        if [[ -f "$f" ]]; then
            sudo systemctl stop    "$svc" 2>/dev/null || true
            sudo systemctl disable "$svc" 2>/dev/null || true
            sudo rm -f "$f"
            echo "  удалён: $svc"
        else
            echo "  пропуск (не установлен): $svc"
        fi
    done
    sudo systemctl daemon-reload
    echo "Готово."
    exit 0
fi

# ── Проверки ──────────────────────────────────────────────────────────────────

echo "=== Установка systemd-сервисов video ==="
echo "  Repo:    $REPO"
echo "  User:    $USER_NAME"
echo "  Python:  $PYTHON"
echo "  Conda:   $CONDA_ENV"
echo "  Сервисы: ${ALL_SERVICES[*]}"
echo ""

if [[ ! -x "$PYTHON" ]]; then
    echo "[!] Python не найден: $PYTHON" >&2
    echo "    Проверьте CONDA_ENV в .env и наличие conda-окружения." >&2
    exit 1
fi

# ── Генерация и установка сервис-файлов ───────────────────────────────────────

_install() {
    local name="$1" content="$2"
    echo "$content" | sudo tee "$SYSTEMD_DIR/$name.service" > /dev/null
    echo "  → $SYSTEMD_DIR/$name.service"
}

# ─── video-transfer ───────────────────────────────────────────────────────────
_install "video-transfer" "[Unit]
Description=Video — Transfer Server (приём файлов с Windows)
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
ExecStart=/bin/bash $REPO/sh/transfer/1_start_server.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"

# ─── video-yolo ───────────────────────────────────────────────────────────────
_install "video-yolo" "[Unit]
Description=Video — YOLO детекция кропов (watch-режим)
After=network.target video-transfer.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
ExecStart=/bin/bash $REPO/sh/pipeline/2_yolo_boxes_files.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"

# ─── video-classify ───────────────────────────────────────────────────────────
_install "video-classify" "[Unit]
Description=Video — Классификация групп, Модель 1 (watch-режим)
After=network.target video-yolo.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
ExecStart=/bin/bash $REPO/sh/pipeline/3_classify_groups.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"

# ─── video-smooth (темпоральное сглаживание классов, между classify и identify) ──
_install "video-smooth" "[Unit]
Description=Video — Темпоральное сглаживание классов Модели 1 (watch-режим)
After=network.target video-classify.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
Nice=10
IOSchedulingClass=idle
ExecStart=/bin/bash $REPO/sh/pipeline/3b_smooth_groups.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"

# ─── video-identify ───────────────────────────────────────────────────────────
_install "video-identify" "[Unit]
Description=Video — Идентификация жителей, Модель 2 (watch-режим)
After=network.target video-classify.service video-smooth.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
ExecStart=/bin/bash $REPO/sh/pipeline/4_identify_residents.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"

# ─── video-smooth-identity (сглаживание идентификации жителей, ПОСЛЕ identify) ──
_install "video-smooth-identity" "[Unit]
Description=Video — Темпоральное сглаживание идентификации жителей, Модель 2 (watch-режим)
After=network.target video-identify.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
Nice=10
IOSchedulingClass=idle
ExecStart=/bin/bash $REPO/sh/pipeline/4b_smooth_identity.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"

# ─── data-pipeline (опциональные) ────────────────────────────────────────────

if [[ "$WITH_DATA" -eq 1 ]]; then
    _install "video-data-check" "[Unit]
Description=Video — проверка дублей кропов с датасетом (watch-режим)
After=network.target video-yolo.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
ExecStart=/bin/bash $REPO/sh/train/groups/1_1_dataset_groups_check.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"

    _install "video-data-dedup" "[Unit]
Description=Video — дедупликация внутри new/ (watch-режим)
After=network.target video-data-check.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
ExecStart=/bin/bash $REPO/sh/train/groups/1_2_dataset_groups_check_new.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"

    _install "video-data-labels" "[Unit]
Description=Video — авто-labels из инференса (watch-режим)
After=network.target video-classify.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
ExecStart=/bin/bash $REPO/sh/train/groups/1_3_dataset_groups_inference_labels.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"
fi

# ── Активация ─────────────────────────────────────────────────────────────────

echo ""
echo "Перезагрузка daemon и активация…"
sudo systemctl daemon-reload

for svc in "${ALL_SERVICES[@]}"; do
    sudo systemctl enable  "$svc"
    sudo systemctl restart "$svc"
    state=$(systemctl is-active "$svc" 2>/dev/null || echo "?")
    echo "  $svc → $state"
done

echo ""
echo "Команды управления:"
echo "  ./sh/system/setup_systemd.sh --status"
echo "  ./sh/system/setup_systemd.sh --logs video-classify --follow"
echo "  ./sh/system/setup_systemd.sh --remove"
echo "  sudo systemctl {start|stop|restart|status} video-classify"
echo "  journalctl -u video-classify -f"
