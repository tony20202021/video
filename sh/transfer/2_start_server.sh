#!/bin/bash
# 2_start_server.sh — запуск Transfer Server на Linux-сервере
#
# Usage:
#   ./sh/transfer/2_start_server.sh
#   ./sh/transfer/2_start_server.sh --port 8765

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="$REPO/.env"
PYTHON="${CONDA_PREFIX:-$HOME/miniconda3/envs/conda_video}/bin/python"

# Читаем .env если есть
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "$ENV_FILE" | grep '=' | grep -v '<')
    set +a
fi

if [[ -z "${TRANSFER_API_KEY:-}" ]]; then
    echo "[!] TRANSFER_API_KEY не задан — сервер открыт без аутентификации" >&2
fi

_HOST="${TRANSFER_HOST:-0.0.0.0}"
_PORT="${TRANSFER_PORT:-8765}"

echo "=== Transfer Server (Linux) ==="
echo "Repo:   $REPO"
echo "Python: $PYTHON"
echo "Host:   $_HOST"
echo "Port:   $_PORT"
echo ""

exec "$PYTHON" "$REPO/scripts/transfer/server.py" \
    --host "$_HOST" \
    --port "$_PORT" \
    "$@"
