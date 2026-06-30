#!/bin/bash
# start_server.sh — запуск Transfer Server на Linux-сервере
#
# Usage:
#   ./sh/transfer/start_server.sh
#   TRANSFER_API_KEY=secret ./sh/transfer/start_server.sh --port 8765

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="$REPO/.env"
PYTHON="${CONDA_PREFIX:-$HOME/miniconda3/envs/conda_video}/bin/python"

# Читаем .env если есть
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "$ENV_FILE" | grep '=')
    set +a
fi

if [[ -z "${TRANSFER_API_KEY:-}" ]]; then
    echo "[!] TRANSFER_API_KEY не задан — сервер открыт без аутентификации" >&2
fi

echo "=== Transfer Server ==="
echo "Repo:   $REPO"
echo "Python: $PYTHON"
echo ""

exec "$PYTHON" "$REPO/scripts/transfer/server.py" "$@"
