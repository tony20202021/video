#!/bin/bash
# 1_send.sh — постоянно следит за каталогом и отправляет новые файлы на Transfer Server
#
# Usage:
#   ./sh/transfer/1_send.sh <watch_dir>
#   ./sh/transfer/1_send.sh .output/pipeline/1_motion_diff/run_XXX/images
#   ./sh/transfer/1_send.sh <watch_dir> --ext jpg,png
#   ./sh/transfer/1_send.sh <watch_dir> --server http://1.2.3.4:8765 --key SECRET

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="$REPO/.env"
PYTHON="${CONDA_PREFIX:-$HOME/miniconda3/envs/conda_video}/bin/python"
SCRIPT="$REPO/scripts/transfer/client.py"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <watch_dir> [--ext jpg] [--run-root <path>] [--server <url>] [--key <key>]"
    exit 1
fi

# Читаем .env если есть
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source <(grep -v '^\s*#' "$ENV_FILE" | grep '=')
    set +a
fi

echo "=== 1_send (transfer watch) ==="
echo "WatchDir: $1"
echo ""

export PYTHONIOENCODING=utf-8

exec "$PYTHON" "$SCRIPT" watch "$@"
