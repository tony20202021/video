#!/usr/bin/env bash
# Статус pipeline-сервисов — таблица на экран + .md в .output/status/
#
# Usage:
#   ./sh/status/status.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$HOME/miniconda3/envs/conda_video/bin/python"

exec "$PYTHON" "$REPO/scripts/utils/pipeline_status.py" "$@"
