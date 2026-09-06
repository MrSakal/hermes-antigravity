#!/usr/bin/env bash
set -euo pipefail
exec python3 "$(dirname "$0")/e2e_live.py" --port "${PORT:-8799}" "$@"
