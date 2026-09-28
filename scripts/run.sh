#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || ./scripts/setup.sh
exec .venv/bin/python -m howvision.launcher
