#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

cd web
npm install
npm run build
cd ..

printf '\nNeural Atlas is installed. Start it with:\n\n'
printf '  source .venv/bin/activate\n'
printf '  neural-atlas\n\n'
printf 'Then open http://127.0.0.1:8000\n'
