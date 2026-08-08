#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "${PROJECT_DIR}"
if ! command -v node >/dev/null 2>&1 && [[ -s "${HOME}/.nvm/nvm.sh" ]]; then
  source "${HOME}/.nvm/nvm.sh"
  nvm use 22 >/dev/null
fi
export PYTHONPATH="${PROJECT_DIR}/backend"

"${PROJECT_DIR}/.venv/bin/python" -m astro2.server &
BACKEND_PID=$!
npm run dev &
FRONTEND_PID=$!

cleanup() {
  kill "${FRONTEND_PID}" "${BACKEND_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait
