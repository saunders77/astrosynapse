#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
if [[ -z "${PROJECT_DIR}" || ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  print -u2 "Could not locate the Astrosynapse 2 project directory."
  exit 1
fi
cd "${PROJECT_DIR}"

if [[ ! -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  print -u2 "Run setup.command once before starting Astrosynapse 2."
  exit 1
fi
if ! command -v node >/dev/null 2>&1 && [[ -s "${HOME}/.nvm/nvm.sh" ]]; then
  source "${HOME}/.nvm/nvm.sh"
  nvm use 22 >/dev/null
fi
if ! command -v npm >/dev/null 2>&1; then
  print -u2 "Node.js is unavailable. Run setup.command from Terminal first."
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/backend"
export ASTRO2_HOST="127.0.0.1"
export ASTRO2_PORT="8765"

"${PROJECT_DIR}/.venv/bin/python" -m astro2.server &
BACKEND_PID=$!
npm run start &
FRONTEND_PID=$!
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w "${BACKEND_PID}" &
  CAFFEINATE_PID=$!
else
  CAFFEINATE_PID=""
fi

cleanup() {
  if [[ -n "${CAFFEINATE_PID}" ]]; then
    kill "${CAFFEINATE_PID}" 2>/dev/null || true
  fi
  kill "${FRONTEND_PID}" "${BACKEND_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
open "http://127.0.0.1:3000/"
print "Astrosynapse 2 is running at http://127.0.0.1:3000/"
print "Keep this window open. Press Control-C here for a safe shutdown."
wait
