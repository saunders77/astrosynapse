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

# Replace a stale dashboard/API pair from this project, but never interrupt an
# unrelated service that happens to use one of the local control-center ports.
typeset -a STALE_PIDS
STALE_PIDS=()
for SERVICE_PORT in 3000 8765; do
  for SERVICE_PID in "${(@f)$(/usr/sbin/lsof -t -nP -iTCP:"${SERVICE_PORT}" -sTCP:LISTEN 2>/dev/null)}"; do
    [[ -z "${SERVICE_PID}" ]] && continue
    SERVICE_CWD="$(/usr/sbin/lsof -a -p "${SERVICE_PID}" -d cwd -Fn 2>/dev/null | /usr/bin/sed -n 's/^n//p')"
    if [[ "${SERVICE_CWD}" != "${PROJECT_DIR}" ]]; then
      print -u2 "Port ${SERVICE_PORT} is in use by an unrelated process (PID ${SERVICE_PID}, ${SERVICE_CWD:-unknown directory})."
      print -u2 "Stop that process or change its port before starting Astrosynapse 2."
      exit 1
    fi
    STALE_PIDS+=("${SERVICE_PID}")
  done
done

if (( ${#STALE_PIDS[@]} )); then
  print "Stopping stale Astrosynapse 2 services..."
  for SERVICE_PID in "${STALE_PIDS[@]}"; do
    kill "${SERVICE_PID}" 2>/dev/null || true
  done
  for _ in {1..40}; do
    if ! /usr/sbin/lsof -nP -iTCP:3000 -sTCP:LISTEN >/dev/null 2>&1 \
      && ! /usr/sbin/lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
fi

for SERVICE_PORT in 3000 8765; do
  if /usr/sbin/lsof -nP -iTCP:"${SERVICE_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    print -u2 "Port ${SERVICE_PORT} did not clear after stopping the prior control center."
    exit 1
  fi
done

# Keep the production bundle synchronized with dashboard source changes.
npm run build

"${PROJECT_DIR}/.venv/bin/python" -m astro2.server &
BACKEND_PID=$!
node "${PROJECT_DIR}/node_modules/.bin/vinext" start --hostname 127.0.0.1 &
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

for _ in {1..40}; do
  if /usr/sbin/lsof -nP -iTCP:3000 -sTCP:LISTEN >/dev/null 2>&1 \
    && /usr/sbin/lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
if ! /usr/sbin/lsof -nP -iTCP:3000 -sTCP:LISTEN >/dev/null 2>&1 \
  || ! /usr/sbin/lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  print -u2 "Astrosynapse 2 did not start correctly; see the messages above."
  exit 1
fi
open "http://127.0.0.1:3000/"
print "Astrosynapse 2 is running at http://127.0.0.1:3000/"
print "Keep this window open. Press Control-C here for a safe shutdown."
wait
