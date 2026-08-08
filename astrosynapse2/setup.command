#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
if [[ -z "${PROJECT_DIR}" || ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  print -u2 "Could not locate the Astrosynapse 2 project directory."
  exit 1
fi

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/.tools" "${PROJECT_DIR}/.cache/uv"
export UV_CACHE_DIR="${PROJECT_DIR}/.cache/uv"

if [[ ! -x "${PROJECT_DIR}/.tools/uv" ]]; then
  INSTALLER="${TMPDIR:-/tmp}/astrosynapse2-uv-install.sh"
  curl -LsSf "https://astral.sh/uv/0.11.32/install.sh" -o "${INSTALLER}"
  sh -n "${INSTALLER}"
  UV_UNMANAGED_INSTALL="${PROJECT_DIR}/.tools" sh "${INSTALLER}"
fi

"${PROJECT_DIR}/.tools/uv" sync --extra dev --python 3.12

if ! command -v node >/dev/null 2>&1 && [[ -s "${HOME}/.nvm/nvm.sh" ]]; then
  source "${HOME}/.nvm/nvm.sh"
  nvm use 22 >/dev/null
fi
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  print -u2 "Node.js 22 or newer is required for the dashboard. Install it, then run setup.command again."
  exit 1
fi

npm ci
npm run build

print ""
print "Astrosynapse 2 is ready. Double-click start.command to launch it."
