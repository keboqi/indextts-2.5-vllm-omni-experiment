#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${INDEXTTS25_DATA_DIR:-${REPO_ROOT}/runtime/indextts25}"
API_HOST="${INDEXTTS25_HOST:-127.0.0.1}"
API_PORT="${INDEXTTS25_PORT:-8092}"
WEB_HOST="${INDEXTTS25_WEB_HOST:-0.0.0.0}"
WEB_PORT="${INDEXTTS25_WEB_PORT:-7860}"
MODEL_DIR="${INDEXTTS25_MODEL_DIR:-${REPO_ROOT}/models/IndexTTS-2.5}"
VENV_DIR="${INDEXTTS25_VENV_DIR:-${REPO_ROOT}/.venv-indextts25}"

mkdir -p "${DATA_DIR}/logs" "${DATA_DIR}/results"
API_LOG="${DATA_DIR}/logs/vllm-api.log"
WEB_LOG="${DATA_DIR}/logs/webui.log"

API_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
    wait "${API_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

INDEXTTS25_INSTALL_WEBUI=1 bash "${SCRIPT_DIR}/serve_api.sh" >"${API_LOG}" 2>&1 &
API_PID=$!

printf '[IndexTTS-2.5] Waiting for the API on %s:%s\n' "${API_HOST}" "${API_PORT}"
READY=0
for _ in $(seq 1 720); do
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    tail -n 200 "${API_LOG}" >&2 || true
    exit 1
  fi
  if curl -fsS "http://${API_HOST}:${API_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 5
done
[[ "${READY}" == "1" ]] || { tail -n 200 "${API_LOG}" >&2 || true; exit 1; }

"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/webui.py" \
  --api-base "http://${API_HOST}:${API_PORT}" \
  --host "${WEB_HOST}" \
  --port "${WEB_PORT}" \
  --model "${MODEL_DIR}" \
  --results-dir "${DATA_DIR}/results" \
  2>&1 | tee "${WEB_LOG}"
