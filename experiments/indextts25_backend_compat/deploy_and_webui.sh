#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${REPO_ROOT}/.venv-indextts25"
MODEL_DIR="${REPO_ROOT}/models/IndexTTS-2.5"
DATA_DIR="${REPO_ROOT}/runtime/indextts25"
API_HOST="127.0.0.1"
API_PORT="8092"
WEB_HOST="0.0.0.0"
WEB_PORT="7860"
GPU_DEVICE="0"
VLLM_VERSION="0.27.0"
MODEL_ID="IndexTeam/IndexTTS-2.5"
WAV2VEC_ID="facebook/w2v-bert-2.0"
CAMPPLUS_ID="funasr/campplus"
BIGVGAN_ID="nvidia/bigvgan_v2_22khz_80band_256x"

mkdir -p "${MODEL_DIR}" "${DATA_DIR}/logs" "${DATA_DIR}/results" \
  "${DATA_DIR}/speakers" "${DATA_DIR}/cache" \
  "${DATA_DIR}/cache/speaker-conditioning" "${DATA_DIR}/cache/torchinductor" \
  "${DATA_DIR}/cache/triton" "${DATA_DIR}/cache/cuda"

log() { printf '[IndexTTS-2.5] %s\n' "$*"; }
fail() { printf '[IndexTTS-2.5] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "This deployment script requires Linux."
command -v nvidia-smi >/dev/null || fail "nvidia-smi is not available. Install the NVIDIA driver first."
command -v curl >/dev/null || fail "curl is required."

log "GPU preflight"
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader | tee "${DATA_DIR}/logs/gpu-preflight.txt"

if ! command -v uv >/dev/null; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null || fail "uv installation did not put uv on PATH."

log "Creating or reusing the project-local Python 3.11 environment"
uv python install 3.11
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv --python 3.11 "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Installing or verifying vLLM ${VLLM_VERSION}"
uv pip install "vllm==${VLLM_VERSION}" --torch-backend=auto

log "Applying the Python 3.11 compatibility fix for FlashInfer 0.6.16"
python "${SCRIPT_DIR}/src/indextts25_compat/patch_flashinfer.py"
python -c 'import flashinfer.comm; print("FlashInfer communication module import: OK")'

log "Installing this pinned vLLM-Omni source with IndexTTS dependencies"
uv pip install -e "${REPO_ROOT}[indextts2]"
uv pip install -e "${SCRIPT_DIR}[webui]"
uv pip install 'huggingface_hub[cli]'

python - <<'PY'
import torch, vllm
print("torch:", torch.__version__, "runtime CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
PY

hf_download() {
  local repo_id="$1"
  local destination="$2"
  shift 2
  local args=(download "${repo_id}" "$@" --local-dir "${destination}")
  if [[ -n "${HF_TOKEN:-}" ]]; then
    args+=(--token "${HF_TOKEN}")
  fi
  hf "${args[@]}"
}

require_model_file() {
  local relative_path="$1"
  [[ -s "${MODEL_DIR}/${relative_path}" ]] \
    || fail "Required model asset is missing or empty: ${MODEL_DIR}/${relative_path}"
}

log "Downloading or verifying the ${MODEL_ID} checkpoint bundle"
hf_download "${MODEL_ID}" "${MODEL_DIR}"

# IndexTTS loads these external models on startup, but IndexTeam/IndexTTS-2.5
# does not bundle them. Keep exact local layouts recognized by vLLM-Omni so
# worker processes never perform surprise downloads during model warmup.
log "Downloading or verifying Wav2Vec2-BERT prompt encoder assets"
hf_download "${WAV2VEC_ID}" "${MODEL_DIR}/w2v-bert-2.0" \
  config.json model.safetensors preprocessor_config.json

log "Downloading or verifying the CAMPPlus speaker encoder"
hf_download "${CAMPPLUS_ID}" "${MODEL_DIR}" campplus_cn_common.bin

log "Downloading or verifying the BigVGAN vocoder"
hf_download "${BIGVGAN_ID}" "${MODEL_DIR}/bigvgan" \
  config.json bigvgan_generator.pt

for model_file in \
  config.yaml gpt.pth codec.pth s2mel.pth wav2vec2bert_stats.pt \
  multilingual_zh_ja_yue_char_del.tiktoken \
  qwen0.6bemo4-merge/config.json qwen0.6bemo4-merge/model.safetensors \
  w2v-bert-2.0/config.json w2v-bert-2.0/model.safetensors \
  w2v-bert-2.0/preprocessor_config.json campplus_cn_common.bin \
  bigvgan/config.json bigvgan/bigvgan_generator.pt; do
  require_model_file "${model_file}"
done
log "Main checkpoint and all external runtime models are ready"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export HF_HOME="${DATA_DIR}/cache/huggingface"
export SPEAKER_SAMPLES_DIR="${DATA_DIR}/speakers"
export SPEAKER_CACHE_DIR="${DATA_DIR}/cache/speaker-conditioning"
export SPEAKER_CACHE_VERSION="indextts25-v1"
export TORCHINDUCTOR_CACHE_DIR="${DATA_DIR}/cache/torchinductor"
export TRITON_CACHE_DIR="${DATA_DIR}/cache/triton"
export CUDA_CACHE_PATH="${DATA_DIR}/cache/cuda"
export PYTHONUNBUFFERED=1

API_LOG="${DATA_DIR}/logs/vllm-api.log"
WEB_LOG="${DATA_DIR}/logs/webui.log"
DEPLOY_CONFIG="${REPO_ROOT}/vllm_omni/deploy/indextts2_5.yaml"

API_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    log "Stopping vLLM-Omni API (PID ${API_PID})"
    kill "${API_PID}" 2>/dev/null || true
    wait "${API_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

log "Starting vLLM-Omni API on ${API_HOST}:${API_PORT}"
vllm serve "${MODEL_DIR}" \
  --omni \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --trust-remote-code \
  --log-stats \
  --deploy-config "${DEPLOY_CONFIG}" >"${API_LOG}" 2>&1 &
API_PID=$!

log "Waiting for model startup; Stage 1 torch.compile can take several minutes"
READY=0
for _ in $(seq 1 360); do
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    tail -n 200 "${API_LOG}" >&2 || true
    fail "vLLM-Omni exited during startup. Full log: ${API_LOG}"
  fi
  if curl -fsS "http://${API_HOST}:${API_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 5
done
[[ "${READY}" == "1" ]] || fail "API was not ready after 30 minutes. See ${API_LOG}"

log "API ready. Starting test UI at http://${WEB_HOST}:${WEB_PORT}"
log "Results: ${DATA_DIR}/results"
log "API log: ${API_LOG}"

python "${SCRIPT_DIR}/webui.py" \
  --api-base "http://${API_HOST}:${API_PORT}" \
  --host "${WEB_HOST}" \
  --port "${WEB_PORT}" \
  --model "${MODEL_DIR}" \
  --results-dir "${DATA_DIR}/results" \
  2>&1 | tee "${WEB_LOG}"
