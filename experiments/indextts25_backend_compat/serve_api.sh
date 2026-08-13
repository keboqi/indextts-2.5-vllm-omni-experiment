#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${INDEXTTS25_VENV_DIR:-${REPO_ROOT}/.venv-indextts25}"
MODEL_DIR="${INDEXTTS25_MODEL_DIR:-${REPO_ROOT}/models/IndexTTS-2.5}"
DATA_DIR="${INDEXTTS25_DATA_DIR:-${REPO_ROOT}/runtime/indextts25}"
API_HOST="${INDEXTTS25_HOST:-127.0.0.1}"
API_PORT="${INDEXTTS25_PORT:-8092}"
GPU_DEVICE="${INDEXTTS25_GPU_DEVICE:-0}"
VLLM_VERSION="${INDEXTTS25_VLLM_VERSION:-0.27.0}"
SERVED_MODEL_NAME="${INDEXTTS25_SERVED_MODEL_NAME:-IndexTeam/IndexTTS-2.5}"
DEPLOY_CONFIG="${INDEXTTS25_DEPLOY_CONFIG:-${REPO_ROOT}/vllm_omni/deploy/indextts2_5.yaml}"
SETUP_RUNTIME="${INDEXTTS25_SETUP_RUNTIME:-1}"
INSTALL_WEBUI="${INDEXTTS25_INSTALL_WEBUI:-0}"

MODEL_ID="IndexTeam/IndexTTS-2.5"
WAV2VEC_ID="facebook/w2v-bert-2.0"
CAMPPLUS_ID="funasr/campplus"
BIGVGAN_ID="nvidia/bigvgan_v2_22khz_80band_256x"

log() { printf '[IndexTTS-2.5] %s\n' "$*"; }
fail() { printf '[IndexTTS-2.5] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "The vLLM-Omni launcher requires Linux."
command -v nvidia-smi >/dev/null || fail "nvidia-smi is not available. Install the NVIDIA driver first."

mkdir -p "${MODEL_DIR}" "${DATA_DIR}/logs" "${DATA_DIR}/speakers" \
  "${DATA_DIR}/cache/huggingface" "${DATA_DIR}/cache/speaker-conditioning" \
  "${DATA_DIR}/cache/torchinductor" "${DATA_DIR}/cache/triton" \
  "${DATA_DIR}/cache/cuda"

if [[ "${SETUP_RUNTIME}" == "1" ]]; then
  if ! command -v uv >/dev/null; then
    command -v curl >/dev/null || fail "curl is required to install uv."
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  command -v uv >/dev/null || fail "uv installation did not put uv on PATH."

  log "Creating or reusing the isolated Python 3.11 environment: ${VENV_DIR}"
  uv python install 3.11
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    uv venv --python 3.11 "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  log "Installing or verifying vLLM ${VLLM_VERSION}"
  uv pip install "vllm==${VLLM_VERSION}" --torch-backend=auto
  uv pip install -e "${REPO_ROOT}[indextts2]"
  if [[ "${INSTALL_WEBUI}" == "1" ]]; then
    uv pip install -e "${SCRIPT_DIR}[webui]"
  else
    uv pip install -e "${SCRIPT_DIR}"
  fi
  uv pip install 'huggingface_hub[cli]'

  log "Applying the Python 3.11 FlashInfer compatibility fix"
  python "${SCRIPT_DIR}/src/indextts25_compat/patch_flashinfer.py"
  python -c 'import flashinfer.comm; print("FlashInfer communication module import: OK")'
else
  [[ -x "${VENV_DIR}/bin/python" ]] || fail "Isolated environment is missing: ${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

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

if [[ "${SETUP_RUNTIME}" == "1" ]]; then
  log "Downloading or verifying the ${MODEL_ID} checkpoint bundle"
  hf_download "${MODEL_ID}" "${MODEL_DIR}"
  hf_download "${WAV2VEC_ID}" "${MODEL_DIR}/w2v-bert-2.0" \
    config.json model.safetensors preprocessor_config.json
  hf_download "${CAMPPLUS_ID}" "${MODEL_DIR}" campplus_cn_common.bin
  hf_download "${BIGVGAN_ID}" "${MODEL_DIR}/bigvgan" \
    config.json bigvgan_generator.pt
fi

for model_file in \
  config.yaml gpt.pth codec.pth s2mel.pth wav2vec2bert_stats.pt \
  multilingual_zh_ja_yue_char_del.tiktoken \
  qwen0.6bemo4-merge/config.json qwen0.6bemo4-merge/model.safetensors \
  w2v-bert-2.0/config.json w2v-bert-2.0/model.safetensors \
  w2v-bert-2.0/preprocessor_config.json campplus_cn_common.bin \
  bigvgan/config.json bigvgan/bigvgan_generator.pt; do
  [[ -s "${MODEL_DIR}/${model_file}" ]] \
    || fail "Required model asset is missing or empty: ${MODEL_DIR}/${model_file}"
done

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export FLASHINFER_DISABLE_VERSION_CHECK="1"
export HF_HOME="${DATA_DIR}/cache/huggingface"
export SPEAKER_SAMPLES_DIR="${DATA_DIR}/speakers"
export SPEAKER_CACHE_DIR="${DATA_DIR}/cache/speaker-conditioning"
export SPEAKER_CACHE_VERSION="indextts25-v1"
export TORCHINDUCTOR_CACHE_DIR="${DATA_DIR}/cache/torchinductor"
export TRITON_CACHE_DIR="${DATA_DIR}/cache/triton"
export CUDA_CACHE_PATH="${DATA_DIR}/cache/cuda"
export PYTHONUNBUFFERED="1"

log "Starting vLLM-Omni API on ${API_HOST}:${API_PORT}"
log "Model: ${MODEL_DIR} (served as ${SERVED_MODEL_NAME})"
log "Deploy config: ${DEPLOY_CONFIG}"
exec vllm serve "${MODEL_DIR}" \
  --omni \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --enable-sleep-mode \
  --log-stats \
  --deploy-config "${DEPLOY_CONFIG}"
