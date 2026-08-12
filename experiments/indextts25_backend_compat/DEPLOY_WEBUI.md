# Native deployment and test UI

This deployment uses no Docker. Copy the complete experimental repository to a
Linux GPU server, run one Bash script, and perform validation in the web UI.
The existing IndexTTS 2.0 application is not installed or modified.

## Server requirements

- Ubuntu 22.04/24.04 x86-64.
- One NVIDIA RTX PRO 6000 Blackwell with a current production driver.
- `nvidia-smi`, `curl`, and normal outbound HTTPS access.
- Enough disk space for the Python environment, build caches, model, and test
  results. Allow at least 80 GB free during setup.
- Port 7860 reachable from the tester, or an SSH tunnel to it.

The script installs `uv` when missing, creates a private Python 3.11
environment, installs vLLM 0.27 and this exact vLLM-Omni source tree, downloads
IndexTTS 2.5, launches the local Omni API, waits for model readiness, and then
starts Gradio. Stage 0 and Stage 1 both use logical GPU 0.

The bootstrap also applies an idempotent compatibility fix to FlashInfer
0.6.16. Its `flashinfer.comm.fd_exchange` module ships an evaluated
`array.array[int]` annotation that fails on Python 3.11 before the engine can
start. The annotation-only patch does not change kernels or communication
behavior, and the script immediately verifies `import flashinfer.comm`.

## Copy and run

The repository contains uncommitted experimental changes, so copy the complete
working tree rather than cloning plain vLLM-Omni `main`:

```bash
rsync -a \
  --exclude .git \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  /local/index-tts-2.5-vllm-omni-experiment/ \
  USER@SERVER:/srv/index-tts-2.5-vllm-omni-experiment/
```

On the server:

```bash
cd /srv/index-tts-2.5-vllm-omni-experiment
chmod +x experiments/indextts25_backend_compat/deploy_and_webui.sh
./experiments/indextts25_backend_compat/deploy_and_webui.sh
```

Open:

```text
http://SERVER_IP:7860
```

The model API binds to `127.0.0.1:8092` by default and is not exposed. The web
UI binds to `0.0.0.0:7860`. Restrict port 7860 to a private network or use an
SSH tunnel; the test UI has no authentication:

```bash
ssh -L 7860:127.0.0.1:7860 USER@SERVER
```

Then open `http://127.0.0.1:7860`.

Press Ctrl+C once to stop the UI and its vLLM-Omni child process cleanly.

## Hugging Face authentication

The model is downloaded without additional settings. If Hugging Face returns an
access error, authenticate once inside the project environment and rerun:

```bash
source .venv-indextts25/bin/activate
hf auth login
./experiments/indextts25_backend_compat/deploy_and_webui.sh
```

The script downloads `IndexTeam/IndexTTS-2.5`. It does not mix any IndexTTS
2.0 checkpoint into the model directory. It also downloads the external models
that the official IndexTTS runtime normally resolves on first use:

- `facebook/w2v-bert-2.0` for reference-audio features (only the Transformers
  config, feature-extractor config, and safetensors weights).
- `funasr/campplus` for speaker embeddings (only `campplus_cn_common.bin`).
- `nvidia/bigvgan_v2_22khz_80band_256x` for waveform generation (only the
  config and generator checkpoint; optimizer and alternate checkpoints are
  excluded).

These assets are stored under `models/IndexTTS-2.5/` in the directory layouts
expected by vLLM-Omni. The script checks every required file before starting
the API, so missing auxiliary models fail early with a specific path.

## Project-local storage and Modal

Defaults are kept under the copied repository:

```text
.venv-indextts25/              Python environment
models/IndexTTS-2.5/           checkpoint plus external runtime models
runtime/indextts25/cache/      Hugging Face/runtime cache
runtime/indextts25/speakers/   uploaded named voices
runtime/indextts25/results/    WAVs, JSON reports, ZIP archives
runtime/indextts25/logs/       API and UI logs
```

No path or port environment variables are required. For Modal, mount or copy
the entire project directory onto persistent storage. The environment, model,
cache, named voices, reports, and logs then persist together. The script always
uses GPU 0, internal API port 8092, and web port 7860.

The first run installs and downloads everything (roughly 8.3 GB of model data
in total). Subsequent runs reuse the project-local virtual environment and
Hugging Face performs an incremental model verification/download, so already
complete files are retained.

## Web UI coverage

### Single synthesis

Exposes:

- Prompt audio or uploaded named voice.
- All supported languages.
- Native, FFmpeg, and unconstrained duration modes.
- Exact target milliseconds and inter-sentence silence.
- Text segmentation limit.
- Request-local diffusion steps.
- Emotion audio, text, vector, random mode, and strength.
- Seed plus raw sampling overrides.

The result panel reports sample format, elapsed time, generated duration,
real-time factor, exact-duration error, and GPU snapshots.

### Automated acceptance suite

Selectable groups include:

- English, Mandarin, mixed Chinese/English, Japanese, and Cantonese.
- Exact-duration sweep with an unconstrained baseline.
- Diffusion-step quality/latency sweep.
- Neutral, emotion-text, and emotion-vector comparisons.
- Concurrent capacity sweep with separate fixed and mixed workloads.
- Ordered sentence-level compatibility streaming with chunk-arrival timing.
- Repeated-inference VRAM stability sampling.
- Temporary named-voice upload, synthesis, listing, and deletion.

The suite produces a ZIP containing every WAV and a machine-readable
`report.json`. A green runtime report is not by itself a quality pass: listen
for word omissions, unnatural pacing, speaker drift, emotion drift, clipping,
silence, and discontinuities between independently generated sentences.

The default high-throughput profile admits up to 100 requests while bounding
the physical GPU work at 32 Stage 0 sequences, 8 Stage 1 sequences, and an
8-request CFM microbatch. Admission capacity is intentionally larger than the
memory-intensive decoder batch: excess work remains queued in vLLM instead of
turning a burst of 100 requests into an effective 200-item CFG DiT batch.

The concurrency benchmark defaults to levels 4, 8, 16, 32, 64, and 100 with
100 measured requests at every level. Before measurement, each level runs a
warm-up wave up to the physical Stage 0 capacity so new Torch batch shapes and
compilation costs are reported separately. Its summary includes:

- Successful/failed requests and success rate.
- Requests per second and generated audio seconds per wall second.
- Aggregate RTF plus mean, p50, p95, p99, and maximum request latency.
- Continuously sampled peak VRAM, average/peak GPU utilization, power, and
  peak temperature.
- Per-request timing and every raw GPU sample in a level-specific
  `benchmark.json`.

The `fixed` workload holds duration and diffusion steps constant to maximize
batch compatibility. The `mixed` workload rotates durations from 2.0 to 3.5
seconds and diffusion steps 10, 15, 25, and 40 to expose real-world batch
fragmentation. Run both before selecting production limits.

### Named voices

Provides upload/list/delete controls with explicit consent metadata. Named
voices persist in `runtime/indextts25/speakers` and are restored on restart.

## Recommended test order

1. Open **Server status** and verify the model and RTX PRO 6000 are visible.
2. Generate one English and one Mandarin clip in **Single synthesis**.
3. Run the default fixed-workload concurrency sweep and ten stability
   repetitions. For a quicker functional check, use levels `4,8` and 16
   measured requests per level.
4. Download and listen to the complete ZIP.
5. Repeat duration tests around the natural baseline rather than accepting
   only the exact file length.
6. Compare diffusion steps 10, 15, 25, and 40 for latency and audible quality.
7. Increase stability repetitions to 100 only after the short suite is clean.
8. Compare the accepted outputs blindly against the current IndexTTS 2.0
   backend before integration.

After the fixed sweep, repeat it with the mixed workload. Select the production
limit from the point where throughput stops improving, p95/p99 latency rises
sharply, or peak VRAM loses a safe margin. The configured 100 is an in-flight
admission limit, not a physical decoder batch of 100.

## Logs and troubleshooting

```bash
tail -f runtime/indextts25/logs/vllm-api.log
tail -f runtime/indextts25/logs/webui.log
nvidia-smi
```

The first startup can remain quiet for several minutes while Stage 1 performs
`torch.compile`; the script waits up to 30 minutes. If startup exits, it prints
the last 200 API log lines.

If an earlier clone failed with `TypeError: type 'array.array' is not
subscriptable`, pull the latest repository update and rerun the same deployment
script. It repairs the existing project-local environment before starting the
server.

For CUDA OOM, first confirm no unrelated GPU processes exist. Do not change
precision, diffusion steps, or the two `gpu_memory_utilization` values until
the original failure and peak VRAM are recorded.
