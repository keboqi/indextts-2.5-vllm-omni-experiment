# IndexTTS 2.5 vLLM-Omni compatibility experiment

This is an isolated experiment. It does not import or modify the existing
`index-tts-vllm` application.

For one-command native deployment and browser-based acceptance testing, see
[`DEPLOY_WEBUI.md`](DEPLOY_WEBUI.md).

## What the Omni route covers

The fetched vLLM-Omni IndexTTS 2.5 implementation is a complete two-stage
pipeline:

1. Stage 0: multilingual frontend, speaker/emotion conditioning and
   autoregressive semantic-code generation.
2. Stage 1: EnhancedCodec semantic decoding, length regulation, S2Mel
   flow-matching/DiT and BigVGAN vocoding to 22,050 Hz audio.

This moves the expensive codec/CFM/vocoder tail into the scheduled Omni
pipeline and allows Stage 1 batching. It does **not** make IndexTTS 2.5 a native
streaming model. S2Mel needs the complete semantic-code sequence, and the PR's
deployment file therefore uses `async_chunk: false` and
`codec_streaming: false`.

The compatibility `stream()` method splits text and submits complete requests,
yielding one WAV per sentence. Its first-audio latency and cross-sentence
prosody must be benchmarked separately from the current 2.0 implementation.

## Compatibility coverage

| Current 2.0 function | Experiment |
|---|---|
| Text-to-WAV and output path | Implemented |
| Prompt-audio voice cloning | Implemented with data URLs |
| Named speaker presets | Implemented with Omni voice CRUD |
| Sentence splitting / max input tokens | Implemented conservatively |
| Parallel segment requests | Implemented with a configurable semaphore (default 100) |
| Natural document benchmark | 32 parallel Chinese chunks, ordered assembly, final-file aggregate RTF |
| Inter-sentence silence | Implemented for final WAV |
| Sentence-level streaming | Implemented; explicitly not native |
| Language selection | Chinese (`zh`), English (`en`), Japanese (`ja`), Spanish (`es`), and Arabic (`ar`); defaults to Chinese |
| Emotion audio/text/vector/random + weight | Implemented |
| Seed and sampling overrides | Forwarded |
| Native precise target duration | Experimental Stage-1 patch |
| Final exact sample duration | Crop/zero-pad after native generation |
| Per-request diffusion steps | Implemented; mixed-step requests form separate CFM batches |
| Status / shutdown | Implemented for the remote client |
| GPU sleep/wake | Not exposed by the Omni OpenAI server |
| Prompt-conditioning cache | Omni prefix cache and uploaded voices; semantics differ |

## Exact-duration design

The experimental Omni patch carries `extra_params.target_duration_ms` from the
OpenAI adapter through Stage 0 into Stage 1. The S2Mel length regulator uses:

```
mel_frames = round(target_ms * 22050 / (1000 * 256))
```

That gives 11.61 ms model-native resolution. The compatibility layer then
crops or zero-pads to the exact requested sample count; it does not use FFmpeg
time stretching, so it does not change pitch or speaking rate after synthesis.

## Running

On the Linux GPU server, run the native bootstrap from the repository root:

```bash
chmod +x experiments/indextts25_backend_compat/deploy_and_webui.sh
./experiments/indextts25_backend_compat/deploy_and_webui.sh
```

No deployment environment variables are required. Everything is stored under
the copied project directory.

Open `http://SERVER_IP:7860`. See [`DEPLOY_WEBUI.md`](DEPLOY_WEBUI.md) for
persistent-volume, authentication, configuration, and test instructions.

GPU end-to-end testing requires the IndexTTS 2.5 checkpoint and a CUDA setup
compatible with this vLLM-Omni PR. Unit tests do not load model weights.
