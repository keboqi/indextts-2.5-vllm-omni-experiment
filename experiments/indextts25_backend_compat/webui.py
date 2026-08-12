from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import subprocess
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import gradio as gr

from indextts25_compat import IndexTTS25Backend, OmniClient, SynthesisRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS 2.5 experimental test UI")
    parser.add_argument("--api-base", default="http://127.0.0.1:8092")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", default="IndexTeam/IndexTTS-2.5")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


ARGS = parse_args()
ARGS.results_dir.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["zhen", "zh", "en", "ja", "yue", "es", "ar", "de", "fr", "ko"]
TEST_CHOICES = [
    "Multilingual",
    "Exact duration",
    "Diffusion steps",
    "Emotion",
    "Concurrency",
    "Compatibility streaming",
    "Stability",
    "Named voice",
]


def unique_run_dir(prefix: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = ARGS.results_dir / f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True)
    return path


def parse_numbers(value: str, kind: type[int] | type[float]) -> list[Any]:
    values = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    return [kind(item) for item in values]


def parse_emotion_vector(value: str) -> list[float] | None:
    if not value.strip():
        return None
    vector = parse_numbers(value, float)
    if len(vector) != 8:
        raise gr.Error("Emotion vector must contain exactly eight comma-separated values.")
    return vector


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as source:
        frames = source.getnframes()
        rate = source.getframerate()
        return {
            "sample_rate": rate,
            "channels": source.getnchannels(),
            "sample_width": source.getsampwidth(),
            "frames": frames,
            "duration_ms": frames * 1000.0 / rate,
        }


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        parts = [part.strip() for part in result.stdout.strip().split(",")]
        return {
            "name": parts[0],
            "memory_used_mb": float(parts[1]),
            "memory_total_mb": float(parts[2]),
            "utilization_percent": float(parts[3]),
            "temperature_c": float(parts[4]),
            "power_w": float(parts[5]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def make_request(
    *,
    text: str,
    output: Path,
    reference: str | None,
    voice: str | None,
    language: str,
    target_ms: int = 0,
    duration_control: str = "native",
    silence_ms: int = 0,
    max_text_tokens: int = 120,
    diffusion_steps: int = 25,
    emotion_audio: str | None = None,
    emotion_text: str | None = None,
    emotion_weight: float = 0.6,
    emotion_vector: list[float] | None = None,
    random_emotion: bool = False,
    seed: int | None = 42,
    sampling: dict[str, Any] | None = None,
) -> SynthesisRequest:
    return SynthesisRequest(
        text=text,
        output_path=output,
        prompt_audio=reference,
        speaker_preset=voice,
        language=language,
        target_duration_ms=target_ms,
        duration_control=duration_control,
        interval_silence_ms=silence_ms,
        max_text_tokens=max_text_tokens,
        diffusion_steps=diffusion_steps,
        emotion_audio=emotion_audio,
        emotion_text=emotion_text,
        emotion_weight=emotion_weight,
        emotion_vector=emotion_vector,
        random_emotion=random_emotion,
        seed=seed,
        sampling=sampling or {},
    )


async def with_backend(max_parallel: int = 100) -> tuple[OmniClient, IndexTTS25Backend]:
    client = OmniClient(ARGS.api_base)
    return client, IndexTTS25Backend(client, model=ARGS.model, max_parallel_segments=max_parallel)


async def synthesize_ui(
    text: str,
    reference: str | None,
    voice: str,
    language: str,
    target_ms: int,
    duration_control: str,
    silence_ms: int,
    max_text_tokens: int,
    diffusion_steps: int,
    emotion_audio: str | None,
    emotion_text: str,
    emotion_weight: float,
    emotion_vector: str,
    random_emotion: bool,
    seed: int,
    sampling_json: str,
) -> tuple[str, str]:
    if not reference and not voice.strip():
        raise gr.Error("Upload reference audio or enter an uploaded voice name.")
    try:
        sampling = json.loads(sampling_json or "{}")
        if not isinstance(sampling, dict):
            raise ValueError("sampling JSON must be an object")
    except Exception as exc:
        raise gr.Error(f"Invalid sampling JSON: {exc}") from exc
    run_dir = unique_run_dir("single")
    output = run_dir / "output.wav"
    client, backend = await with_backend()
    started = time.perf_counter()
    before = gpu_snapshot()
    try:
        request = make_request(
            text=text,
            output=output,
            reference=reference,
            voice=voice.strip() or None,
            language=language,
            target_ms=int(target_ms or 0),
            duration_control=duration_control,
            silence_ms=int(silence_ms or 0),
            max_text_tokens=int(max_text_tokens),
            diffusion_steps=int(diffusion_steps),
            emotion_audio=emotion_audio,
            emotion_text=emotion_text.strip() or None,
            emotion_weight=float(emotion_weight),
            emotion_vector=parse_emotion_vector(emotion_vector),
            random_emotion=random_emotion,
            seed=int(seed) if seed is not None else None,
            sampling=sampling,
        )
        result = await backend.synthesize(request)
        elapsed = time.perf_counter() - started
        info = wav_info(result)
        report = {
            "status": "pass",
            "output": str(result),
            "elapsed_s": round(elapsed, 3),
            "real_time_factor": round(elapsed / (info["duration_ms"] / 1000.0), 3),
            "audio": info,
            "target_error_ms": round(info["duration_ms"] - target_ms, 3) if target_ms else None,
            "gpu_before": before,
            "gpu_after": gpu_snapshot(),
        }
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return str(result), json.dumps(report, indent=2)
    finally:
        await backend.shutdown()


async def server_status() -> str:
    client = OmniClient(ARGS.api_base)
    try:
        status = await client.status()
        status["gpu"] = gpu_snapshot()
        status["model_argument"] = ARGS.model
        return json.dumps(status, indent=2)
    finally:
        await client.close()


class SuiteRecorder:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.rows: list[dict[str, Any]] = []

    async def synth(
        self,
        backend: IndexTTS25Backend,
        label: str,
        request: SynthesisRequest,
        **extra: Any,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        row: dict[str, Any] = {"label": label, **extra}
        try:
            result = await backend.synthesize(request)
            elapsed = time.perf_counter() - started
            info = wav_info(result)
            row.update(
                status="pass",
                output=str(result),
                elapsed_s=round(elapsed, 3),
                duration_ms=round(info["duration_ms"], 3),
                rtf=round(elapsed / (info["duration_ms"] / 1000.0), 3),
                sample_rate=info["sample_rate"],
                channels=info["channels"],
            )
        except Exception as exc:
            row.update(status="fail", error=f"{type(exc).__name__}: {exc}")
        self.rows.append(row)
        return row


async def run_suite(
    reference: str | None,
    tests: list[str],
    duration_targets: str,
    diffusion_values: str,
    concurrency: int,
    stability_repeats: int,
    seed: int,
) -> tuple[str, str | None]:
    if not reference:
        raise gr.Error("A reference-audio upload is required for the automated suite.")
    if not tests:
        raise gr.Error("Select at least one test group.")
    targets = parse_numbers(duration_targets, int)
    steps_values = parse_numbers(diffusion_values, int)
    run_dir = unique_run_dir("suite")
    recorder = SuiteRecorder(run_dir)
    client, backend = await with_backend(max_parallel=max(1, int(concurrency)))
    gpu_before = gpu_snapshot()
    suite_started = time.perf_counter()

    language_cases = {
        "en": "Hello, this is an English synthesis validation sentence.",
        "zh": "你好，这是中文语音合成验证。",
        "zhen": "Hello，欢迎进行 IndexTTS 二点五混合语言测试。",
        "ja": "こんにちは、これは日本語音声合成のテストです。",
        "yue": "你好，呢段係粵語語音合成測試。",
    }
    try:
        if "Multilingual" in tests:
            for language, text in language_cases.items():
                await recorder.synth(
                    backend,
                    f"language-{language}",
                    make_request(
                        text=text,
                        output=run_dir / f"language-{language}.wav",
                        reference=reference,
                        voice=None,
                        language=language,
                        diffusion_steps=25,
                        seed=seed,
                    ),
                    group="multilingual",
                    language=language,
                )

        if "Exact duration" in tests:
            text = "This sentence validates exact duration while preserving complete, natural speech."
            baseline = await recorder.synth(
                backend,
                "duration-baseline",
                make_request(
                    text=text,
                    output=run_dir / "duration-baseline.wav",
                    reference=reference,
                    voice=None,
                    language="en",
                    diffusion_steps=25,
                    seed=seed,
                ),
                group="duration",
                target_ms=0,
            )
            for target in targets:
                row = await recorder.synth(
                    backend,
                    f"duration-{target}",
                    make_request(
                        text=text,
                        output=run_dir / f"duration-{target}.wav",
                        reference=reference,
                        voice=None,
                        language="en",
                        target_ms=target,
                        diffusion_steps=25,
                        seed=seed,
                    ),
                    group="duration",
                    target_ms=target,
                    baseline_ms=baseline.get("duration_ms"),
                )
                if row["status"] == "pass":
                    row["target_error_ms"] = round(row["duration_ms"] - target, 3)

        if "Diffusion steps" in tests:
            for steps in steps_values:
                await recorder.synth(
                    backend,
                    f"diffusion-{steps}",
                    make_request(
                        text="A fixed sentence and seed reveal the quality and latency effect of diffusion steps.",
                        output=run_dir / f"diffusion-{steps}.wav",
                        reference=reference,
                        voice=None,
                        language="en",
                        diffusion_steps=steps,
                        seed=seed,
                    ),
                    group="diffusion",
                    diffusion_steps=steps,
                )

        if "Emotion" in tests:
            for label, kwargs in (
                ("neutral", {}),
                ("emotion-text", {"emotion_text": "happy, energetic, excited, but natural"}),
                ("emotion-vector", {"emotion_vector": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}),
            ):
                await recorder.synth(
                    backend,
                    label,
                    make_request(
                        text="We finally completed the project successfully!",
                        output=run_dir / f"{label}.wav",
                        reference=reference,
                        voice=None,
                        language="en",
                        diffusion_steps=25,
                        emotion_weight=0.8,
                        seed=seed,
                        **kwargs,
                    ),
                    group="emotion",
                )

        if "Concurrency" in tests:
            batch_started = time.perf_counter()
            tasks = [
                recorder.synth(
                    backend,
                    f"concurrency-{index + 1}",
                    make_request(
                        text=f"Concurrent request number {index + 1} checks scheduling and request isolation.",
                        output=run_dir / f"concurrency-{index + 1}.wav",
                        reference=reference,
                        voice=None,
                        language="en",
                        diffusion_steps=10 + (index % 2) * 15,
                        target_ms=2500 + (index % 2) * 500,
                        seed=seed + index,
                    ),
                    group="concurrency",
                    requested_concurrency=concurrency,
                )
                for index in range(int(concurrency))
            ]
            await asyncio.gather(*tasks)
            recorder.rows.append(
                {
                    "label": "concurrency-summary",
                    "group": "concurrency",
                    "status": "pass",
                    "request_count": int(concurrency),
                    "wall_s": round(time.perf_counter() - batch_started, 3),
                }
            )

        if "Compatibility streaming" in tests:
            stream_started = time.perf_counter()
            arrivals = []
            stream_request = make_request(
                text=(
                    "The first sentence should arrive first. "
                    "The second sentence checks ordered delivery. "
                    "The third sentence completes the compatibility stream."
                ),
                output=run_dir / "stream-unused.wav",
                reference=reference,
                voice=None,
                language="en",
                diffusion_steps=15,
                seed=seed,
            )
            try:
                index = 0
                async for chunk in backend.stream(stream_request):
                    index += 1
                    output = run_dir / f"stream-chunk-{index}.wav"
                    output.write_bytes(chunk)
                    info = wav_info(output)
                    arrivals.append(round(time.perf_counter() - stream_started, 3))
                    recorder.rows.append(
                        {
                            "label": f"stream-chunk-{index}",
                            "group": "compatibility_streaming",
                            "status": "pass",
                            "output": str(output),
                            "arrival_s": arrivals[-1],
                            "duration_ms": round(info["duration_ms"], 3),
                            "sample_rate": info["sample_rate"],
                            "channels": info["channels"],
                        }
                    )
                recorder.rows.append(
                    {
                        "label": "stream-summary",
                        "group": "compatibility_streaming",
                        "status": "pass" if index == 3 else "fail",
                        "native_streaming": False,
                        "expected_chunks": 3,
                        "actual_chunks": index,
                        "arrival_s": arrivals,
                        "wall_s": round(time.perf_counter() - stream_started, 3),
                    }
                )
            except Exception as exc:
                recorder.rows.append(
                    {
                        "label": "stream-summary",
                        "group": "compatibility_streaming",
                        "status": "fail",
                        "native_streaming": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if "Stability" in tests:
            memory_samples = []
            for index in range(int(stability_repeats)):
                await recorder.synth(
                    backend,
                    f"stability-{index + 1}",
                    make_request(
                        text="Repeated inference checks memory stability and stale request state.",
                        output=run_dir / f"stability-{index + 1}.wav",
                        reference=reference,
                        voice=None,
                        language="en",
                        diffusion_steps=10,
                        seed=seed + index,
                    ),
                    group="stability",
                    iteration=index + 1,
                )
                memory_samples.append(gpu_snapshot().get("memory_used_mb"))
            finite = [float(value) for value in memory_samples if isinstance(value, (int, float)) and math.isfinite(value)]
            recorder.rows.append(
                {
                    "label": "stability-memory-summary",
                    "group": "stability",
                    "status": "pass" if finite else "warning",
                    "memory_used_mb": finite,
                    "memory_growth_mb": round(finite[-1] - finite[0], 1) if len(finite) > 1 else None,
                }
            )

        if "Named voice" in tests:
            voice_name = f"webui_test_{uuid.uuid4().hex[:8]}"
            try:
                upload = await client.add_voice(
                    voice_name,
                    reference,
                    consent="interactive-webui-test",
                    speaker_description="Temporary automated-suite voice",
                )
                row = await recorder.synth(
                    backend,
                    "named-voice-synthesis",
                    make_request(
                        text="This sentence uses a temporarily uploaded named voice.",
                        output=run_dir / "named-voice.wav",
                        reference=None,
                        voice=voice_name,
                        language="en",
                        seed=seed,
                    ),
                    group="named_voice",
                    voice_name=voice_name,
                )
                row["upload_response"] = upload
                row["voice_list_after_upload"] = await client.list_voices()
            finally:
                try:
                    delete_response = await client.delete_voice(voice_name)
                    recorder.rows.append(
                        {
                            "label": "named-voice-delete",
                            "group": "named_voice",
                            "status": "pass",
                            "response": delete_response,
                        }
                    )
                except Exception as exc:
                    recorder.rows.append(
                        {
                            "label": "named-voice-delete",
                            "group": "named_voice",
                            "status": "fail",
                            "error": str(exc),
                        }
                    )
    finally:
        await backend.shutdown()

    report = {
        "api_base": ARGS.api_base,
        "model": ARGS.model,
        "selected_tests": tests,
        "elapsed_s": round(time.perf_counter() - suite_started, 3),
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "rows": recorder.rows,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    archive_path = Path(shutil.make_archive(str(run_dir), "zip", root_dir=run_dir))
    failures = sum(row.get("status") == "fail" for row in recorder.rows)
    passes = sum(row.get("status") == "pass" for row in recorder.rows)
    markdown = (
        f"## Suite complete\n\n"
        f"- Passed records: **{passes}**\n"
        f"- Failed records: **{failures}**\n"
        f"- Wall time: **{report['elapsed_s']} s**\n"
        f"- Results directory: `{run_dir}`\n\n"
        "Exact file duration alone does not prove acceptable duration control. "
        "Download and listen to the generated clips for word omissions, unnatural pacing, "
        "speaker drift, emotion drift, and sentence-boundary discontinuity."
    )
    return markdown + "\n\n```json\n" + json.dumps(recorder.rows, indent=2) + "\n```", str(archive_path)


async def voice_list() -> str:
    client = OmniClient(ARGS.api_base)
    try:
        return json.dumps(await client.list_voices(), indent=2)
    finally:
        await client.close()


async def voice_add(name: str, audio: str | None, consent: str, ref_text: str, description: str) -> str:
    if not name.strip() or not audio:
        raise gr.Error("Voice name and audio are required.")
    client = OmniClient(ARGS.api_base)
    try:
        result = await client.add_voice(
            name.strip(),
            audio,
            consent=consent.strip() or "interactive-webui-consent",
            ref_text=ref_text.strip() or None,
            speaker_description=description.strip() or None,
        )
        return json.dumps(result, indent=2)
    finally:
        await client.close()


async def voice_delete(name: str) -> str:
    if not name.strip():
        raise gr.Error("Voice name is required.")
    client = OmniClient(ARGS.api_base)
    try:
        return json.dumps(await client.delete_voice(name.strip()), indent=2)
    finally:
        await client.close()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="IndexTTS 2.5 vLLM-Omni Experiment") as demo:
        gr.Markdown(
            "# IndexTTS 2.5 · vLLM-Omni experiment\n"
            "This UI tests the isolated 2.5 backend. The pipeline is end-to-end but **not native streaming**; "
            "compatibility streaming uses complete sentence requests."
        )

        with gr.Tab("Single synthesis"):
            with gr.Row():
                with gr.Column(scale=2):
                    text = gr.Textbox(
                        label="Text",
                        lines=6,
                        value="Hello，欢迎测试 IndexTTS 二点五。",
                    )
                    reference = gr.Audio(label="Speaker reference WAV", type="filepath", sources=["upload"])
                    voice = gr.Textbox(label="Uploaded voice name (alternative to reference audio)")
                    language = gr.Dropdown(LANGUAGES, value="zhen", label="Language")
                with gr.Column():
                    target_ms = gr.Number(value=0, precision=0, label="Target duration (ms; 0 disables)")
                    duration_control = gr.Radio(
                        ["native", "ffmpeg", "original"], value="native", label="Duration control"
                    )
                    silence_ms = gr.Number(value=0, precision=0, label="Inter-sentence silence (ms)")
                    max_text_tokens = gr.Slider(20, 500, value=120, step=1, label="Max input tokens per segment")
                    diffusion_steps = gr.Slider(1, 100, value=25, step=1, label="Diffusion steps")
                    seed = gr.Number(value=42, precision=0, label="Seed")
            with gr.Accordion("Emotion and sampling", open=False):
                emotion_audio = gr.Audio(label="Emotion reference audio", type="filepath", sources=["upload"])
                emotion_text = gr.Textbox(label="Emotion description")
                emotion_weight = gr.Slider(0, 1, value=0.6, label="Emotion weight")
                emotion_vector = gr.Textbox(
                    label="Emotion vector (8 values: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm)"
                )
                random_emotion = gr.Checkbox(label="Random emotion", value=False)
                sampling_json = gr.Code(
                    label="Sampling overrides JSON",
                    language="json",
                    value='{"temperature": 0.8, "top_p": 0.8, "top_k": 30}',
                )
            synth_button = gr.Button("Synthesize", variant="primary")
            with gr.Row():
                output_audio = gr.Audio(label="Generated audio", type="filepath")
                output_report = gr.Code(label="Measurement", language="json")
            synth_button.click(
                synthesize_ui,
                inputs=[
                    text,
                    reference,
                    voice,
                    language,
                    target_ms,
                    duration_control,
                    silence_ms,
                    max_text_tokens,
                    diffusion_steps,
                    emotion_audio,
                    emotion_text,
                    emotion_weight,
                    emotion_vector,
                    random_emotion,
                    seed,
                    sampling_json,
                ],
                outputs=[output_audio, output_report],
            )

        with gr.Tab("Automated acceptance suite"):
            gr.Markdown(
                "Runs selected feature groups and packages every WAV plus a JSON report. "
                "The suite catches transport/runtime failures; human listening remains required for quality acceptance."
            )
            suite_reference = gr.Audio(label="Consented speaker reference WAV", type="filepath", sources=["upload"])
            selected_tests = gr.CheckboxGroup(TEST_CHOICES, value=TEST_CHOICES, label="Test groups")
            with gr.Row():
                duration_targets = gr.Textbox(value="2000,4000,6000", label="Duration targets (ms)")
                diffusion_values = gr.Textbox(value="10,15,25,40", label="Diffusion step values")
                concurrency = gr.Slider(1, 100, value=100, step=1, label="Concurrent requests")
                stability_repeats = gr.Slider(1, 100, value=10, step=1, label="Stability repetitions")
                suite_seed = gr.Number(value=42, precision=0, label="Base seed")
            suite_button = gr.Button("Run selected suite", variant="primary")
            suite_report = gr.Markdown()
            suite_archive = gr.File(label="Download complete result archive")
            suite_button.click(
                run_suite,
                inputs=[
                    suite_reference,
                    selected_tests,
                    duration_targets,
                    diffusion_values,
                    concurrency,
                    stability_repeats,
                    suite_seed,
                ],
                outputs=[suite_report, suite_archive],
            )

        with gr.Tab("Named voices"):
            voice_name = gr.Textbox(label="Voice name")
            voice_audio = gr.Audio(label="Voice sample", type="filepath", sources=["upload"])
            consent = gr.Textbox(value="interactive-webui-consent", label="Consent record ID")
            voice_ref_text = gr.Textbox(label="Reference transcript (optional)")
            voice_description = gr.Textbox(label="Description (optional)")
            with gr.Row():
                add_button = gr.Button("Upload voice")
                list_button = gr.Button("Refresh list")
                delete_button = gr.Button("Delete voice", variant="stop")
            voice_result = gr.Code(label="Voice API result", language="json")
            add_button.click(
                voice_add,
                [voice_name, voice_audio, consent, voice_ref_text, voice_description],
                voice_result,
            )
            list_button.click(voice_list, outputs=voice_result)
            delete_button.click(voice_delete, voice_name, voice_result)

        with gr.Tab("Server status"):
            refresh_status = gr.Button("Refresh")
            status_output = gr.Code(label="Status", language="json")
            refresh_status.click(server_status, outputs=status_output)

    return demo


if __name__ == "__main__":
    build_ui().queue(default_concurrency_limit=100).launch(
        server_name=ARGS.host,
        server_port=ARGS.port,
        show_error=True,
        allowed_paths=[str(ARGS.results_dir.resolve())],
    )
