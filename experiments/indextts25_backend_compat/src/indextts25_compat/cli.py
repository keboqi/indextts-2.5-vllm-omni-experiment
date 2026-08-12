from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .backend import IndexTTS25Backend
from .client import OmniClient
from .models import SUPPORTED_LANGUAGES, SynthesisRequest


async def run(args: argparse.Namespace) -> None:
    client = OmniClient(args.server)
    backend = IndexTTS25Backend(client, max_parallel_segments=args.parallel)
    try:
        status = await backend.status()
        if not status.get("ready"):
            raise RuntimeError(f"vLLM-Omni is not ready: {status}")
        output = await backend.synthesize(
            SynthesisRequest(
                text=args.text,
                output_path=args.output,
                speaker_preset=args.voice,
                prompt_audio=args.ref_audio,
                language=args.language,
                target_duration_ms=args.target_ms,
                duration_control=args.duration_control,
                interval_silence_ms=args.silence_ms,
                diffusion_steps=args.diffusion_steps,
                emotion_text=args.emotion_text,
                seed=args.seed,
            )
        )
        print(output.resolve())
    finally:
        await backend.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="IndexTTS 2.5/vLLM-Omni experimental smoke test")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--text", required=True)
    speaker = parser.add_mutually_exclusive_group(required=True)
    speaker.add_argument("--ref-audio")
    speaker.add_argument("--voice")
    parser.add_argument("--output", type=Path, default=Path("indextts25-smoke.wav"))
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="zh")
    parser.add_argument("--target-ms", type=int, default=0)
    parser.add_argument("--duration-control", choices=("original", "native", "ffmpeg"), default="native")
    parser.add_argument("--silence-ms", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--emotion-text")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--parallel", type=int, default=100)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
