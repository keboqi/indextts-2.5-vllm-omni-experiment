from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from .audio import ffmpeg_fit_duration, fit_wav_duration, join_wav, write_audio
from .client import OmniClient, audio_reference
from .models import BackendCapabilities, SynthesisRequest
from .text import allocate_durations, split_text


class IndexTTS25Backend:
    name = "index25_omni_experimental"
    capabilities = BackendCapabilities(
        native_streaming=False,
        native_duration=True,
        emotion_text=True,
    )

    def __init__(
        self,
        client: OmniClient,
        model: str = "IndexTeam/IndexTTS-2.5",
        max_parallel_segments: int = 100,
    ) -> None:
        if max_parallel_segments < 1:
            raise ValueError("max_parallel_segments must be positive")
        self.client = client
        self.model = model
        self._segment_slots = asyncio.Semaphore(max_parallel_segments)

    def build_payload(self, request: SynthesisRequest, *, text: str, target_ms: int = 0) -> dict[str, Any]:
        extras: dict[str, Any] = dict(request.sampling)
        extras.update(
            {
                "lang": request.language.strip().lower() if request.language else "zh",
                "text_normalization": True,
                "diffusion_steps": request.diffusion_steps,
                "cache_prompt_audio": request.cache_prompt_audio,
            }
        )
        if target_ms and request.duration_control != "ffmpeg":
            extras["target_duration_ms"] = target_ms
        if request.emotion_audio:
            extras["emo_audio"] = audio_reference(
                request.emotion_audio,
                use_cache=request.cache_prompt_audio,
            )
        if request.emotion_vector is not None:
            extras["emo_vector"] = list(request.emotion_vector)
        if request.emotion_text:
            extras.update({"emo_text": request.emotion_text, "use_emo_text": True})
        if request.random_emotion:
            extras["use_random"] = True
        if request.emotion_audio or request.emotion_text or request.emotion_vector is not None:
            extras["emo_alpha"] = request.emotion_weight
        payload: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "response_format": "wav",
            "stream": False,
            "extra_params": extras,
        }
        max_new_tokens = request.sampling.get("max_new_tokens")
        if max_new_tokens is not None:
            payload["max_new_tokens"] = int(max_new_tokens)
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.prompt_audio:
            payload["ref_audio"] = audio_reference(
                request.prompt_audio,
                use_cache=request.cache_prompt_audio,
            )
        else:
            payload["voice"] = request.speaker_preset
        if request.reference_text:
            payload["ref_text"] = request.reference_text
        return payload

    async def _segments(self, request: SynthesisRequest) -> tuple[list[str], list[int]]:
        request.validate()
        texts = split_text(request.text, request.max_text_tokens)
        durations = allocate_durations(texts, request.target_duration_ms, request.interval_silence_ms)
        return texts, durations

    async def _synthesize_segment(
        self,
        request: SynthesisRequest,
        text: str,
        duration: int,
    ) -> bytes:
        async with self._segment_slots:
            return await self.client.synthesize(self.build_payload(request, text=text, target_ms=duration))

    async def synthesize(self, request: SynthesisRequest) -> Path:
        texts, durations = await self._segments(request)
        chunks = await asyncio.gather(
            *(
                self._synthesize_segment(
                    replace(request, seed=None if request.seed is None else request.seed + index),
                    text,
                    duration,
                )
                for index, (text, duration) in enumerate(zip(texts, durations))
            )
        )
        audio = join_wav(chunks, request.interval_silence_ms)
        if request.target_duration_ms:
            if request.duration_control == "ffmpeg":
                audio = ffmpeg_fit_duration(audio, request.target_duration_ms)
            else:
                # Native control targets mel frames. Final sample fitting
                # removes hop quantization without changing tempo.
                audio = fit_wav_duration(audio, request.target_duration_ms)
        return write_audio(request.output_path, audio)

    async def stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        """Yield sentence WAVs; IndexTTS 2.5 itself remains non-streaming."""
        texts, durations = await self._segments(request)
        tasks = []
        for index, (text, duration) in enumerate(zip(texts, durations)):
            segment_request = replace(request, seed=None if request.seed is None else request.seed + index)
            tasks.append(asyncio.create_task(self._synthesize_segment(segment_request, text, duration)))
        try:
            for task, duration in zip(tasks, durations):
                chunk = await task
                if duration:
                    chunk = (
                        ffmpeg_fit_duration(chunk, duration)
                        if request.duration_control == "ffmpeg"
                        else fit_wav_duration(chunk, duration)
                    )
                yield chunk
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def status(self) -> dict[str, Any]:
        status = await self.client.status()
        return {
            **status,
            "backend": self.name,
            "native_streaming": False,
            "streaming_mode": "sentence-level complete requests",
        }

    async def list_speakers(self) -> Any:
        return await self.client.list_voices()

    async def add_speaker(self, name: str, audio_path: str) -> Any:
        return await self.client.add_voice(name, audio_path)

    async def delete_speaker(self, name: str) -> Any:
        return await self.client.delete_voice(name)

    async def shutdown(self) -> None:
        await self.client.close()
