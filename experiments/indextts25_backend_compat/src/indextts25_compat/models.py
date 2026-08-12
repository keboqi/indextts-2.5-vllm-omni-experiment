from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


SUPPORTED_LANGUAGES = ("zh", "en", "ja", "es", "ar")


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    native_streaming: bool
    native_duration: bool
    emotion_text: bool
    reference_audio: bool = True


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """Superset of the current IndexTTS 2.0 backend request contract."""

    text: str
    output_path: Path
    speaker_preset: str | None = None
    prompt_audio: str | None = None
    reference_text: str | None = None
    language: str | None = None
    interval_silence_ms: int = 0
    target_duration_ms: int = 0
    duration_control: str = "original"
    max_text_tokens: int = 120
    diffusion_steps: int = 10
    verbose: bool = False
    emotion_audio: str | None = None
    emotion_text: str | None = None
    emotion_weight: float = 0.6
    emotion_vector: Sequence[float] | None = None
    random_emotion: bool = False
    cache_prompt_audio: bool = True
    seed: int | None = None
    sampling: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.text.strip():
            raise ValueError("text cannot be empty")
        if not self.prompt_audio and not self.speaker_preset:
            raise ValueError("prompt_audio or speaker_preset is required")
        if self.interval_silence_ms < 0 or self.target_duration_ms < 0:
            raise ValueError("durations cannot be negative")
        if self.duration_control not in {"original", "native", "ffmpeg"}:
            raise ValueError("duration_control must be original, native, or ffmpeg")
        if self.max_text_tokens < 1 or self.diffusion_steps < 1:
            raise ValueError("max_text_tokens and diffusion_steps must be positive")
        if not 0.0 <= self.emotion_weight <= 1.0:
            raise ValueError("emotion_weight must be between 0 and 1")
        if self.emotion_vector is not None and len(self.emotion_vector) != 8:
            raise ValueError("emotion_vector must contain exactly 8 values")
        if self.language is not None and self.language.strip().lower() not in SUPPORTED_LANGUAGES:
            raise ValueError(
                "unsupported IndexTTS 2.5 synthesis language; expected one of "
                + ", ".join(SUPPORTED_LANGUAGES)
            )
