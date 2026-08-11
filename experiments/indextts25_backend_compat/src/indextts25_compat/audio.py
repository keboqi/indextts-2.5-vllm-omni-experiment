from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path


def join_wav(chunks: list[bytes], silence_ms: int = 0) -> bytes:
    if not chunks:
        raise ValueError("no WAV chunks to join")
    output = io.BytesIO()
    params = None
    frames: list[bytes] = []
    for index, chunk in enumerate(chunks):
        with wave.open(io.BytesIO(chunk), "rb") as source:
            current = source.getparams()
            signature = (current.nchannels, current.sampwidth, current.framerate, current.comptype)
            if params is None:
                params = current
            elif signature != (params.nchannels, params.sampwidth, params.framerate, params.comptype):
                raise ValueError("all WAV chunks must use the same audio format")
            frames.append(source.readframes(source.getnframes()))
            if index + 1 < len(chunks) and silence_ms:
                silence_frames = round(current.framerate * silence_ms / 1000)
                frames.append(b"\x00" * silence_frames * current.nchannels * current.sampwidth)
    assert params is not None
    with wave.open(output, "wb") as target:
        target.setparams(params)
        target.writeframes(b"".join(frames))
    return output.getvalue()


def fit_wav_duration(wav: bytes, target_ms: int) -> bytes:
    """Crop or zero-pad to the nearest sample; no time stretching is applied."""
    source_io = io.BytesIO(wav)
    output = io.BytesIO()
    with wave.open(source_io, "rb") as source:
        params = source.getparams()
        raw = source.readframes(source.getnframes())
    frame_width = params.nchannels * params.sampwidth
    target_frames = round(params.framerate * target_ms / 1000)
    target_bytes = target_frames * frame_width
    fitted = raw[:target_bytes].ljust(target_bytes, b"\x00")
    with wave.open(output, "wb") as target:
        target.setparams(params)
        target.writeframes(fitted)
    return output.getvalue()


def wav_duration_ms(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav), "rb") as source:
        return source.getnframes() * 1000.0 / source.getframerate()


def ffmpeg_fit_duration(wav: bytes, target_ms: int) -> bytes:
    """Match duration with FFmpeg atempo, mirroring the current fallback."""
    if target_ms <= 0:
        return wav
    tempo = wav_duration_ms(wav) / target_ms
    factors: list[float] = []
    while tempo > 2.0:
        factors.append(2.0)
        tempo /= 2.0
    while tempo < 0.5:
        factors.append(0.5)
        tempo /= 0.5
    factors.append(tempo)
    audio_filter = ",".join(f"atempo={factor:.10f}" for factor in factors)
    with tempfile.TemporaryDirectory(prefix="indextts25-duration-") as directory:
        source = Path(directory) / "source.wav"
        target = Path(directory) / "target.wav"
        source.write_bytes(wav)
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-filter:a",
                audio_filter,
                str(target),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"FFmpeg duration adjustment failed: {result.stderr.decode(errors='replace')}")
        return fit_wav_duration(target.read_bytes(), target_ms)


def write_audio(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
