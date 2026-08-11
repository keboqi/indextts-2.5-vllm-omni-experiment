from __future__ import annotations

import io
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

from indextts25_compat.audio import ffmpeg_fit_duration, join_wav, wav_duration_ms
from indextts25_compat.backend import IndexTTS25Backend
from indextts25_compat.client import OmniClient
from indextts25_compat.models import SynthesisRequest
from indextts25_compat.text import allocate_durations, split_text


def wav_bytes(frames: int = 2205) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(22050)
        target.writeframes(b"\x01\x00" * frames)
    return output.getvalue()


class FakeClient:
    def __init__(self) -> None:
        self.payloads = []

    async def synthesize(self, payload):
        self.payloads.append(payload)
        return wav_bytes()

    async def status(self):
        return {"ready": True}

    async def close(self):
        pass


class FakeResponse:
    content = b'{"success":true}'

    def raise_for_status(self):
        pass

    def json(self):
        return {"success": True}


class FakeHTTP:
    def __init__(self) -> None:
        self.call = None

    async def post(self, path, **kwargs):
        self.call = (path, kwargs)
        return FakeResponse()


class CompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_named_voice_upload_matches_omni_multipart_contract(self):
        client = object.__new__(OmniClient)
        client._http = FakeHTTP()
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "voice.wav"
            audio.write_bytes(wav_bytes())
            result = await client.add_voice(
                "alice",
                str(audio),
                consent="consent-1",
                ref_text="hello",
                speaker_description="test voice",
            )
        self.assertEqual(result, {"success": True})
        path, kwargs = client._http.call
        self.assertEqual(path, "/v1/audio/voices")
        self.assertEqual(kwargs["data"]["name"], "alice")
        self.assertEqual(kwargs["data"]["consent"], "consent-1")
        self.assertIn("audio_sample", kwargs["files"])

    def test_split_and_duration_allocation_preserve_total(self):
        texts = split_text("One. Two words! Three.", 120)
        durations = allocate_durations(texts, total_ms=1800, silence_ms=100)
        self.assertEqual(texts, ["One.", "Two words!", "Three."])
        self.assertEqual(sum(durations) + 200, 1800)

    def test_split_long_cjk_text(self):
        self.assertEqual(split_text("一二三四五六七", 3), ["一二三", "四五六", "七"])

    def test_join_wav_inserts_silence(self):
        joined = join_wav([wav_bytes(2205), wav_bytes(2205)], silence_ms=100)
        with wave.open(io.BytesIO(joined), "rb") as source:
            self.assertEqual(source.getnframes(), 2205 * 3)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_ffmpeg_duration_fallback(self):
        fitted = ffmpeg_fit_duration(wav_bytes(2205), target_ms=200)
        self.assertAlmostEqual(wav_duration_ms(fitted), 200, delta=2)

    async def test_payload_maps_current_backend_controls(self):
        client = FakeClient()
        backend = IndexTTS25Backend(client)  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            request = SynthesisRequest(
                text="Hello.",
                output_path=Path(directory) / "out.wav",
                speaker_preset="alice",
                language="en",
                target_duration_ms=1000,
                emotion_text="happy",
                emotion_weight=0.7,
                diffusion_steps=12,
                seed=42,
                sampling={"temperature": 0.8, "max_new_tokens": 700},
            )
            await backend.synthesize(request)
        payload = client.payloads[0]
        self.assertEqual(payload["voice"], "alice")
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["max_new_tokens"], 700)
        self.assertEqual(payload["extra_params"]["lang"], "en")
        self.assertEqual(payload["extra_params"]["target_duration_ms"], 1000)
        self.assertEqual(payload["extra_params"]["diffusion_steps"], 12)
        self.assertTrue(payload["extra_params"]["use_emo_text"])

    async def test_stream_is_sentence_level_and_seeded_per_segment(self):
        client = FakeClient()
        backend = IndexTTS25Backend(client)  # type: ignore[arg-type]
        request = SynthesisRequest(
            text="First. Second.",
            output_path=Path("unused.wav"),
            speaker_preset="alice",
            seed=10,
        )
        chunks = [chunk async for chunk in backend.stream(request)]
        self.assertEqual(len(chunks), 2)
        self.assertEqual([payload["seed"] for payload in client.payloads], [10, 11])


if __name__ == "__main__":
    unittest.main()
