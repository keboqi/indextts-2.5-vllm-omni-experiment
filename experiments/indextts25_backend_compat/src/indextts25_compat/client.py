from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:  # Allows contract/unit tests with an injected fake client.
    httpx = None  # type: ignore[assignment]


def audio_reference(value: str) -> str:
    if value.startswith(("http://", "https://", "data:", "file://")):
        return value
    path = Path(value).expanduser().resolve()
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class OmniClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 600.0) -> None:
        if httpx is None:
            raise RuntimeError("OmniClient requires the optional runtime dependency 'httpx'")
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def synthesize(self, payload: dict[str, Any]) -> bytes:
        response = await self._http.post("/v1/audio/speech", json=payload)
        response.raise_for_status()
        return response.content

    async def status(self) -> dict[str, Any]:
        try:
            response = await self._http.get("/v1/models")
            response.raise_for_status()
            return {"ready": True, "base_url": self.base_url, "models": response.json().get("data", [])}
        except Exception as exc:
            return {"ready": False, "base_url": self.base_url, "error": str(exc)}

    async def list_voices(self) -> Any:
        response = await self._http.get("/v1/audio/voices")
        response.raise_for_status()
        return response.json()

    async def add_voice(
        self,
        name: str,
        audio_path: str,
        *,
        consent: str = "user-consent",
        ref_text: str | None = None,
        speaker_description: str | None = None,
    ) -> Any:
        path = Path(audio_path)
        data = {"name": name, "consent": consent}
        if ref_text:
            data["ref_text"] = ref_text
        if speaker_description:
            data["speaker_description"] = speaker_description
        with path.open("rb") as audio:
            response = await self._http.post(
                "/v1/audio/voices",
                data=data,
                files={"audio_sample": (path.name, audio, mimetypes.guess_type(path.name)[0] or "audio/wav")},
            )
        response.raise_for_status()
        return response.json()

    async def delete_voice(self, name: str) -> Any:
        response = await self._http.delete(f"/v1/audio/voices/{name}")
        response.raise_for_status()
        return response.json() if response.content else None

    async def close(self) -> None:
        await self._http.aclose()
