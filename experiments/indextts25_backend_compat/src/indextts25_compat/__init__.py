"""Experimental IndexTTS 2.5 backend compatibility layer."""

from .backend import IndexTTS25Backend
from .client import OmniClient
from .models import BackendCapabilities, SynthesisRequest

__all__ = ["BackendCapabilities", "IndexTTS25Backend", "OmniClient", "SynthesisRequest"]
