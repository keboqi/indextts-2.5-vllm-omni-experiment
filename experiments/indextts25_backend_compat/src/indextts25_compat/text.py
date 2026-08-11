from __future__ import annotations

import re

_BOUNDARY = re.compile(r"(?<=[.!?。！？；;])\s*")


def split_text(text: str, max_tokens: int) -> list[str]:
    """Split conservatively without rewriting IndexTTS pronunciation markup."""
    text = text.strip()
    if not text:
        return []
    pieces = [piece.strip() for piece in _BOUNDARY.split(text) if piece.strip()]
    result: list[str] = []
    for piece in pieces:
        words = piece.split()
        if len(words) == 1 and len(piece) > max_tokens:
            result.extend(piece[i : i + max_tokens] for i in range(0, len(piece), max_tokens))
            continue
        if len(words) <= max_tokens:
            result.append(piece)
            continue
        # This is deliberately a transport guard, not the model tokenizer.
        # Languages without spaces remain intact and are normalized by 2.5.
        result.extend(" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens))
    return result


def allocate_durations(texts: list[str], total_ms: int, silence_ms: int) -> list[int]:
    if not texts:
        return []
    if total_ms <= 0:
        return [0] * len(texts)
    available = total_ms - max(0, len(texts) - 1) * silence_ms
    if available < len(texts):
        raise ValueError("target duration is shorter than requested inter-segment silence")
    weights = [max(1, len(text)) for text in texts]
    weight_sum = sum(weights)
    durations = [max(1, round(available * weight / weight_sum)) for weight in weights]
    durations[-1] += available - sum(durations)
    return durations
