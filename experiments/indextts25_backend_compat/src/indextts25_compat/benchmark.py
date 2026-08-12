from __future__ import annotations

import math
from typing import Any, Iterable


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_gpu_samples(samples: list[dict[str, Any]]) -> dict[str, float | int | None]:
    def values(key: str) -> list[float]:
        return [
            float(sample[key])
            for sample in samples
            if isinstance(sample.get(key), (int, float)) and math.isfinite(float(sample[key]))
        ]

    memory = values("memory_used_mb")
    utilization = values("utilization_percent")
    power = values("power_w")
    temperature = values("temperature_c")
    return {
        "gpu_sample_count": len(samples),
        "memory_start_mb": round(memory[0], 1) if memory else None,
        "memory_peak_mb": round(max(memory), 1) if memory else None,
        "memory_peak_delta_mb": round(max(memory) - memory[0], 1) if memory else None,
        "gpu_utilization_avg_percent": round(sum(utilization) / len(utilization), 1) if utilization else None,
        "gpu_utilization_peak_percent": round(max(utilization), 1) if utilization else None,
        "power_avg_w": round(sum(power) / len(power), 1) if power else None,
        "power_peak_w": round(max(power), 1) if power else None,
        "temperature_peak_c": round(max(temperature), 1) if temperature else None,
    }


def summarize_concurrency_results(
    results: list[dict[str, Any]],
    *,
    wall_s: float,
) -> dict[str, float | int | None]:
    passed = [result for result in results if result.get("status") == "pass"]
    latencies = [float(result["elapsed_s"]) for result in passed]
    total_audio_s = sum(float(result["duration_ms"]) for result in passed) / 1000.0
    failures = len(results) - len(passed)
    return {
        "request_count": len(results),
        "successes": len(passed),
        "failures": failures,
        "success_rate": round(len(passed) / len(results), 4) if results else 0.0,
        "wall_s": round(wall_s, 3),
        "requests_per_s": round(len(passed) / wall_s, 3) if wall_s > 0 else None,
        "total_audio_s": round(total_audio_s, 3),
        "audio_s_per_wall_s": round(total_audio_s / wall_s, 3) if wall_s > 0 else None,
        "aggregate_rtf": round(wall_s / total_audio_s, 4) if total_audio_s > 0 else None,
        "latency_mean_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "latency_p50_s": round(percentile(latencies, 0.50), 3) if latencies else None,
        "latency_p95_s": round(percentile(latencies, 0.95), 3) if latencies else None,
        "latency_p99_s": round(percentile(latencies, 0.99), 3) if latencies else None,
        "latency_max_s": round(max(latencies), 3) if latencies else None,
    }
