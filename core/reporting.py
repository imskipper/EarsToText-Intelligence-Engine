from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


def build_research_report(
    asr_data: Dict[str, Any],
    acoustic_data: Dict[str, Any],
    redaction_data: Dict[str, Any],
    semantic_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asr": asr_data,
        "acoustic_intelligence": acoustic_data,
        "redaction": redaction_data,
        "semantic_report": semantic_report,
    }


def summarize_acoustic_context(acoustic_data: Dict[str, Any]) -> str:
    metrics = acoustic_data.get("operational_metrics", {})
    pitch = acoustic_data.get("pitch_metrics", {})
    return (
        f"Duration: {acoustic_data.get('duration_seconds')} seconds. "
        f"Active speech: {metrics.get('active_speaking_percent')}%. "
        f"Silence: {metrics.get('silence_percent')}%. "
        f"Cross-talk events: {metrics.get('cross_talk_count')}. "
        f"Mean F0: {pitch.get('f0_mean_hz')} Hz. "
        f"Jitter: {pitch.get('jitter_percent')}%. "
        f"Shimmer: {pitch.get('shimmer_percent')}%."
    )
