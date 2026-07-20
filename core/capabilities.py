from __future__ import annotations

from typing import Any, Dict

from config.settings import settings


class CapabilityRegistry:
    """Reports advanced backend readiness without pretending disabled models are active."""

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            "diarization": {
                "status": "configured" if settings.has_diarization_backend else "not_configured",
                "backend": "pyannote.audio",
                "required": ["ENABLE_DIARIZATION=true", "PYANNOTE_AUTH_TOKEN"],
            },
            "supervised_emotion_recognition": {
                "status": "configured" if settings.ENABLE_SUPERVISED_EMOTION else "not_configured",
                "backend": settings.EMOTION_MODEL_ID,
                "required": ["ENABLE_SUPERVISED_EMOTION=true", "transformers", "torch"],
            },
            "native_audio_model": {
                "status": "configured" if settings.ENABLE_NATIVE_AUDIO_MODEL and settings.NATIVE_AUDIO_MODEL_ID else "not_configured",
                "backend": settings.NATIVE_AUDIO_MODEL_ID or "unset",
                "required": ["ENABLE_NATIVE_AUDIO_MODEL=true", "NATIVE_AUDIO_MODEL_ID"],
            },
            "streaming_barge_in": {
                "status": "planned" if settings.ENABLE_STREAMING_PLACEHOLDER else "batch_mode",
                "backend": "Pipecat/WebRTC service",
                "required": ["streaming server", "browser microphone permission", "VAD event loop"],
            },
        }

