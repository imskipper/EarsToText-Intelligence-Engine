from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


@dataclass
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int


class AcousticIntelligencePipeline:
    """Extracts deterministic acoustic metrics with optional librosa pitch support."""

    def analyze(self, audio_path: str, speech_segments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        audio = self._load_audio(audio_path)
        duration = len(audio.samples) / audio.sample_rate if audio.sample_rate else 0.0
        frame_times, rms = self._frame_rms(audio.samples, audio.sample_rate)
        silence_regions = self._silence_regions(frame_times, rms)
        pitch = self._pitch_metrics(audio_path, audio)
        operational = self._operational_metrics(duration, silence_regions, speech_segments or [])

        return {
            "duration_seconds": round(duration, 3),
            "rms_mean": round(float(np.mean(rms)), 6) if len(rms) else 0.0,
            "rms_peak": round(float(np.max(rms)), 6) if len(rms) else 0.0,
            "silence_regions": silence_regions,
            "operational_metrics": operational,
            "pitch_metrics": pitch,
            "speaker_timeline": self._speaker_timeline(speech_segments or []),
            "emotion_regions": self._emotion_regions(frame_times, rms, pitch),
            "capabilities": {
                "diarization": "not_configured",
                "emotion_model": "heuristic_only",
                "native_audio_model": "not_configured",
                "streaming": "batch_upload",
            },
        }

    def _load_audio(self, audio_path: str) -> AudioBuffer:
        try:
            import librosa

            samples, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            return AudioBuffer(samples.astype(np.float32), int(sample_rate))
        except Exception:
            samples, sample_rate = sf.read(audio_path, always_2d=False)
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1)
            return AudioBuffer(samples.astype(np.float32), int(sample_rate))

    def _frame_rms(self, samples: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(samples) == 0 or sample_rate <= 0:
            return np.array([]), np.array([])

        frame_length = max(int(sample_rate * 0.025), 1)
        hop_length = max(int(sample_rate * 0.010), 1)
        starts = np.arange(0, max(len(samples) - frame_length, 1), hop_length)
        if len(starts) == 0:
            starts = np.array([0])

        rms_values = []
        for start in starts:
            frame = samples[start:start + frame_length]
            rms_values.append(np.sqrt(np.mean(np.square(frame))) if len(frame) else 0.0)

        times = starts / sample_rate
        return times, np.array(rms_values, dtype=np.float32)

    def _silence_regions(self, frame_times: np.ndarray, rms: np.ndarray) -> List[Dict[str, float]]:
        if len(frame_times) == 0 or len(rms) == 0:
            return []

        noise_floor = float(np.percentile(rms, 10))
        speech_level = float(np.percentile(rms, 95))
        dynamic_threshold = noise_floor + ((speech_level - noise_floor) * 0.1)
        threshold = max(dynamic_threshold, float(np.max(rms)) * 0.02)
        silent = rms <= threshold
        regions = []
        start: Optional[float] = None

        for idx, is_silent in enumerate(silent):
            time_value = float(frame_times[idx])
            if is_silent and start is None:
                start = time_value
            elif not is_silent and start is not None:
                if time_value - start >= 0.25:
                    regions.append({"start": round(start, 2), "end": round(time_value, 2), "duration": round(time_value - start, 2)})
                start = None

        if start is not None:
            end = float(frame_times[-1])
            if end - start >= 0.25:
                regions.append({"start": round(start, 2), "end": round(end, 2), "duration": round(end - start, 2)})

        return regions

    def _pitch_metrics(self, audio_path: str, audio: AudioBuffer) -> Dict[str, Any]:
        try:
            import librosa

            f0, voiced_flag, _ = librosa.pyin(
                audio.samples,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=audio.sample_rate,
            )
            voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
            if len(voiced_f0) < 3:
                return {"status": "insufficient_voiced_audio", "f0_mean_hz": None, "jitter_percent": None, "shimmer_percent": None}

            periods = 1.0 / voiced_f0
            jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100
            shimmer = np.std(voiced_f0) / np.mean(voiced_f0) * 100
            return {
                "status": "ok",
                "f0_mean_hz": round(float(np.mean(voiced_f0)), 2),
                "f0_min_hz": round(float(np.min(voiced_f0)), 2),
                "f0_max_hz": round(float(np.max(voiced_f0)), 2),
                "jitter_percent": round(float(jitter), 3),
                "shimmer_percent": round(float(shimmer), 3),
            }
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": str(exc),
                "f0_mean_hz": None,
                "jitter_percent": None,
                "shimmer_percent": None,
            }

    def _operational_metrics(
        self,
        duration: float,
        silence_regions: List[Dict[str, float]],
        speech_segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        silence_seconds = sum(region["duration"] for region in silence_regions)
        active_seconds = max(duration - silence_seconds, 0.0)
        cross_talk = self._cross_talk_regions(speech_segments)

        return {
            "active_speaking_seconds": round(active_seconds, 2),
            "silence_seconds": round(silence_seconds, 2),
            "silence_percent": round((silence_seconds / duration * 100), 2) if duration else 0.0,
            "active_speaking_percent": round((active_seconds / duration * 100), 2) if duration else 0.0,
            "cross_talk_regions": cross_talk,
            "cross_talk_count": len(cross_talk),
        }

    def _cross_talk_regions(self, speech_segments: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        regions = []
        sorted_segments = sorted(speech_segments, key=lambda item: item["timestamps"][0])
        for previous, current in zip(sorted_segments, sorted_segments[1:]):
            overlap_start = max(previous["timestamps"][0], current["timestamps"][0])
            overlap_end = min(previous["timestamps"][1], current["timestamps"][1])
            if overlap_end > overlap_start:
                regions.append({"start": round(overlap_start, 2), "end": round(overlap_end, 2), "duration": round(overlap_end - overlap_start, 2)})
        return regions

    def _speaker_timeline(self, speech_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "speaker": "Unknown",
                "start": segment["timestamps"][0],
                "end": segment["timestamps"][1],
                "text": segment["text"],
            }
            for segment in speech_segments
        ]

    def _emotion_regions(self, frame_times: np.ndarray, rms: np.ndarray, pitch: Dict[str, Any]) -> List[Dict[str, Any]]:
        if len(frame_times) == 0 or len(rms) == 0:
            return []

        high_energy_threshold = float(np.percentile(rms, 90))
        regions = []
        for time_value, rms_value in zip(frame_times, rms):
            if rms_value >= high_energy_threshold and high_energy_threshold > 0:
                regions.append({
                    "start": round(float(time_value), 2),
                    "end": round(float(time_value + 0.25), 2),
                    "label": "high_arousal_candidate",
                    "basis": "energy_spike",
                })

        return regions[:20]
