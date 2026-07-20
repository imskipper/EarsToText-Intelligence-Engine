import logging
import math
import time
from typing import Dict, Any

from faster_whisper import WhisperModel

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ASREngine")


class OptimizedASRPipeline:
    def __init__(self):
        logger.info(f"Loading Quantized Neural ASR Engine: [{settings.WHISPER_MODEL_SIZE}]")
        # Engine auto-detects CPU vs GPU footprints
        self.model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=settings.ASR_DEVICE,
            compute_type=settings.COMPUTE_TYPE
        )

    def process_audio_stream(self, audio_source_path: str) -> Dict[str, Any]:
        """
        Executes high-speed speech-to-text decoding.
        Applies Silero VAD filters natively to strip non-speech artifacts.
        """
        start_time = time.time()

        segments, info = self.model.transcribe(
            audio_source_path,
            beam_size=settings.BEAM_SIZE,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=settings.VAD_MIN_SILENCE_MS),
            word_timestamps=settings.ENABLE_WORD_TIMESTAMPS,
        )

        compiled_segments = []
        full_transcript = []
        compiled_words = []

        for segment in segments:
            full_transcript.append(segment.text)
            segment_words = []

            for word in segment.words or []:
                probability = float(word.probability or 0.0)
                word_payload = {
                    "word": word.word.strip(),
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                    "confidence": round(probability, 3),
                }
                segment_words.append(word_payload)
                compiled_words.append(word_payload)

            confidence_score = float(segment.avg_logprob)
            if math.isnan(confidence_score) or math.isinf(confidence_score):
                confidence_score = 0.0

            compiled_segments.append({
                "id": segment.id,
                "timestamps": [round(segment.start, 2), round(segment.end, 2)],
                "text": segment.text.strip(),
                "confidence_score": round(confidence_score, 4),
                "words": segment_words,
            })

        execution_latency = time.time() - start_time

        return {
            "metadata": {
                "inferred_language": info.language,
                "language_confidence": round(info.language_probability, 2),
                "pipeline_latency_seconds": round(execution_latency, 3)
            },
            "raw_transcript": " ".join(full_transcript).strip(),
            "timeline_matrix": compiled_segments,
            "word_confidence": compiled_words,
        }
