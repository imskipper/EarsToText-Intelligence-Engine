import os
import tempfile

import numpy as np
import soundfile as sf

from core.acoustic import AcousticIntelligencePipeline


def test_acoustic_pipeline_handles_simple_tone():
    sample_rate = 16000
    timeline = np.linspace(0, 1, sample_rate, False)
    samples = (0.1 * np.sin(2 * np.pi * 220 * timeline)).astype("float32")
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_file.close()

    try:
        sf.write(audio_file.name, samples, sample_rate)
        result = AcousticIntelligencePipeline().analyze(
            audio_file.name,
            [{"timestamps": [0.0, 1.0], "text": "test"}],
        )
    finally:
        os.unlink(audio_file.name)

    assert result["duration_seconds"] == 1.0
    assert result["operational_metrics"]["active_speaking_percent"] >= 95.0
    assert result["pitch_metrics"]["status"] in {"ok", "insufficient_voiced_audio", "unavailable"}
