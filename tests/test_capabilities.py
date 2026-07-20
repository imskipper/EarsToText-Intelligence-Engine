from core.capabilities import CapabilityRegistry


def test_capability_snapshot_exposes_required_backends():
    snapshot = CapabilityRegistry().snapshot()

    assert "diarization" in snapshot
    assert "supervised_emotion_recognition" in snapshot
    assert "native_audio_model" in snapshot
    assert "streaming_barge_in" in snapshot
