from core.reporting import build_research_report, summarize_acoustic_context


def test_build_research_report_has_expected_sections():
    report = build_research_report(
        {"raw_transcript": "hello"},
        {"duration_seconds": 1.0, "operational_metrics": {}, "pitch_metrics": {}},
        {"redacted_text": "hello"},
    )

    assert "generated_at" in report
    assert report["asr"]["raw_transcript"] == "hello"
    assert report["semantic_report"] is None


def test_summarize_acoustic_context_is_stable():
    summary = summarize_acoustic_context(
        {
            "duration_seconds": 2.0,
            "operational_metrics": {
                "active_speaking_percent": 80.0,
                "silence_percent": 20.0,
                "cross_talk_count": 0,
            },
            "pitch_metrics": {
                "f0_mean_hz": 220.0,
                "jitter_percent": 0.2,
                "shimmer_percent": 1.5,
            },
        }
    )

    assert "Duration: 2.0 seconds" in summary
    assert "Active speech: 80.0%" in summary
