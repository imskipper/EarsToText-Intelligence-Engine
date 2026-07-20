import os
import sys
import tempfile
import json
from html import escape

import pandas as pd
import streamlit as st

# Ensure project root is in system path for clean relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from core.acoustic import AcousticIntelligencePipeline
from core.analytics import StructuredAnalyticsPipeline
from core.asr_engine import OptimizedASRPipeline
from core.capabilities import CapabilityRegistry
from core.redaction import EntityRedactor
from core.reporting import build_research_report, summarize_acoustic_context

st.set_page_config(page_title="OmniAudio Intelligence Platform", layout="wide")


@st.cache_resource
def get_asr_engine():
    return OptimizedASRPipeline()


@st.cache_resource
def get_analytics_engine():
    return StructuredAnalyticsPipeline()


@st.cache_resource
def get_acoustic_engine():
    return AcousticIntelligencePipeline()


@st.cache_resource
def get_redactor():
    return EntityRedactor()


@st.cache_resource
def get_capabilities():
    return CapabilityRegistry()


def render_confidence_heatmap(words):
    if not words:
        st.info("Word-level confidence was not returned for this clip.")
        return

    html_parts = []
    for item in words:
        confidence = item["confidence"]
        if confidence < 0.7:
            color = "#f8d7da"
            border = "#dc3545"
        elif confidence < 0.85:
            color = "#fff3cd"
            border = "#ffc107"
        else:
            color = "#d1e7dd"
            border = "#198754"

        html_parts.append(
            f"<span title='confidence {confidence:.0%}' "
            f"style='display:inline-block;margin:3px;padding:3px 6px;border:1px solid {border};"
            f"background:{color};border-radius:4px;color:#1f1f1f'>{escape(item['word'])}</span>"
        )

    st.markdown("".join(html_parts), unsafe_allow_html=True)


st.title("OmniAudio Intelligence Platform")
st.markdown("*Production-Grade End-to-End Speech Decoding & Analytics Engine*")
st.markdown("---")

left_panel, right_panel = st.columns([1, 1])

with left_panel:
    st.subheader("Audio Intake Aggregator")
    audio_file = st.file_uploader("Upload Acoustic Log File", type=["wav", "mp3", "m4a"])

    if audio_file:
        st.audio(audio_file)

        if st.button("Execute Pipeline Ingestion", use_container_width=True):
            suffix = os.path.splitext(audio_file.name)[1]
            runtime_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                    temp_audio.write(audio_file.getbuffer())
                    runtime_path = temp_audio.name

                with st.spinner("Loading ASR model and transcribing audio..."):
                    asr_engine = get_asr_engine()
                    asr_data = asr_engine.process_audio_stream(runtime_path)
                    st.session_state["asr_data"] = asr_data

                with st.spinner("Extracting acoustic intelligence metrics..."):
                    acoustic_data = get_acoustic_engine().analyze(runtime_path, asr_data["timeline_matrix"])
                    st.session_state["acoustic_data"] = acoustic_data

                redaction_data = get_redactor().redact(asr_data["raw_transcript"])
                st.session_state["redaction_data"] = redaction_data

                if settings.has_openai_api_key:
                    with st.spinner("Generating structured semantic report..."):
                        analytics_engine = get_analytics_engine()
                        acoustic_context = summarize_acoustic_context(acoustic_data)
                        report_data = analytics_engine.analyze_transcript(redaction_data["redacted_text"], acoustic_context)
                        st.session_state["report_data"] = report_data.model_dump()
                else:
                    st.session_state.pop("report_data", None)
                    st.warning("OPENAI_API_KEY is not configured. Transcript completed; analytics report skipped.")

                st.session_state["research_report"] = build_research_report(
                    st.session_state["asr_data"],
                    st.session_state["acoustic_data"],
                    st.session_state["redaction_data"],
                    st.session_state.get("report_data"),
                )
            except Exception as exc:
                st.session_state.pop("report_data", None)
                st.error(f"Pipeline failed: {exc}")
            finally:
                if runtime_path and os.path.exists(runtime_path):
                    os.remove(runtime_path)

    if "asr_data" in st.session_state:
        st.success("ASR Phase Terminated Successfully")
        meta = st.session_state["asr_data"]["metadata"]
        st.metric(label="ASR Engine System Latency", value=f"{meta['pipeline_latency_seconds']} Seconds")
        st.text_area("Decoded Transcript Raw Output", value=st.session_state["asr_data"]["raw_transcript"], height=200)

        if "redaction_data" in st.session_state:
            redaction = st.session_state["redaction_data"]
            st.text_area("PII-Redacted Transcript", value=redaction["redacted_text"], height=160)
            st.caption(f"Redactions applied: {redaction['redaction_count']}")

with right_panel:
    st.subheader("Structured Semantic Reporting")

    if "report_data" in st.session_state:
        report = st.session_state["report_data"]

        priority = report["triage_metrics"]["priority_level"]
        if priority in ["HIGH", "CRITICAL"]:
            st.error(f"SYSTEM CLASSIFICATION: {priority}")
        else:
            st.info(f"SYSTEM CLASSIFICATION: {priority}")
        st.caption(f"**Justification:** {report['triage_metrics']['justification']}")

        st.markdown(f"### **Executive Breakdown:**\n{report['executive_summary']}")

        st.markdown("**Identified Key Entities & Context Matrix:**")
        for ent in report["extracted_entities"]:
            st.markdown(f"- **`{ent['entity_name']}`**: {ent['contextual_metadata']}")

        if report.get("domain_terms"):
            st.markdown("**Medical / Technical Jargon Map:**")
            for term in report["domain_terms"]:
                st.markdown(f"- **`{term['entity_name']}`**: {term['contextual_metadata']}")

        st.markdown("**Automated Next Steps Protocol:**")
        for step in report["recommended_next_steps"]:
            st.markdown(f"- [ ] {step}")
    else:
        if settings.has_openai_api_key:
            st.warning("Awaiting Ingestion Pipeline Trigger.")
        else:
            st.info("Set OPENAI_API_KEY in .env to enable structured semantic reporting.")

if "asr_data" in st.session_state:
    st.markdown("---")
    tab_timeline, tab_acoustic, tab_confidence, tab_capabilities = st.tabs(
        ["Speaker Timeline", "Acoustic Metrics", "ASR Confidence", "Model Capabilities"]
    )

    with tab_timeline:
        acoustic = st.session_state.get("acoustic_data", {})
        timeline = acoustic.get("speaker_timeline", [])
        if timeline:
            st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)
        else:
            st.info("No speech timeline is available for this clip.")

        st.caption("Speaker labels are marked Unknown until a diarization backend such as PyAnnote or SpeechBrain is configured.")

    with tab_acoustic:
        acoustic = st.session_state.get("acoustic_data", {})
        metrics = acoustic.get("operational_metrics", {})
        pitch = acoustic.get("pitch_metrics", {})

        metric_cols = st.columns(4)
        metric_cols[0].metric("Active Speech", f"{metrics.get('active_speaking_percent', 0)}%")
        metric_cols[1].metric("Silence", f"{metrics.get('silence_percent', 0)}%")
        metric_cols[2].metric("Cross-Talk Events", metrics.get("cross_talk_count", 0))
        metric_cols[3].metric("Mean F0", f"{pitch.get('f0_mean_hz') or 'N/A'} Hz")

        st.write("Pitch and Voice Stability")
        st.json({
            "pitch_status": pitch.get("status"),
            "jitter_percent": pitch.get("jitter_percent"),
            "shimmer_percent": pitch.get("shimmer_percent"),
            "f0_min_hz": pitch.get("f0_min_hz"),
            "f0_max_hz": pitch.get("f0_max_hz"),
        })

        silence_regions = acoustic.get("silence_regions", [])
        if silence_regions:
            st.write("Silence Regions")
            st.dataframe(pd.DataFrame(silence_regions), use_container_width=True, hide_index=True)

        emotion_regions = acoustic.get("emotion_regions", [])
        if emotion_regions:
            st.write("High-Arousal Acoustic Candidates")
            st.dataframe(pd.DataFrame(emotion_regions), use_container_width=True, hide_index=True)

    with tab_confidence:
        st.write("Per-Word Confidence Heatmap")
        render_confidence_heatmap(st.session_state["asr_data"].get("word_confidence", []))

        words = st.session_state["asr_data"].get("word_confidence", [])
        if words:
            low_confidence = [word for word in words if word["confidence"] < 0.7]
            st.caption(f"Low-confidence words: {len(low_confidence)}")

    with tab_capabilities:
        st.json(get_capabilities().snapshot())
        st.info(
            "Diarization, supervised emotion recognition, native audio foundation models, and websocket barge-in "
            "need additional model credentials or a streaming service. The current build keeps those integrations "
            "as explicit extension points instead of failing at startup."
        )

if "research_report" in st.session_state:
    st.download_button(
        "Download Research Report JSON",
        data=json.dumps(st.session_state["research_report"], indent=2),
        file_name="omnispeech_research_report.json",
        mime="application/json",
        use_container_width=True,
    )
