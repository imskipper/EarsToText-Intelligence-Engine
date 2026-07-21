# OmniSpeech-Pipeline-Engine(OmniAudio Intelligence Platform)

An end-to-end, multi-stage speech intelligence pipeline that converts raw audio into high-confidence transcripts, detailed acoustic profiles, PII-redacted text, and schema-validated structured semantic reports.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![ASR Engine](https://img.shields.io/badge/ASR-faster--whisper-orange?style=flat-square)
![Structured LLM](https://img.shields.io/badge/LLM-Instructor%20%2B%20OpenAI-412991?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)

---

## 💡 Overview

**OmniSpeech-ATS-Pipeline** provides a full-spectrum audio analytics platform designed for compliance-sensitive environments, customer interaction analysis, and automated triage.

The pipeline combines whisper-based ASR with signal-processing acoustic algorithms (`librosa`), strict local regex-based PII redaction, and schema-validated LLM extraction (`instructor`) to extract both **what was said** and **how it was said** safely and accurately.

---

## 🏗️ Architecture & Pipeline Flow

~~~text
   [ Raw Audio Input ]
           │
           ▼
┌───────────────────────┐
│     ASR Engine        │  ► faster-whisper + Silero VAD
│                       │  ► Segment & word-level confidence + timestamps
└──────────┬────────────┘
           │
     ┌─────┴────────────────────────┐
     ▼                              ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│  Acoustic Analytics     │  │   PII Redaction Engine  │
│  ► Frame-level RMS      │  │   ► Pre-LLM scrubbing   │
│  ► PyIN Pitch (F0)      │  │   ► Emails, phones, SSNs│
│  ► Jitter & Shimmer     │  │   ► Dosages & accounts  │
│  ► Cross-talk regions   │  └──────────┬──────────────┘
└─────────────────────────┘             │
                                        ▼
                             ┌─────────────────────────┐
                             │    Semantic Analytics   │
                             │    ► Instructor + LLM   │
                             │    ► Urgency Triage     │
                             │    ► Structured JSON    │
                             └──────────┬──────────────┘
                                        │
                                        ▼
                             ┌─────────────────────────┐
                             │    Streamlit Dashboard  │
                             │    ► Heatmaps & Metrics │
                             │    ► Capability Registry│
                             └─────────────────────────┘
~~~

---

## ✨ Key Features & Modules

### 1. 🎙️ High-Precision ASR Engine (`asr_engine.py`)
* Powered by **`faster-whisper`** with integrated **Voice Activity Detection (VAD)** filtering.
* Extracts segment-level and word-level timestamps with token confidence scores.

### 2. 📊 Acoustic & Signal Processing (`acoustic.py`)
* **Voice Quality Metrics:** Uses `librosa.pyin` for fundamental frequency tracking, computing exact **jitter** (pitch instability) and **shimmer** (amplitude variation).
* **Cross-Talk & Energy Spike Analysis:** RMS-based frame evaluation detects audio clipping, loud energy bursts, and potential cross-talk regions across overlapping segments.

### 3. 🛡️ Local PII Redaction Layer (`redaction.py`)
* **Deterministic & Privacy-First:** Redacts sensitive Personal Identifiable Information (PII)—including email addresses, phone numbers, SSNs, financial account IDs, and medical dosages—*before* sending text off-device.

### 4. 🧠 Schema-Validated Semantic Analytics (`analytics.py`)
* Built with **`instructor`** on top of OpenAI models to force strictly validated outputs.
* Extracts key entities, technical jargon, urgency triage levels, and actionable call summaries.

### 5. 🎛️ Interactive Streamlit Interface (`web_app/main.py`)
* Real-time transcript visualization with word-confidence heatmaps.
* **Capability Registry Pattern:** Explicitly surfaces engine state (active vs. unconfigured modules like diarization or supervised emotion) rather than silently failing.

---

## 🛠️ Design Decisions

* **Pre-LLM Redaction:** PII scrubbing runs locally *prior* to calling external LLM APIs to guarantee privacy and regulatory compliance.
* **Decoupled Acoustic Signal Analysis:** Voice quality indicators (RMS, jitter, shimmer) run independently of text transcription, ensuring acoustic metrics remain unskewed by language models.
* **Schema Enforcement over Prompting:** Leveraging `instructor` guarantees structured JSON output matching exact Pydantic schemas.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* FFmpeg installed on your system path (required for audio decoding via `librosa`/`whisper`)

### Installation

1. **Clone the repository:**
   ~~~bash
   git clone https://github.com/imskipper/OmniSpeech-ATS-Pipeline.git
   cd OmniSpeech-ATS-Pipeline
   ~~~

2. **Create and activate a virtual environment:**
   ~~~bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ~~~

3. **Install dependencies:**
   ~~~bash
   pip install -r requirements.txt
   ~~~

4. **Set up Environment Variables:**
   Create a `.env` file in the root directory:
   ~~~env
   OPENAI_API_KEY=your_openai_api_key_here
   ~~~

---

## 🖥️ Usage

Run the Streamlit web application:

~~~bash
streamlit run web_app/main.py
~~~

### Running Tests

Execute the unit test suite:

~~~bash
pytest
~~~

---

## 🛣️ Capabilities & Roadmap

| Feature | Status | Implementation Details |
| :--- | :--- | :--- |
| **VAD & ASR** | ✅ Active | `faster-whisper` + Silero |
| **Pitch & Jitter Analysis** | ✅ Active | `librosa.pyin` RMS frame analysis |
| **PII Redaction** | ✅ Active | Local regex-based deterministic scrubber |
| **Structured LLM Triage** | ✅ Active | `instructor` + Pydantic validation |
| **Speaker Diarization** | 🟡 Roadmap | Extension point stubbed for `pyannote.audio` integration |
| **Supervised Emotion Models** | 🟡 Roadmap | Currently powered by energy/RMS heuristics |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
