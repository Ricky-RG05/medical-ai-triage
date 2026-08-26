[README.md](https://github.com/user-attachments/files/31490692/README.md)
# 🏥 Offline AI Medical Triage System

A fully offline, voice-enabled AI system that triages patient symptoms against official Mexican Clinical Practice Guidelines (GPCs) and generates structured, doctor-ready referral reports — no internet connection required at runtime.

Built independently and demoed live to the leadership of a Farmacias del Ahorro chain in Monterrey's Headquarters.

## Status

Core system is complete and functional — successfully demoed end-to-end with real clinical guideline data. Currently pending the client's internal rollout decision. The project is actively maintained and continues to evolve (e.g. broader condition coverage, dialect-tuned speech recognition).

## How it works

1. The AI greets the patient and has an ongoing conversation with the patient, in which the purpose of the visit, the symptoms and known medical conditions are explained and discussed.
2. Speech is transcribed locally (no cloud API calls).
3. A RAG pipeline retrieves the relevant official clinical guidelines stored in an embeddings system (ChromaDB), scoped to the matching condition category described by the patient.
4. (To be added soon!): The AI-model retrieves the medical record of the patient based on his/her personal credentials, obtained from a Database managed by Farmacias del Ahorro. 
5. A structured PDF report is generated for the patient/pharmacist/government, with a `NO_MATCH` fallback that flags out-of-scope cases for human referral instead of guessing.
6. Report is analyzed by a pharmacist online, further doctor referal and/or a specific treatment for the patient is confirmed, and concrete drug is provided automatically by an automatic dispenser after payment.

## Tech stack

| Component | Tool |
|---|---|
| Speech-to-text | [Whisper](https://github.com/openai/whisper) |
| Text-to-speech | [MeloTTS](https://github.com/myshell-ai/MeloTTS) |
| LLM runtime | [Ollama](https://ollama.com) (fully local) |
| Vector store / RAG | [ChromaDB](https://www.trychroma.com/) + `nomic-embed-text-v2-moe` embeddings |
| Knowledge base | Official Mexican Clinical Practice Guidelines (GPCs), auto-synced from Google Drive |
| Report generation | [ReportLab](https://www.reportlab.com/) (PDF) |
| Language | Python |

## Architecture

```
main.py                 → entry point / full orchestration
conversation.py          → dialogue flow & symptom intake
voice_io.py               → speech-to-text / text-to-speech handling
embedding_generator.py    → embedding pipeline for the knowledge base (GPC-Guidelines)
gpc_updater.py            → Google Drive sync for clinical guideline updates (Updated once per month automatically with fallback solution in case no internet is available at the time of execution; sha256)
report_generator.py       → PDF triage report generation
```

**Design notes:**
- Retrieval is scoped across nine condition-specific folders rather than one flat knowledge base, which sharply reduces irrelevant context during retrieval.
- A dedicated `NO_MATCH` path routes out-of-scope symptoms to human referral instead of forcing a guess.
- Chunk overlap and batching were tuned specifically to make local embedding generation reliable and to improve retrieval quality on scanned/OCR'd source PDFs.

## Installation (macOS)

### 1. Install prerequisites

```bash
# Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# FFmpeg
brew install ffmpeg

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Run the setup script

```bash
cd path/to/medical-ai-triage
chmod +x install.sh
./install.sh
```

This downloads the local AI models the first time, so it takes roughly 30–45 minutes. Let it run without closing the terminal.

### 3. Run the system

```bash
cd path/to/medical-ai-triage
./run.sh
```

## Disclaimer

This is a research/demo prototype, not a certified medical device. It is not intended to replace professional medical diagnosis or treatment, and the clinical guideline data used here should be validated for any real-world deployment.

## Author

Ricardo Ramírez Gutiérrez
