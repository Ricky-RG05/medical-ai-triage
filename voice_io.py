import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import tempfile
import os
import time
import torch
from melo.api import TTS as MeloTTS

# ─────────────────────────────────────────────
# Device detection (GPU if available, else CPU)
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Load Whisper (Speech-to-Text)
# ─────────────────────────────────────────────
print(f"🎙️  Cargando Whisper en: {device.upper()}...")
whisper_model = whisper.load_model("small", device=device)
print(f"✅ Whisper listo en {device.upper()}.\n")

# ─────────────────────────────────────────────
# Load MeloTTS (Text-to-Speech) — Spanish (Spain)
# ─────────────────────────────────────────────
print("🔊 Cargando MeloTTS...")
tts_model  = MeloTTS(language="ES", device=device)
speaker_id = tts_model.hps.data.spk2id["ES"]
print("✅ MeloTTS listo.\n")

# ─────────────────────────────────────────────
# Tunable config — adjust after looking at the debug output
# ─────────────────────────────────────────────
SILENCE_THRESHOLD = 0.008   # lowered from 0.02 — typical laptop mic baseline is ~0.003
SILENCE_SECONDS   = 1.8     # slightly longer so we don't cut off mid-thought
POST_SPEAK_DELAY  = 0.35    # seconds to wait AFTER TTS finishes, to avoid echo into mic
DEBUG_LISTEN      = True    # prints max volume + saves last recording to debug_last.wav


def listen(duration: int = 10,
           samplerate: int = 16000,
           silence_threshold: float = SILENCE_THRESHOLD,
           silence_seconds: float = SILENCE_SECONDS) -> str:
    """
    Records microphone audio with automatic silence-based cutoff,
    then transcribes it with Whisper. Returns the Spanish transcript.
    """
    print("  🎙️  Escuchando... (silencio para terminar)")

    recorded_chunks   = []
    silent_chunks     = 0
    started_speaking  = False
    stop_flag         = [False]
    max_volume_seen   = [0.0]
    chunk_samples     = int(samplerate * 0.1)
    max_silent_chunks = int(silence_seconds / 0.1)

    def callback(indata, frames, time_info, status):
        nonlocal silent_chunks, started_speaking
        chunk = indata.copy()
        recorded_chunks.append(chunk)
        volume = float(np.abs(chunk).mean())
        if volume > max_volume_seen[0]:
            max_volume_seen[0] = volume

        if volume > silence_threshold:
            started_speaking = True
            silent_chunks = 0
        elif started_speaking:
            silent_chunks += 1
            if silent_chunks >= max_silent_chunks:
                stop_flag[0] = True

    with sd.InputStream(samplerate=samplerate, channels=1,
                        dtype="float32", blocksize=chunk_samples,
                        callback=callback):
        elapsed = 0.0
        while elapsed < duration:
            sd.sleep(100)
            elapsed += 0.1
            if stop_flag[0]:
                break

    total_duration = len(recorded_chunks) * 0.1
    print(f"  ✅ Grabación terminada. "
          f"Duración: {total_duration:.1f}s | "
          f"Volumen máx: {max_volume_seen[0]:.4f} | "
          f"Umbral: {silence_threshold} | "
          f"Habla detectada: {started_speaking}")

    if not recorded_chunks:
        return ""

    audio = np.concatenate(recorded_chunks, axis=0)

    # ── DEBUG: save the raw recording so you can play it back ──
    if DEBUG_LISTEN:
        sf.write("debug_last.wav", audio, samplerate)
        print(f"  🐛 Audio crudo guardado en: debug_last.wav")

    # Safety: if we clearly didn't capture any speech, bail early
    if max_volume_seen[0] < silence_threshold * 0.5:
        print("  ⚠️  No se detectó voz (volumen muy bajo). Ajusta SILENCE_THRESHOLD.")
        return ""

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, samplerate)
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path, language="es")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass

    text = result["text"].strip()
    print(f"  ✍️  Transcrito: {text!r}")
    return text


def speak(text: str, speed: float = 1.1) -> None:
    """
    Synthesizes `text` in Spanish with MeloTTS and plays it blockingly.
    """
    if not text or not text.strip():
        return

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        tts_model.tts_to_file(
            text       = text,
            speaker_id = speaker_id,
            output_path= tmp_path,
            speed      = speed,
            quiet      = True,
        )

        data, sr = sf.read(tmp_path)
        sd.play(data, sr)
        sd.wait()

        # Small post-speak delay so speaker echo doesn't leak into the next listen()
        time.sleep(POST_SPEAK_DELAY)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass


# ─────────────────────────────────────────────
# Standalone diagnostic — shows you exactly what the mic is hearing
# Run:  python voice_io.py
# Then speak clearly for ~5 seconds. Look at "Volumen máx" in the output.
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 Prueba de síntesis de voz...")
    speak("Hola, este es un sistema de triaje médico. ¿Cómo se siente hoy?")
    print("\n🧪 Prueba de reconocimiento. Habla claramente durante 5 segundos:")
    heard = listen(duration=8)
    print(f"\n📝 Resultado final: {heard!r}")
    print("\n▶️  Puedes reproducir 'debug_last.wav' para escuchar lo que grabó el micrófono.")