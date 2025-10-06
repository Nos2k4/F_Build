import streamlit as st
import sounddevice as sd
import queue
import threading
import numpy as np
import whisper
import tempfile
import os
import soundfile as sf
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Load models
# ----------------------------
whisper_model = whisper.load_model("base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Shared queue for audio
# ----------------------------
audio_q = queue.Queue()

# ----------------------------
# Example reference dataset (can be replaced with plagiarism DB or AI samples)
# ----------------------------
reference_texts = [
    "Artificial intelligence is transforming the world.",
    "Deep learning is a subset of machine learning.",
    "Neural networks can learn patterns in data.",
    "Transformers revolutionized natural language processing.",
]

reference_embeddings = embedding_model.encode(reference_texts, convert_to_tensor=True)

# ----------------------------
# Helper Functions
# ----------------------------
def plagiarism_confidence(text: str) -> (str, float):
    """Return plagiarism message and confidence score (0-100%)."""
    if not text.strip():
        return "No speech detected yet.", 0.0

    test_emb = embedding_model.encode([text], convert_to_tensor=True)
    cos_scores = util.cos_sim(test_emb, reference_embeddings)
    max_score = float(cos_scores.max())  # highest similarity with DB

    if max_score > 0.65:
        return "⚠️ Possible plagiarism / AI-generated", max_score * 100
    else:
        return "✅ Likely original speech", max_score * 100

# ----------------------------
# Streamlit State
# ----------------------------
if "transcribed" not in st.session_state:
    st.session_state.transcribed = ""
if "plagiarism" not in st.session_state:
    st.session_state.plagiarism = "Waiting..."
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "listening" not in st.session_state:
    st.session_state.listening = False

# ----------------------------
# Audio Callback
# ----------------------------
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

# ----------------------------
# Background Worker
# ----------------------------
def worker():
    buffer = []
    sr = 16000
    while st.session_state.listening:
        try:
            data = audio_q.get(timeout=1)
            buffer.extend(data[:, 0].tolist())  # mono channel

            if len(buffer) > sr * 3:  # process ~3 sec chunks
                audio = np.array(buffer, dtype=np.float32)
                buffer = []

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    sf.write(tmpfile.name, audio, sr)
                    fname = tmpfile.name

                try:
                    result = whisper_model.transcribe(fname)
                    text = result["text"].strip()
                    if text:
                        st.session_state.transcribed += " " + text
                        plag_msg, conf = plagiarism_confidence(text)
                        st.session_state.plagiarism = plag_msg
                        st.session_state.confidence = conf
                except Exception as e:
                    st.session_state.plagiarism = f"❌ Error: {e}"

                os.remove(fname)
        except queue.Empty:
            continue

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎙️ Real-Time Verbal AI Plagiarism Checker (No WebRTC)")
st.markdown("Speak into your microphone. The app will transcribe speech, check for plagiarism, and display a **confidence score**.")

# Start/Stop Buttons
col1, col2 = st.columns(2)
with col1:
    if not st.session_state.listening:
        if st.button("▶️ Start Recording"):
            st.session_state.listening = True
            threading.Thread(target=worker, daemon=True).start()
            sd.InputStream(callback=audio_callback, channels=1, samplerate=16000).start()
with col2:
    if st.session_state.listening:
        if st.button("⏹️ Stop Recording"):
            st.session_state.listening = False

# Display transcription
st.subheader("📝 Live Transcription")
st.write(st.session_state.transcribed if st.session_state.transcribed else "No speech yet...")

# Display plagiarism detection
st.subheader("📊 Plagiarism / AI Detection Result")
st.write(st.session_state.plagiarism)

# Confidence score
st.metric("Confidence Score", f"{st.session_state.confidence:.2f}%")
