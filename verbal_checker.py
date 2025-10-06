import tkinter as tk
from tkinter import messagebox, scrolledtext
import sounddevice as sd
import numpy as np
import whisper
from sentence_transformers import SentenceTransformer, util
import torch
import scipy.io.wavfile as wav
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Load models
# -----------------------------
print("Loading models... this may take a moment.")

# Whisper ASR
whisper_model = whisper.load_model("base")

# Sentence-BERT (for semantic similarity plagiarism check)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# RoBERTa AI text detector
detector_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base-openai-detector"
)
detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")

print("Models loaded successfully.")

# Reference texts (replace with larger dataset for stronger plagiarism detection)
reference_texts = [
    "Artificial intelligence is transforming the world.",
    "Deep learning is a subset of machine learning.",
    "Neural networks can learn patterns in data.",
    "Transformers revolutionized natural language processing.",
]
reference_embeddings = embedding_model.encode(reference_texts, convert_to_tensor=True)

# -----------------------------
# Helper functions
# -----------------------------
fs = 16000  # sampling rate
duration = 30  # seconds of recording


def record_audio():
    """Record audio from mic and save as input.wav"""
    try:
        messagebox.showinfo("Recording", f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        wav.write("input.wav", fs, (audio * 32767).astype(np.int16))
        messagebox.showinfo("Done", "Audio recorded successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def roberta_classify(text):
    """Run RoBERTa AI detector"""
    inputs = detector_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = detector_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    ai_prob = probs[0][1].item()  # Probability of AI
    return ai_prob


def analyze_audio():
    """Run transcription + plagiarism checks"""
    try:
        # Step 1: Transcribe
        result = whisper_model.transcribe("input.wav")
        text = result["text"].strip()

        if not text:
            output_text.insert(tk.END, "\nNo speech detected.\n")
            return

        # Step 2: RoBERTa AI detection
        ai_prob = roberta_classify(text)
        if ai_prob > 0.5:
            ai_status = f"⚠️ Likely AI-generated (Score: {ai_prob*100:.2f}%)"
        else:
            ai_status = f"✅ Likely human (Score: {(1-ai_prob)*100:.2f}%)"

        # Step 3: Embedding-based plagiarism similarity
        test_emb = embedding_model.encode([text], convert_to_tensor=True)
        cos_scores = util.cos_sim(test_emb, reference_embeddings)
        max_score = float(cos_scores.max())

        if max_score > 0.65:
            sim_status = "⚠️ Possible plagiarism / text reuse"
        else:
            sim_status = "✅ Likely original"

        # Step 4: Display results
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Transcription:\n{text}\n\n")
        output_text.insert(tk.END, f"RoBERTa AI Detector: {ai_status}\n")
        output_text.insert(tk.END, f"Plagiarism (Similarity Check): {sim_status}\n")
        output_text.insert(tk.END, f"Confidence (Similarity): {max_score*100:.2f}%\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# -----------------------------
# GUI Setup
# -----------------------------
root = tk.Tk()
root.title("🎙️ Verbal Plagiarism Checker")
root.geometry("700x500")

label = tk.Label(root, text="Record and Analyze Speech", font=("Arial", 16))
label.pack(pady=10)

btn_record = tk.Button(root, text="🎤 Record", command=record_audio, bg="lightblue", font=("Arial", 12))
btn_record.pack(pady=5)

btn_analyze = tk.Button(root, text="🔎 Analyze", command=analyze_audio, bg="lightgreen", font=("Arial", 12))
btn_analyze.pack(pady=5)

output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20, font=("Arial", 11))
output_text.pack(pady=10)

root.mainloop()
