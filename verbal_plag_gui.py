# verbal_plagiarism_gui.py

import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox

# ----------- Load Models (once at startup) -----------
print("Loading Whisper (speech-to-text)...")
stt_model = whisper.load_model("base")   # change to "small"/"medium" for higher accuracy

print("Loading AI text detector (RoBERTa)...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
detector_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")


# ----------- Functions -----------
def transcribe_audio(audio_path):
    """Convert speech to text using Whisper"""
    result = stt_model.transcribe(audio_path)
    return result["text"]


def detect_ai_text(text):
    """Classify text as Human or AI-generated"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = detector_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    human_prob, ai_prob = probs[0].tolist()
    return human_prob, ai_prob


def run_check():
    audio_path = file_path_var.get()
    if not audio_path:
        messagebox.showwarning("No file", "Please select an audio file first.")
        return

    output_box.delete(1.0, tk.END)  # clear output

    output_box.insert(tk.END, f"Processing file: {audio_path}\n\n")

    # Step 1: Transcribe
    transcribed_text = transcribe_audio(audio_path)
    output_box.insert(tk.END, "📝 Transcribed Text:\n")
    output_box.insert(tk.END, transcribed_text + "\n\n")

    # Step 2: AI detection
    human_prob, ai_prob = detect_ai_text(transcribed_text)

    output_box.insert(tk.END, "🔎 AI Detection Results:\n")
    output_box.insert(tk.END, f"   Human-written probability: {human_prob:.4f}\n")
    output_box.insert(tk.END, f"   AI-generated probability:  {ai_prob:.4f}\n\n")

    if ai_prob > 0.5:
        output_box.insert(tk.END, "⚠️ Plagiarism Suspected: Text seems AI-generated!\n")
    else:
        output_box.insert(tk.END, "✅ Text seems Human-written.\n")

    accuracy = max(human_prob, ai_prob) * 100
    output_box.insert(tk.END, f"\n📊 Model Confidence (accuracy): {accuracy:.2f}%\n")


def browse_file():
    filename = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=(("Audio Files", "*.wav *.mp3 *.opus *.m4a"), ("All Files", "*.*"))
    )
    if filename:
        file_path_var.set(filename)


# ----------- GUI -----------
root = tk.Tk()
root.title("Verbal Plagiarism Checker")
root.geometry("650x500")

file_path_var = tk.StringVar()

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Entry(frame, textvariable=file_path_var, width=50).pack(side=tk.LEFT, padx=5)
tk.Button(frame, text="Browse", command=browse_file).pack(side=tk.LEFT)
tk.Button(frame, text="Run Check", command=run_check, bg="lightblue").pack(side=tk.LEFT, padx=5)

output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20)
output_box.pack(padx=10, pady=10)

root.mainloop()
