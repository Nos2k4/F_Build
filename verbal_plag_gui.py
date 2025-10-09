# verbal_plagiarism_ctk.py

import os
import threading
import time
import sounddevice as sd
import soundfile as sf
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import customtkinter as ctk

# ------------------- CustomTkinter Settings -------------------
ctk.set_appearance_mode("Dark")  # "Light" or "Dark"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# ------------------- Load Models -------------------
print("Loading Whisper (speech-to-text)...")
stt_model = whisper.load_model("base")

print("Loading AI text detector (RoBERTa)...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
detector_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

# ------------------- Audio Functions -------------------
def record_chunk(filename="chunk.wav", duration=10, fs=16000):
    output_box.insert(ctk.END, f"\n🎙️ Recording {duration} seconds audio...\n")
    output_box.see(ctk.END)
    root.update()
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, recording, fs)
    return filename

def transcribe_audio(audio_path):
    result = stt_model.transcribe(audio_path)
    return result["text"]

def detect_ai_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = detector_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    human_prob, ai_prob = probs[0].tolist()
    return human_prob, ai_prob

def verbal_plagiarism_check(audio_path):
    output_box.insert(ctk.END, f"\nProcessing file: {audio_path}\n")
    output_box.see(ctk.END)
    root.update()

    # Transcribe
    transcribed_text = transcribe_audio(audio_path)
    output_box.insert(ctk.END, "📝 Transcribed Text:\n")
    output_box.insert(ctk.END, transcribed_text + "\n")
    output_box.see(ctk.END)
    root.update()

    # AI detection
    human_prob, ai_prob = detect_ai_text(transcribed_text)

    if ai_prob > 0.5:
        color = "#FF5555"  # red
        verdict = "⚠️ Plagiarism Suspected: Text seems AI-generated!"
    else:
        color = "#55FF55"  # green
        verdict = "✅ Text seems Human-written."

    output_box.insert(ctk.END, f"\n🔎 AI Detection Results:\n")
    output_box.insert(ctk.END, f"   Human-written probability: {human_prob:.4f}\n")
    output_box.insert(ctk.END, f"   AI-generated probability:  {ai_prob:.4f}\n")
    output_box.insert(ctk.END, f"{verdict}\n")
    output_box.tag_add("verdict", "end-2l", "end")
    output_box.tag_config("verdict", foreground=color)
    output_box.insert(ctk.END, f"\n📊 Model Confidence: {max(human_prob, ai_prob)*100:.2f}%\n")
    output_box.see(ctk.END)
    root.update()

# ------------------- GUI Functions -------------------
def run_check_manual():
    audio_path = file_path_var.get()
    if not audio_path:
        ctk.messagebox.showwarning("No file", "Please select an audio file first.")
        return
    output_box.delete(1.0, ctk.END)
    verbal_plagiarism_check(audio_path)

# ------------------- Real-Time Monitoring -------------------
monitoring = False

def start_monitoring():
    global monitoring
    monitoring = True
    output_box.delete(1.0, ctk.END)
    output_box.insert(ctk.END, "🚀 Starting Real-Time Monitoring Mode...\n")
    threading.Thread(target=monitor_audio_loop, daemon=True).start()

def stop_monitoring():
    global monitoring
    monitoring = False
    output_box.insert(ctk.END, "\n🛑 Monitoring stopped.\n")
    output_box.see(ctk.END)

def monitor_audio_loop(chunk_duration=10):
    while monitoring:
        audio_file = "temp_chunk.wav"
        record_chunk(filename=audio_file, duration=chunk_duration)
        verbal_plagiarism_check(audio_file)
        os.remove(audio_file)
        output_box.insert(ctk.END, f"\nWaiting for next {chunk_duration}s chunk...\n")
        output_box.see(ctk.END)
        root.update()

# ------------------- File Browse -------------------
def browse_file():
    filename = ctk.filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=(("Audio Files", "*.wav *.mp3 *.opus *.m4a"), ("All Files", "*.*"))
    )
    if filename:
        file_path_var.set(filename)

# ------------------- GUI -------------------
root = ctk.CTk()
root.title("Verbal Plagiarism Detector")
root.geometry("750x600")

file_path_var = ctk.StringVar()

# Manual Mode Frame
frame_manual = ctk.CTkFrame(root)
frame_manual.pack(pady=10, padx=10, fill="x")
ctk.CTkLabel(frame_manual, text="Manual Mode:", font=("Arial", 14, "bold")).pack(anchor="w", padx=5)
manual_controls = ctk.CTkFrame(frame_manual)
manual_controls.pack(pady=5, padx=5, fill="x")
ctk.CTkEntry(manual_controls, textvariable=file_path_var, width=400).pack(side="left", padx=5)
ctk.CTkButton(manual_controls, text="Browse", command=browse_file).pack(side="left", padx=5)
ctk.CTkButton(manual_controls, text="Run Check", command=run_check_manual, fg_color="#1E90FF").pack(side="left", padx=5)

# Real-Time Monitoring Frame
frame_auto = ctk.CTkFrame(root)
frame_auto.pack(pady=10, padx=10, fill="x")
ctk.CTkLabel(frame_auto, text="Real-Time Monitoring Mode:", font=("Arial", 14, "bold")).pack(anchor="w", padx=5)
auto_controls = ctk.CTkFrame(frame_auto)
auto_controls.pack(pady=5, padx=5)
ctk.CTkButton(auto_controls, text="Start Monitoring", command=start_monitoring, fg_color="#32CD32").pack(side="left", padx=5)
ctk.CTkButton(auto_controls, text="Stop Monitoring", command=stop_monitoring, fg_color="#FF6347").pack(side="left", padx=5)

# Output Box
output_box = ctk.CTkTextbox(root, width=720, height=350)
output_box.pack(padx=10, pady=10)
output_box.insert(ctk.END, "📌 Output will appear here...\n")

root.mainloop()
