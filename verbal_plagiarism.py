# verbal_plagiarism.py

import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------- STEP 1: Load Models -----------
print("Loading Whisper (speech-to-text)...")
stt_model = whisper.load_model("base")   # You can use "small" or "medium" for higher accuracy

print("Loading AI text detector (RoBERTa)...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
detector_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")


# ----------- STEP 2: Speech to Text -----------
def transcribe_audio(audio_path):
    """Convert speech to text using Whisper"""
    result = stt_model.transcribe(audio_path)
    return result["text"]


# ----------- STEP 3: AI Text Detection -----------
def detect_ai_text(text):
    """Classify text as Human or AI-generated"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = detector_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    human_prob, ai_prob = probs[0].tolist()
    return human_prob, ai_prob


# ----------- STEP 4: Main Pipeline -----------
def verbal_plagiarism_check(audio_path):
    print(f"\nProcessing file: {audio_path}")
    
    # Step 1: Transcribe speech
    transcribed_text = transcribe_audio(audio_path)
    print("\n📝 Transcribed Text:\n", transcribed_text)

    # Step 2: Detect AI-generated probability
    human_prob, ai_prob = detect_ai_text(transcribed_text)
    
    print("\n🔎 AI Detection Results:")
    print(f"   Human-written probability: {human_prob:.4f}")
    print(f"   AI-generated probability:  {ai_prob:.4f}")

    if ai_prob > 0.5:
        print("\n⚠️ Plagiarism Suspected: Text seems AI-generated!")
    else:
        print("\n✅ Text seems Human-written.")

    accuracy = max(human_prob, ai_prob) * 100
    print(f"\n📊 Model Confidence (accuracy): {accuracy:.2f}%")

# ----------- RUN TEST -----------
if __name__ == "__main__":
    # Replace with your test audio file
    test_audio = "new.opus"
    verbal_plagiarism_check(test_audio)
