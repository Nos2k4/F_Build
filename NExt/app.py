from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# -----------------------------
# Load models once at startup
# -----------------------------
print("Loading models...")
whisper_model = whisper.load_model("tiny")  # faster; use "base" if GPU available
detector_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
print("Models loaded successfully!")

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI()

# Allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing, allow all. Lock down later!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    # Save uploaded audio file
    with open("temp.wav", "wb") as f:
        f.write(await file.read())

    # Transcribe with Whisper
    result = whisper_model.transcribe("temp.wav")
    text = result["text"]

    # Run RoBERTa AI detector
    inputs = detector_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = detector_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    ai_prob = probs[0][1].item()

    return {
        "transcription": text,
        "ai_prob": ai_prob,
        "human_prob": 1 - ai_prob
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
