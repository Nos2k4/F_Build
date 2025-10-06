import gradio as gr
import whisper
from sentence_transformers import SentenceTransformer, util

# Load models
whisper_model = whisper.load_model("base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reference DB
reference_texts = [
    "Artificial intelligence is transforming the world.",
    "Deep learning is a subset of machine learning.",
    "Neural networks can learn patterns in data.",
    "Transformers revolutionized natural language processing.",
]
reference_embeddings = embedding_model.encode(reference_texts, convert_to_tensor=True)

def analyze(audio):
    if audio is None:
        return "No audio", "No text", "0%"

    # Transcribe with Whisper
    result = whisper_model.transcribe(audio)
    text = result["text"].strip()

    if not text:
        return "No audio", "No text", "0%"

    # Compare embeddings
    test_emb = embedding_model.encode([text], convert_to_tensor=True)
    cos_scores = util.cos_sim(test_emb, reference_embeddings)
    max_score = float(cos_scores.max())

    if max_score > 0.65:
        status = "⚠️ Possible plagiarism / AI-generated"
    else:
        status = "✅ Likely original"

    return status, text, f"{max_score*100:.2f}%"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎙️ Real-Time Verbal Plagiarism Checker")

    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak here")
    
    with gr.Row():
        status = gr.Label(label="Detection Result")
        transcription = gr.Textbox(label="Transcription")
        confidence = gr.Label(label="Confidence")

    mic.change(analyze, inputs=mic, outputs=[status, transcription, confidence])

# Run app
if __name__ == "__main__":
    demo.launch(share=True)

