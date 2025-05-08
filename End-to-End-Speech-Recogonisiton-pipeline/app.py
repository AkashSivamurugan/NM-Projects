import streamlit as st
import torch
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# === Load Pretrained Model ===
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model

# === Record Audio ===
def record_audio(duration=3, fs=16000):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.squeeze()

# === Inference ===
def transcribe(audio, processor, model, sample_rate=16000):
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# === Streamlit UI ===
def main():
    st.title("🎙️ Live Speech-to-Text Transcription")
    st.write("Click the button below to record your voice and get a text transcription using Wav2Vec2.")

    duration = st.slider("Select recording duration (seconds)", 1, 10, 3)
    if st.button("🔴 Record"):
        audio = record_audio(duration)
        st.success("Recording complete. Transcribing...")

        processor, model = load_model()
        text = transcribe(audio, processor, model)
        st.subheader("📝 Transcribed Text:")
        st.write(text)

if __name__ == "__main__":
    main()
