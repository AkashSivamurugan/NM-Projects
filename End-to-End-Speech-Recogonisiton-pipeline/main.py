import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd

# === 1. Feature Extraction (MFCC or Wav2Vec2 directly handles raw waveform) ===
def extract_mfcc(signal, sample_rate=16000, n_mfcc=13):
    # This function is no longer needed if you're using Wav2Vec2, as it processes raw audio
    pass

# === 2. Live Audio Recording ===
def record_audio(duration=3, sample_rate=16000):
    print(f"Recording {duration} seconds of audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = torch.from_numpy(audio.squeeze())
    return audio

# === 3. Load Pre-trained Wav2Vec2 Model ===
def load_pretrained_model():
    # Load the pre-trained Wav2Vec2 model and processor from Hugging Face
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model

# === 4. Inference (using pre-trained model) ===
def infer(processor, model, audio_signal, sample_rate=16000):
    # Process the raw audio signal into the format expected by the model
    input_values = processor(audio_signal.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate).input_values
    
    # Make the prediction using the pre-trained model
    model.eval()
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode the predicted logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# === 5. Main ===
if __name__ == "__main__":
    # Load pre-trained Wav2Vec2 model
    processor, model = load_pretrained_model()

    # Record new audio
    signal = record_audio(duration=3, sample_rate=16000)

    # Perform inference
    predicted_text = infer(processor, model, signal, sample_rate=16000)
    print(f"Predicted Text: {predicted_text}")
