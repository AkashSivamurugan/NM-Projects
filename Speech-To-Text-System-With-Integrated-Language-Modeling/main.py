import streamlit as st
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# Load Wav2Vec2 model and processor
@st.cache_resource
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()
    return processor, model

processor, model = load_model()

st.title("🎙️ Real-time Speech-to-Text with Language Modeling")
st.markdown("Record audio from your microphone and get real-time transcription using Wav2Vec2.")

# Audio Processor to handle real-time audio
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame):
        # Get the audio from the frame and convert to numpy array
        audio_data = frame.numpy().flatten()

        # Convert stereo to mono if stereo input
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Convert to torch tensor and resample if needed
        data = torch.tensor(audio_data, dtype=torch.float32)

        # Resample to 16kHz if needed
        samplerate = 16000
        if frame.sample_rate != samplerate:
            data = torchaudio.functional.resample(data, orig_freq=frame.sample_rate, new_freq=samplerate)

        # Prepare input for the model
        input_tensor = data.unsqueeze(0)  # Shape: [1, time]

        # Process the input tensor for the model
        inputs = processor(input_tensor, sampling_rate=samplerate, return_tensors="pt", padding=True)

        # Ensure that the processor returns correctly formatted tensors
        if 'input_values' in inputs:
            input_values = inputs['input_values']

            # Perform inference
            with torch.no_grad():
                logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)

            return transcription[0]
        else:
            return "Error: Invalid input format"

# Initialize the WebRTC streamer
webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

