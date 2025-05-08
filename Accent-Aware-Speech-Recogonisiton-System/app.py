import streamlit as st
import pyaudio
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

# Constants
ACCENT_CLASSES = ['English', 'French', 'German', 'Spanish']  # Modify based on your dataset
SAMPLING_RATE = 16000
MFCC_FEATURES = 13
SPEAKER_EMBEDDING_SIZE = 10  # Modify as per your speaker model

# Function to extract MFCC features from audio
def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLING_RATE, n_mfcc=MFCC_FEATURES)
    return np.mean(mfcc.T, axis=0)

# Define the LSTM-based model for accent classification
class AccentRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AccentRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = hn[-1, :, :]
        x = self.fc(x)
        return x

# Function to load a pre-trained model (Here we use a dummy model for example purposes)
def load_model():
    model = AccentRecognitionModel(input_size=MFCC_FEATURES, hidden_size=64, num_classes=len(ACCENT_CLASSES))
    model.eval()  # Set model to evaluation mode
    return model

# Dummy speaker embedding function (replace with a real model)
def dummy_speaker_embedding(y):
    return np.random.rand(SPEAKER_EMBEDDING_SIZE)

# Audio stream capture and prediction function
def predict_accent_from_audio(model):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE, input=True, frames_per_buffer=1024)

    st.write("Listening... Please speak into the microphone.")
    
    while True:
        data = np.frombuffer(stream.read(1024), dtype=np.int16)
        y = data.astype(np.float32)  # Convert audio data to float32

        # Extract MFCC features
        mfcc_features = extract_mfcc(y)
        speaker_embedding = dummy_speaker_embedding(y)  # Use the real speaker model for actual use

        # Prepare input for model: Combine MFCC and speaker embedding
        input_data = np.concatenate([mfcc_features, speaker_embedding])

        # Convert input data to torch tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Predict accent using the model
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_accent = ACCENT_CLASSES[torch.argmax(prediction).item()]

        st.write(f"Predicted Accent: {predicted_accent}")
        break

# Streamlit UI
def main():
    st.title("Accent-Aware Speech Recognition System")

    st.markdown("""
    This app uses a deep learning model to predict the accent of the speaker in real-time.
    It listens to your speech, processes it, and predicts your accent.
    """)

    st.sidebar.title("Controls")
    start_button = st.sidebar.button("Start Listening")

    if start_button:
        model = load_model()  # Load the model
        st.write("Model loaded. Ready to listen for speech.")
        predict_accent_from_audio(model)

if __name__ == '__main__':
    main()
