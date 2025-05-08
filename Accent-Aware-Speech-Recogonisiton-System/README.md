# 🎙️ Accent-Aware Speech Recognition System

This project is a real-time accent-aware speech recognition system that uses PyTorch and Streamlit. It listens to live speech from the user's microphone and predicts the accent (e.g., English, French, German, Spanish) using a deep learning model.

---

## 🔧 Features

- Real-time audio streaming using `pyaudio`
- MFCC-based audio feature extraction using `librosa`
- Deep learning model with LSTM for accent classification
- Dummy speaker adaptation (can be replaced with real embeddings)
- Simple and interactive Streamlit UI

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/accent-recognition
cd accent-recognition
