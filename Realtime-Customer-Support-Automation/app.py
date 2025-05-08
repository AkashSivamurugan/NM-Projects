import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import whisper
import queue
import av
import numpy as np

st.set_page_config(page_title="Real-Time Customer Support", layout="centered")

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Set up session state to track conversation
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# Simple bot response logic
def get_bot_response(text):
    if "price" in text.lower():
        return "Our pricing information is available on our website."
    elif "support" in text.lower():
        return "Support is available 24/7 through our chatbot or email."
    elif "refund" in text.lower():
        return "I can help with refunds. Could you provide your order number?"
    return "Thank you! A human agent will follow up if needed."

# Audio processing
audio_queue = queue.Queue()

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
    audio_queue.put(audio)
    return frame

# Streamlit UI
st.title("🎙️ Real-Time Speech-to-Text Customer Support")

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"video": False, "audio": True},
)

# Display conversation
st.subheader("📝 Live Transcript")
chat_area = st.empty()

# Real-time transcription loop
if webrtc_ctx.state.playing:
    st.info("🎧 Listening... Speak into your microphone.")
    buffer = []
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=2)
            buffer.extend(audio_chunk)
            if len(buffer) > 16000 * 5:  # 5 seconds buffer
                audio_array = np.array(buffer)
                result = model.transcribe(audio_array, fp16=False)
                text = result["text"].strip()
                if text:
                    bot_reply = get_bot_response(text)
                    st.session_state.chat_log.append(("User", text))
                    st.session_state.chat_log.append(("Bot", bot_reply))
                buffer = []
        except queue.Empty:
            break

# Display chat log
with chat_area.container():
    for speaker, message in st.session_state.chat_log:
        if speaker == "User":
            st.markdown(f"🗣️ **You:** {message}")
        else:
            st.markdown(f"🤖 **Bot:** {message}")
