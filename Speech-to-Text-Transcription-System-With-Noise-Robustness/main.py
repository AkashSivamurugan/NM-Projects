import sounddevice as sd
import numpy as np
import queue
import tkinter as tk
from tkinter import ttk
import threading
import whisper
import tempfile
import os
import wave
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Whisper model
model = whisper.load_model("base")  # use "tiny" if base is slow

# Audio setup
q = queue.Queue()
recording = False
audio_data = []

# Transcription data
full_transcript = []

# Tkinter GUI setup
root = tk.Tk()
root.title("Real-Time Speech Transcriber")
root.geometry("600x500")

lbl_transcript = tk.Label(root, text="", wraplength=550, justify="left", font=("Arial", 12))
lbl_transcript.pack(pady=10)

fig, ax = plt.subplots(figsize=(4, 1))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
line, = ax.plot([], [])
ax.set_ylim(0, 1)
ax.set_xlim(0, 100)
volume_history = []

# Functions
def audio_callback(indata, frames, time, status):
    if recording:
        q.put(indata.copy())
        audio_data.append(indata.copy())
        volume = np.linalg.norm(indata) * 10
        volume_history.append(volume)
        if len(volume_history) > 100:
            volume_history.pop(0)
        update_volume_meter()
    
def update_volume_meter():
    line.set_ydata(volume_history)
    line.set_xdata(range(len(volume_history)))
    ax.relim()
    ax.autoscale_view()
    canvas.draw()

def record_audio():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        while recording:
            sd.sleep(100)

def transcription_worker():
    while recording:
        try:
            data = q.get(timeout=1)
        except queue.Empty:
            continue

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            with wave.open(temp_wav.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((data * 32767).astype(np.int16).tobytes())

            try:
                result = model.transcribe(temp_wav.name, fp16=False)
                text = result['text'].strip()
                if text:
                    full_transcript.append(text)
                    label_text = lbl_transcript.cget("text") + " " + text
                    lbl_transcript.after(0, lambda t=label_text: lbl_transcript.config(text=t))
            except Exception as e:
                print("Transcription error:", e)

            os.remove(temp_wav.name)

def start_recording():
    global recording, audio_data, full_transcript, volume_history
    lbl_transcript.config(text="")
    audio_data = []
    full_transcript = []
    volume_history = []
    recording = True
    threading.Thread(target=record_audio, daemon=True).start()
    threading.Thread(target=transcription_worker, daemon=True).start()

def stop_recording():
    global recording
    recording = False

def transcribe_and_save():
    stop_recording()

    # Save full audio to WAV
    if not audio_data:
        print("No audio recorded.")
        return

    combined_audio = np.concatenate(audio_data)
    filename = "recorded_audio.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes((combined_audio * 32767).astype(np.int16).tobytes())

    # Final transcription (optional second-pass)
    result = model.transcribe(filename, fp16=False)
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(result['text'])

    print("Saved transcription.txt")

# Buttons
ttk.Button(root, text="Start Recording", command=start_recording).pack(pady=5)
ttk.Button(root, text="Stop Recording", command=stop_recording).pack(pady=5)
ttk.Button(root, text="Transcribe & Save Full", command=transcribe_and_save).pack(pady=5)

root.mainloop()
