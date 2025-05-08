import os
import torch
import torchaudio
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from jiwer import wer
import sounddevice as sd
from scipy.signal import resample
import argparse

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Train the model instead of running inference')
args = parser.parse_args()

# === 1. Signal Processing ===
def pre_emphasis(signal, coeff=0.97):
    return torch.cat((signal[:1], signal[1:] - coeff * signal[:-1]))

# === 2. Feature Extraction ===
from torchaudio.transforms import MFCC, MelSpectrogram, AmplitudeToDB

def extract_mfcc(signal, sample_rate=16000, n_mfcc=13):
    mfcc = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)(signal)
    return mfcc.transpose(0,1)

def extract_log_mel(signal, sample_rate=16000, n_mels=40):
    mel_spec = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(signal)
    log_mel = AmplitudeToDB()(mel_spec)
    return log_mel.transpose(0,1)

# === 3. Live Audio Recording ===
def record_audio(duration=3, sample_rate=16000):
    print(f"Recording {duration} seconds of audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    audio = torch.from_numpy(audio.squeeze())
    return audio

# === 4. Acoustic Model ===
class AcousticModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.fc(unpacked)
        return logits.log_softmax(dim=-1)

# === 5. Greedy Decoding ===
def greedy_decode(log_probs, blank=0):
    tokens = torch.argmax(log_probs, -1)
    seq, prev = [], blank
    for t in tokens:
        if t != prev and t != blank:
            seq.append(t.item())
        prev = t
    return seq

# === 6. Training Function ===
def train(model, loader, optimizer, criterion, device):
    print("Training started...")
    model.train()
    total_loss = 0
    for feats, flens, labels, llens in loader:
        print(f"Processing batch...")  # Debugging line
        feats, labels = torch.stack(list(feats)), torch.cat(list(labels))
        flens = torch.tensor(list(flens))
        llens = torch.tensor(list(llens))
        feats, labels = feats.to(device), labels.to(device)

        # Forward pass
        out = model(feats, flens)
        out = out.permute(1, 0, 2)  # (T, N, C)
        loss = criterion(out, labels, flens, llens)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Print loss every 10 iterations
        if total_loss % 10 == 0:
            print(f"Loss: {loss.item():.4f}")
    
    print("Training finished...")
    return total_loss / len(loader)

# === 7. Dummy Dataset ===
class DummyDataset(Dataset):
    def __init__(self, vocab, sample_rate=16000):
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.data = [
            ('hello', record_audio(1, sample_rate)),  # Record and say 'hello'
            ('yes', record_audio(1, sample_rate)),    # Say 'yes'
        ]
        print(f"Dataset loaded with {len(self.data)} samples.")  # Debugging line

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_text, signal = self.data[idx]
        signal = pre_emphasis(signal)
        feat = extract_log_mel(signal.unsqueeze(0), sample_rate=self.sample_rate)
        label = torch.tensor([self.vocab[c] for c in label_text], dtype=torch.long)
        print(f"Processed sample {idx}: {label_text}")  # Debugging line
        return feat.squeeze(0), feat.shape[1], label, len(label)

# === 8. Main ===
if __name__ == "__main__":
    vocab = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz '")}
    vocab['<blank>'] = 0
    inv_vocab = {i: c for c, i in vocab.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AcousticModel(input_dim=40, hidden_dim=256, output_dim=len(vocab)).to(device)

    if args.train:
        dataset = DummyDataset(vocab)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda b: list(zip(*b)))
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        for epoch in range(5):
            print(f"Epoch {epoch + 1}:")
            loss = train(model, loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

        torch.save(model.state_dict(), "acoustic_model.pt")
        print("Model saved successfully.")
    else:
        if os.path.exists('acoustic_model.pt'):
            model.load_state_dict(torch.load('acoustic_model.pt', map_location=device))
            print("Loaded pre-trained model.")
        else:
            print("Model not found. Please train first.")
            exit()

        signal = record_audio(duration=3, sample_rate=16000)
        signal = pre_emphasis(signal)

        features = extract_log_mel(signal.unsqueeze(0), sample_rate=16000).unsqueeze(0)
        lengths = torch.tensor([features.shape[1]])

        model.eval()
        with torch.no_grad():
            features = features.to(device)
            output = model(features, lengths)
            seq = greedy_decode(output[0, :lengths[0]].cpu())
            result = ''.join(inv_vocab[t] for t in seq)
            print(f"Predicted: {result}")
