# real_time_filter.py
import sounddevice as sd
import torch
import librosa
import numpy as np
from models.cnn_denoiser import CNNDenoiser

def real_time_filter(model_path, duration=5, sr=16000, n_mfcc=40):
    print("🎙️ Recording... Speak now.")
    noisy = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
    sd.wait()

    noisy = noisy.flatten()

    # Extract MFCC
    noisy_mfcc = librosa.feature.mfcc(noisy, sr=sr, n_mfcc=n_mfcc)
    noisy_tensor = torch.tensor(noisy_mfcc).unsqueeze(0).unsqueeze(0).float()

    # Load model
    model = CNNDenoiser()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        clean_tensor = model(noisy_tensor)
    clean_mfcc = clean_tensor.squeeze().numpy()

    # Convert MFCC to audio
    clean_audio = librosa.feature.inverse.mfcc_to_audio(clean_mfcc, sr=sr)

    print("🔊 Playing filtered audio...")
    sd.play(clean_audio, sr)
    sd.wait()

if __name__ == "__main__":
    real_time_filter("cnn_denoiser.pth")
