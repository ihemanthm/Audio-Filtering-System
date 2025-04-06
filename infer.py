# infer.py
import torch
import librosa
import soundfile as sf
import numpy as np
from models.cnn_denoiser import CNNDenoiser
from utils.utils import compute_snr, compute_pesq

def denoise_audio_file(model_path, noisy_path, output_path, sr=16000, n_mfcc=40):
    # Load model
    model = CNNDenoiser()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Load audio
    noisy, _ = librosa.load(noisy_path, sr=sr)

    # Extract MFCC
    noisy_mfcc = librosa.feature.mfcc(noisy, sr=sr, n_mfcc=n_mfcc)
    noisy_tensor = torch.tensor(noisy_mfcc).unsqueeze(0).unsqueeze(0).float()

    # Predict clean MFCC
    with torch.no_grad():
        clean_tensor = model(noisy_tensor)
    clean_mfcc = clean_tensor.squeeze().numpy()

    # Convert MFCC to waveform
    clean_audio = librosa.feature.inverse.mfcc_to_audio(clean_mfcc, sr=sr)

    # Save result
    sf.write(output_path, clean_audio, sr)
    print(f"[✓] Denoised audio saved to: {output_path}")

    return clean_audio, noisy, sr

if __name__ == "__main__":
    clean_audio, noisy_audio, sr = denoise_audio_file(
        model_path="cnn_denoiser.pth",
        noisy_path="data/test_noisy.wav",
        output_path="data/cleaned_output.wav"
    )

    # Evaluate
    snr = compute_snr(clean_audio, noisy_audio)
    pesq = compute_pesq("data/test_noisy.wav", "data/cleaned_output.wav")

    print(f"🔊 SNR: {snr:.2f} dB")
    print(f"🗣️ PESQ: {pesq:.2f}")
