# gradio_app.py
import gradio as gr
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from models.cnn_denoiser import CNNDenoiser

# Load your trained model
model = CNNDenoiser()
model.load_state_dict(torch.load("cnn_denoiser.pth", map_location=torch.device('cpu')))
model.eval()

def denoise_audio(audio):
    input_file, _ = audio  # audio is a tuple (file_obj, sample_rate)
    noisy_waveform, sr = torchaudio.load(input_file)
    noisy_np = noisy_waveform.squeeze(0).numpy()

    # Convert to spectrogram or MFCC
    mfcc = librosa.feature.mfcc(y=noisy_np, sr=sr, n_mfcc=40)
    mfcc = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float()  # (1, 1, n_mfcc, time)

    with torch.no_grad():
        cleaned_mfcc = model(mfcc)

    # Inverse MFCC → waveform (approximate)
    cleaned_mfcc = cleaned_mfcc.squeeze().numpy()
    cleaned_audio = librosa.feature.inverse.mfcc_to_audio(cleaned_mfcc, sr=sr)

    # Save output as .wav
    sf.write("cleaned_output.wav", cleaned_audio, sr)

    return "cleaned_output.wav"

# Gradio UI
interface = gr.Interface(
    fn=denoise_audio,
    inputs=gr.Audio(source="upload", type="filepath", label="Upload Noisy Audio (.wav)"),
    outputs=gr.Audio(type="filepath", label="Denoised Audio Output"),
    title="🎧 AI Audio-Filtering System",
    description="Upload a noisy speech audio file (.wav). This tool will reduce background noise and output a cleaner version using a trained CNN model."
)

if __name__ == "__main__":
    interface.launch()
