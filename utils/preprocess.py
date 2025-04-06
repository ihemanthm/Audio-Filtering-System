import librosa
import numpy as np
import soundfile as sf
import os
import random
from glob import glob

def load_audio(file_path, sr=16000):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y

def normalize_audio(y):
    return y / np.max(np.abs(y))

def mix_audio(speech, noise, snr_db):
    """
    Mixes speech with noise at given SNR (Signal-to-Noise Ratio)
    """
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    mixed = speech + noise
    return normalize_audio(mixed)

def extract_mfcc(y, sr=16000, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

def process_pair(speech_path, noise_path, output_dir, snr_db=5):
    speech = load_audio(speech_path)
    noise = load_audio(noise_path)

    # Trim noise to speech length
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[:len(speech)]

    mixed = mix_audio(speech, noise, snr_db)

    base_name = os.path.basename(speech_path).replace('.mp3', '.wav')
    sf.write(os.path.join(output_dir, f'clean_{base_name}'), speech, 16000)
    sf.write(os.path.join(output_dir, f'noisy_{base_name}'), mixed, 16000)

    return speech, mixed

def preprocess_dataset(speech_folder, noise_folder, output_dir, sample_count=100):
    speech_files = glob(os.path.join(speech_folder, '*.mp3'))
    noise_files = glob(os.path.join(noise_folder, '**/*.wav'), recursive=True)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(sample_count):
        s_path = random.choice(speech_files)
        n_path = random.choice(noise_files)
        process_pair(s_path, n_path, output_dir)

if __name__ == "__main__":
    preprocess_dataset(
        speech_folder='data/commonvoice/clips/',
        noise_folder='data/musan/noise/',
        output_dir='data/processed/',
        sample_count=200
    )
