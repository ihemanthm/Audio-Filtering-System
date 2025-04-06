# 🎧 Audio-Filtering System

An AI-powered audio filtering system that isolates the **primary speaker’s voice** and suppresses background noise in real time. Built using **Python**, **PyTorch**, and **Gradio**, this tool supports both audio files and real-time input streams.

---

## 📌 Features

- 🎙️ Isolates the primary speaker's voice
- 🔇 Suppresses background noise (e.g., traffic, music, chatter)
- 🧠 Uses CNN-based deep learning models
- 📊 Supports evaluation using SNR, PESQ, and MOS
- 🌐 Includes an interactive Gradio web UI

---

## 🧠 Technologies Used

| Technology     | Purpose                          |
|----------------|----------------------------------|
| Python         | Programming language             |
| PyTorch        | Deep learning model training     |
| Torchaudio     | Efficient audio processing       |
| Librosa        | Audio feature extraction (MFCCs) |
| SoundFile      | WAV file reading/writing         |
| Gradio         | Web UI for demo and deployment   |

---


## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Harshad712/audio-filtering-system.git
cd audio-filtering-system
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Download Datasets

You'll need:

- **Clean speech**: [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- **Noise samples**: [MUSAN Dataset](https://www.openslr.org/17/)

Place files in a `data/` directory.

---

### 4️⃣ Preprocess Data

Create noisy audio by mixing clean speech with noise:

```bash
python utils/preprocess.py
```

---

### 5️⃣ Train the Model

```bash
python train.py
```

- Trains a CNN denoiser on synthetic noisy audio
- Saves model weights to `cnn_denoiser.pth`

---

### 6️⃣ Denoise New Audio (Inference)

```bash
python infer.py
```

- Input: `test_noisy.wav`
- Output: `cleaned_output.wav`

---

### 7️⃣ Evaluate Performance

```python
from utils import calculate_snr, calculate_pesq

snr = calculate_snr("data/clean.wav", "output/denoised.wav")
pesq = calculate_pesq("data/clean.wav", "output/denoised.wav")

print("SNR:", snr)
print("PESQ:", pesq)
```

---

### 8️⃣ Launch the Gradio UI

```bash
python gradio_app.py
```

- Access at `http://127.0.0.1:7860`
- Upload `.wav` file → receive denoised output

---

## 🌐 Deployment

You can Access this project in hugging face at `https://huggingface.co/spaces/Harshad712/audio-filtering-system`

---

## 📊 Evaluation Metrics

| Metric | Description                              |
|--------|------------------------------------------|
| **SNR**  | Signal-to-Noise Ratio (in dB)             |
| **PESQ** | Perceptual Evaluation of Speech Quality  |
| **MOS**  | Mean Opinion Score (simulated)           |

These help assess how well the model enhances speech quality.

---

## 📦 `requirements.txt`

```txt
torch
torchaudio
librosa
soundfile
numpy
scipy
gradio
```

---


## 🔧 Future Enhancements

- [ ] Add RNN/Transformer models
- [ ] Real-time mic audio denoising
- [ ] Quantized/lightweight models for mobile
- [ ] Online demo hosted on Hugging Face

---

## 🤝 Contributing

Pull requests and improvements are welcome!

```bash
# Fork it
# Create your branch: git checkout -b feature-xyz
# Commit changes: git commit -m 'add feature'
# Push branch: git push origin feature-xyz
# Submit a pull request
```

---

## 👨‍💻 Author

Developed by Harshad Kokkinti  
Connect on [GitHub](https://github.com/Harshad712)

---

## 📚 References

- [MUSAN Dataset](https://www.openslr.org/17/)
- [Common Voice](https://commonvoice.mozilla.org/)
- [Gradio Documentation](https://www.gradio.app/)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

---
