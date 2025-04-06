import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.audio_dataset import AudioDataset
from models.cnn_denoiser import CNNDenoiser

# --- Config ---
EPOCHS = 10
BATCH_SIZE = 8
DATA_DIR = "data/processed/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset & Loader ---
dataset = AudioDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model, Loss, Optimizer ---
model = CNNDenoiser().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training Loop ---
for epoch in range(EPOCHS):
    total_loss = 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)

        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

# --- Save the model ---
torch.save(model.state_dict(), "cnn_denoiser.pth")
