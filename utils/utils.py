# utils.py
import numpy as np
from pesq import pesq
from scipy.io import wavfile

def compute_snr(clean, noisy):
    noise = noisy - clean
    snr = 10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2) + 1e-10)
    return snr

def compute_pesq(ref_path, deg_path):
    sr_ref, ref = wavfile.read(ref_path)
    sr_deg, deg = wavfile.read(deg_path)

    assert sr_ref == sr_deg
    ref = ref.astype(np.float32)
    deg = deg.astype(np.float32)

    try:
        score = pesq(sr_ref, ref, deg, 'wb')  # Wideband mode
        return score
    except Exception as e:
        print("PESQ error:", e)
        return -1.0
