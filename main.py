import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.ndimage import uniform_filter1d
import tkinter as tk
from tkinter import filedialog

# Function to load and normalize audio
def load_audio(filepath):
    y, sr = sf.read(filepath)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y / np.max(np.abs(y))
    return y, sr

# Extract amplitude envelope using Hilbert transform
def get_amplitude_envelope(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

# Compute modulation spectrum
def compute_modulation_spectrum(envelope, sr):
    n = len(envelope)
    timestep = 1.0 / sr
    freq = fftfreq(n, d=timestep)
    spectrum = np.abs(fft(envelope))
    return freq[:n // 2], spectrum[:n // 2]

# Compute energy in a frequency band
def get_band_energy(freqs, spectrum, band):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.sum(spectrum[mask])

# Main analysis function
def analyze_beta_modulation(audio_path):
    y, sr = load_audio(audio_path)
    window_length_sec = 2.0
    hop_length_sec = 1.0
    window_length = int(window_length_sec * sr)
    hop_length = int(hop_length_sec * sr)

    beta_band = (12, 30)
    window_starts = range(0, len(y) - window_length + 1, hop_length)
    envelopes = [get_amplitude_envelope(y[start:start + window_length]) for start in window_starts]

    beta_energies = []
    for envelope in envelopes:
        freqs, spectrum = compute_modulation_spectrum(envelope, sr)
        beta_energy = get_band_energy(freqs, spectrum, beta_band)
        beta_energies.append(beta_energy)

    smoothed = uniform_filter1d(beta_energies, size=5)
    window_centers = np.array([(start + window_length // 2) / sr for start in window_starts])
    return window_centers, smoothed, beta_energies

# Tkinter UI for file selection
def run_analysis_ui():
    root = tk.Tk()
    root.withdraw()
    filepaths = filedialog.askopenfilenames(title="Select Audio Files", filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])

    for filepath in filepaths:
        print(f"Analyzing {filepath}...")
        times, smoothed_beta, raw_beta = analyze_beta_modulation(filepath)
        threshold = np.percentile(raw_beta, 75)

        plt.figure(figsize=(12, 4))
        plt.plot(times, smoothed_beta, label='Smoothed Beta Power (12–30 Hz)')
        plt.axhline(threshold, color='red', linestyle='--', label='75th Percentile Threshold')
        plt.title(f"Beta Modulation Power Over Time\n{filepath}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Beta Power (a.u.)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Run the UI
run_analysis_ui()