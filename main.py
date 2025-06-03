import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
from scipy.ndimage import uniform_filter1d
import tkinter as tk
from tkinter import filedialog
import soundfile as sf

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

    # Focus on narrower beta band around 16 Hz (14-18 Hz) based on research
    beta_band = (14, 20)
    window_starts = range(0, len(y) - window_length + 1, hop_length)
    envelopes = [get_amplitude_envelope(y[start:start + window_length]) for start in window_starts]

    beta_energies = []
    for envelope in envelopes:
        freqs, spectrum = compute_modulation_spectrum(envelope, sr)
        beta_energy = get_band_energy(freqs, spectrum, beta_band)
        beta_energies.append(beta_energy)

    smoothed = uniform_filter1d(beta_energies, size=5)
    window_centers = np.array([(start + window_length // 2) / sr for start in window_starts])
    
    # Calculate percentage of windows above threshold
    threshold = np.percentile(beta_energies, 40)  # Adaptive threshold
    windows_above_threshold = np.sum(smoothed > threshold)
    percentage_above = (windows_above_threshold / len(smoothed)) * 100
    
    verdict = "Likely beneficial for ADHD focus" if percentage_above >= 80 else \
             "Possibly beneficial" if percentage_above >= 50 else \
             "Unlikely to sustain attention"
    
    return window_centers, smoothed, beta_energies, percentage_above, verdict

class PlotViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Beta Modulation Analysis")
        self.current_plot = 0
        self.plots_data = []

        # Create main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create navigation frame
        self.nav_frame = tk.Frame(self.main_frame)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Create navigation buttons
        self.prev_button = tk.Button(self.nav_frame, text="Previous", command=self.prev_plot)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = tk.Button(self.nav_frame, text="Next", command=self.next_plot)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Create label for file name
        self.file_label = tk.Label(self.nav_frame, text="")
        self.file_label.pack(side=tk.TOP, pady=5)

        # Bind keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.prev_plot())
        self.root.bind('<Right>', lambda e: self.next_plot())

        self.figure = None
        self.canvas = None

    def load_files(self):
        filepaths = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
        )

        if not filepaths:
            self.root.destroy()
            return

        for filepath in filepaths:
            print(f"Analyzing {filepath}...")
            times, smoothed_beta, raw_beta, percentage, verdict = analyze_beta_modulation(filepath)
            threshold = np.percentile(raw_beta, 60)
            self.plots_data.append({
                'filepath': filepath,
                'times': times,
                'smoothed_beta': smoothed_beta,
                'threshold': threshold,
                'percentage': percentage,
                'verdict': verdict
            })

        self.show_plot()

    def show_plot(self):
        if self.figure is not None:
            plt.close(self.figure)

        if not self.plots_data:
            return

        data = self.plots_data[self.current_plot]

        self.figure = plt.Figure(figsize=(12, 4))
        ax = self.figure.add_subplot(111)
        ax.plot(data['times'], data['smoothed_beta'], label='Smoothed Beta Power (12–30 Hz)')
        ax.axhline(data['threshold'], color='red', linestyle='--', label='75th Percentile Threshold')
        # Set background color based on verdict
        bg_color = '#E8F5E9' if data['verdict'].startswith('Likely') else \
                  '#FFF3E0' if data['verdict'].startswith('Possibly') else \
                  '#FFEBEE'
        
        ax.set_title(f"Beta (14-20 Hz) Modulation Analysis\n{data['filepath']}\n{data['verdict']} ({data['percentage']:.1f}% above threshold)")
        ax.set_facecolor(bg_color)
        self.figure.patch.set_facecolor('white')
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Beta Power (a.u.)")
        ax.legend()
        ax.grid(True)

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Update file label
        filename = data['filepath'].split('/')[-1]
        self.file_label.config(text=f"File {self.current_plot + 1} of {len(self.plots_data)}: {filename}")

        # Update button states
        self.prev_button.config(state=tk.NORMAL if self.current_plot > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_plot < len(self.plots_data) - 1 else tk.DISABLED)

    def prev_plot(self):
        if self.current_plot > 0:
            self.current_plot -= 1
            self.show_plot()

    def next_plot(self):
        if self.current_plot < len(self.plots_data) - 1:
            self.current_plot += 1
            self.show_plot()

def run_analysis_ui():
    root = tk.Tk()
    root.geometry("1000x600")
    viewer = PlotViewer(root)
    viewer.load_files()
    root.mainloop()

# Run the UI
if __name__ == "__main__":
    run_analysis_ui()