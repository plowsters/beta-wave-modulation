import numpy as np
import soundfile as sf
from scipy.signal import hilbert, butter, filtfilt, resample
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Parameters
resample_rate = 1000
lowpass_cutoff = 32
fft_window_sec = 2.0
fft_hop_sec = 1.0
beta_band = (14, 20)

# Reference tone
reference_path = "/home/archbug/projects/beta-wave-modulation/amplitudeModulation16Hz.flac"

def compute_reference_threshold():
    y, sr = sf.read(reference_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y / np.max(np.abs(y))
    envelope = np.abs(hilbert(y))
    b, a = butter(N=4, Wn=lowpass_cutoff / (0.5 * sr), btype='low')
    envelope_filtered = filtfilt(b, a, envelope)
    envelope_resampled = resample(envelope_filtered, int(len(envelope_filtered) * resample_rate / sr))

    window_size = int(fft_window_sec * resample_rate)
    hop_size = int(fft_hop_sec * resample_rate)
    beta_powers_db = []

    for i in range(0, len(envelope_resampled) - window_size + 1, hop_size):
        window = envelope_resampled[i:i + window_size] * np.hanning(window_size)
        spectrum = np.abs(fft(window))
        freqs = fftfreq(len(window), d=1.0 / resample_rate)
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        spectrum = spectrum[pos_mask]
        total_power = np.sum(spectrum[(freqs >= 1) & (freqs <= 32)])
        beta_power = np.sum(spectrum[(freqs >= beta_band[0]) & (freqs <= beta_band[1])])
        ratio = beta_power / total_power if total_power != 0 else 1e-12
        beta_db = 10 * np.log10(ratio)
        beta_powers_db.append(beta_db)

    return np.percentile(beta_powers_db, 20)

reference_threshold = compute_reference_threshold()

def analyze_beta_modulation(filepath):
    y, sr = sf.read(filepath)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y / np.max(np.abs(y))
    envelope = np.abs(hilbert(y))
    b, a = butter(N=4, Wn=lowpass_cutoff / (0.5 * sr), btype='low')
    envelope_filtered = filtfilt(b, a, envelope)
    envelope_resampled = resample(envelope_filtered, int(len(envelope_filtered) * resample_rate / sr))

    window_size = int(fft_window_sec * resample_rate)
    hop_size = int(fft_hop_sec * resample_rate)
    times = []
    beta_powers_db = []

    for i in range(0, len(envelope_resampled) - window_size + 1, hop_size):
        window = envelope_resampled[i:i + window_size] * np.hanning(window_size)
        spectrum = np.abs(fft(window))
        freqs = fftfreq(len(window), d=1.0 / resample_rate)
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        spectrum = spectrum[pos_mask]
        total_power = np.sum(spectrum[(freqs >= 1) & (freqs <= 32)])
        beta_power = np.sum(spectrum[(freqs >= beta_band[0]) & (freqs <= beta_band[1])])
        ratio = beta_power / total_power if total_power != 0 else 1e-12
        beta_db = 10 * np.log10(ratio)
        beta_powers_db.append(beta_db)
        times.append(i / resample_rate)

    percentage_above = np.sum(np.array(beta_powers_db) > reference_threshold) / len(beta_powers_db) * 100
    verdict = "Likely beneficial for ADHD focus" if percentage_above >= 80 else \
              "Possibly beneficial" if percentage_above >= 50 else \
              "Unlikely to sustain attention"
    return times, beta_powers_db, percentage_above, verdict

class PlotViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Beta Modulation Analysis (Log dB Threshold Calibrated)")
        self.current_plot = 0
        self.plots_data = []

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.nav_frame = tk.Frame(self.main_frame)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.prev_button = tk.Button(self.nav_frame, text="Previous", command=self.prev_plot)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = tk.Button(self.nav_frame, text="Next", command=self.next_plot)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.file_label = tk.Label(self.nav_frame, text="")
        self.file_label.pack(side=tk.TOP, pady=5)

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
            times, beta_dB, percentage, verdict = analyze_beta_modulation(filepath)
            self.plots_data.append({
                'filepath': filepath,
                'times': times,
                'beta_dB': beta_dB,
                'threshold': reference_threshold,
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
        ax.plot(data['times'], data['beta_dB'], label='Beta Modulation Strength (log dB)')
        ax.axhline(data['threshold'], color='red', linestyle='--', label='Reference Threshold')

        bg_color = '#E8F5E9' if data['verdict'].startswith('Likely') else \
                   '#FFF3E0' if data['verdict'].startswith('Possibly') else \
                   '#FFEBEE'

        ax.set_title(f"{data['filepath'].split('/')[-1]}\n{data['verdict']} ({data['percentage']:.1f}% above threshold)")
        ax.set_facecolor(bg_color)
        self.figure.patch.set_facecolor('white')
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Beta Modulation Strength (log dB)")
        ax.legend()
        ax.grid(True)

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        filename = data['filepath'].split('/')[-1]
        self.file_label.config(text=f"File {self.current_plot + 1} of {len(self.plots_data)}: {filename}")
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

run_analysis_ui()