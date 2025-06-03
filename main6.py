import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Config
BETA_BAND = (14, 20)
REFERENCE_PATH = "/home/archbug/projects/beta-wave-modulation/isochronic16hz.flac"  # Update with the actual reference file path
WINDOW_SEC = 2.0  # Energy calculation window

def bandpass_filter(y, sr, low, high):
    nyq = 0.5 * sr
    b, a = butter(N=4, Wn=[low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, y)

def compute_reference_threshold():
    y, sr = sf.read(REFERENCE_PATH)
    if y.ndim > 1:
        y = y[:, 0]
    y = y / np.max(np.abs(y))
    y_beta = bandpass_filter(y, sr, *BETA_BAND)
    window_size = int(WINDOW_SEC * sr)
    energies = [np.sqrt(np.mean(y_beta[i:i+window_size]**2))
                for i in range(0, len(y_beta)-window_size, window_size)]
    return np.percentile(energies, 10)

def analyze_beta_modulation(filepath, threshold):
    y, sr = sf.read(filepath)
    if y.ndim > 1:
        y = y[:, 0]
    y = y / np.max(np.abs(y))
    y_beta = bandpass_filter(y, sr, *BETA_BAND)
    window_size = int(WINDOW_SEC * sr)
    times = []
    energies = []
    for i in range(0, len(y_beta)-window_size, window_size):
        window = y_beta[i:i+window_size]
        times.append(i / sr)
        energies.append(np.sqrt(np.mean(window**2)))
    energies = np.array(energies)
    active = energies > threshold
    percentage = np.mean(active) * 100
    verdict = "Likely beneficial" if percentage >= 80 else \
              "Possibly beneficial" if percentage >= 50 else \
              "Unlikely to sustain attention"
    return times, energies, percentage, verdict, threshold

class PlotViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Beta Modulation Analysis")
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

        threshold = compute_reference_threshold()
        for filepath in filepaths:
            print(f"Analyzing {filepath}...")
            times, energies, percentage, verdict, _ = analyze_beta_modulation(filepath, threshold)
            self.plots_data.append({
                'filepath': filepath,
                'times': times,
                'energies': energies,
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
        ax.plot(data['times'], data['energies'], label='Beta Band RMS Energy')
        ax.axhline(data['threshold'], color='red', linestyle='--', label='Reference Threshold')
        bg_color = '#E8F5E9' if data['verdict'].startswith('Likely') else \
                   '#FFF3E0' if data['verdict'].startswith('Possibly') else \
                   '#FFEBEE'
        ax.set_title(f"Beta (14–20 Hz) Modulation Analysis\n{data['filepath']}\n{data['verdict']} ({data['percentage']:.1f}% above threshold)")
        ax.set_facecolor(bg_color)
        self.figure.patch.set_facecolor('white')
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Beta Modulation Strength (RMS)")
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