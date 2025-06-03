import numpy as np
import soundfile as sf
from scipy.signal import hilbert, butter, filtfilt, resample
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def analyze_modulation(file_path,
                       chunk_duration_sec=30,
                       overlap_duration_sec=1,
                       resample_rate=1000,
                       lowpass_cutoff=32,
                       fft_window_sec=2.0,
                       fft_hop_sec=1.0,
                       beta_band=(14, 20),  # Expanded beta range
                       percentile_thresh=75):  # 75% of windows must exceed this
    
    # Audio preprocessing
    info = sf.info(file_path)
    sr = info.samplerate
    total_frames = info.frames
    chunk_samples = int(chunk_duration_sec * sr)
    overlap_samples = int(overlap_duration_sec * sr)
    hop_samples = chunk_samples - overlap_samples

    # Lowpass filter for envelope
    b, a = butter(N=4, Wn=lowpass_cutoff / (0.5 * sr), btype='low')

    # Process audio in chunks
    envelope_chunks = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_samples, total_frames)
        y, _ = sf.read(file_path, start=start, stop=end)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y
        
        # Extract and filter envelope
        envelope = np.abs(hilbert(y))
        envelope_filtered = filtfilt(b, a, envelope)
        
        # Resample envelope
        duration = (end - start) / sr
        num_samples_resampled = int(duration * resample_rate)
        envelope_resampled = resample(envelope_filtered, num_samples_resampled)
        
        if start != 0:
            envelope_resampled = envelope_resampled[int(overlap_duration_sec * resample_rate):]
        envelope_chunks.append(envelope_resampled)
        start += hop_samples

    envelope_full = np.concatenate(envelope_chunks)
    envelope_sr = resample_rate

    # Compute modulation spectrogram
    window_size = int(fft_window_sec * envelope_sr)
    hop_size = int(fft_hop_sec * envelope_sr)
    num_windows = (len(envelope_full) - window_size) // hop_size + 1

    modulation_spectrogram = []
    for i in range(num_windows):
        win_start = i * hop_size
        win_end = win_start + window_size
        window = envelope_full[win_start:win_end] * np.hanning(win_end - win_start)  # Hanning window
        spectrum = np.abs(fft(window))
        freqs = fftfreq(len(window), d=1.0 / envelope_sr)
        modulation_spectrogram.append(spectrum[freqs > 0])

    modulation_spectrogram = np.array(modulation_spectrogram).T
    freqs = freqs[freqs > 0]
    times = np.arange(num_windows) * fft_hop_sec

    # Beta band analysis (percentile-based)
    beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
    beta_power = modulation_spectrogram[beta_mask, :].mean(axis=0)  # Mean power per window
    threshold = np.percentile(beta_power, percentile_thresh)
    high_beta_windows = beta_power > threshold
    beta_ratio = np.sum(high_beta_windows) / len(beta_power) * 100

    # Diagnostic output
    print(f"\nAnalysis for: {file_path}")
    print(f"Beta power stats: Mean={np.mean(beta_power):.2f}, Threshold ({percentile_thresh}th %ile)={threshold:.2f}")
    print(f"Beta modulation coverage: {beta_ratio:.1f}% of song")

    verdict = (
        "Likely beneficial for ADHD focus" if beta_ratio >= 80 else
        "Possibly beneficial" if beta_ratio >= 50 else
        "Unlikely to sustain attention"
    )

    return {
        "filepath": file_path,
        "modulation_spectrogram": modulation_spectrogram,
        "freqs": freqs,
        "times": times,
        "beta_power": beta_power,
        "beta_ratio": beta_ratio,
        "verdict": verdict,
        "threshold": threshold,
        "beta_band": beta_band
    }


class PlotViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("ADHD Focus Music Analyzer")
        self.current_plot = 0
        self.plots_data = []

        # UI Setup
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.nav_frame = tk.Frame(self.main_frame)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.prev_button = tk.Button(self.nav_frame, text="Previous", command=self.prev_plot)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.next_button = tk.Button(self.nav_frame, text="Next", command=self.next_plot)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.file_label = tk.Label(self.nav_frame, text="", font=('Arial', 10))
        self.file_label.pack(side=tk.TOP, pady=5)

        # Key bindings
        self.root.bind('<Left>', lambda e: self.prev_plot())
        self.root.bind('<Right>', lambda e: self.next_plot())

        self.figure = None
        self.canvas = None

    def load_files(self):
        filepaths = filedialog.askopenfilenames(
            title="Select Music Files",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
        )

        if not filepaths:
            return

        for filepath in filepaths:
            print(f"\nProcessing: {filepath}")
            result = analyze_modulation(filepath)
            self.plots_data.append(result)

        self.show_plot()

    def show_plot(self):
        if self.figure is not None:
            plt.close(self.figure)

        if not self.plots_data:
            return

        data = self.plots_data[self.current_plot]

        # Create plot
        self.figure = plt.Figure(figsize=(12, 6))
        ax = self.figure.add_subplot(111)
        
        # Plot spectrogram (normalized for visualization)
        spectrogram_db = 10 * np.log10(data['modulation_spectrogram'] + 1e-12)
        img = ax.imshow(spectrogram_db, aspect='auto', origin='lower',
                        extent=[data['times'][0], data['times'][-1], 
                                data['freqs'][0], data['freqs'][-1]],
                        cmap='viridis')
        
        # Highlight beta band
        ax.axhspan(data['beta_band'][0], data['beta_band'][1], 
                  color='red', alpha=0.15, label='Target Beta Band')
        
        # Mark high-beta windows
        high_beta_times = data['times'][data['beta_power'] > data['threshold']]
        for t in high_beta_times:
            ax.axvline(t, color='lime', alpha=0.3, linewidth=0.5)
        
        # Formatting
        self.figure.colorbar(img, ax=ax, label='Power (dB)')
        ax.set_title(
            f"Modulation Spectrogram\n"
            f"{data['filepath'].split('/')[-1]}\n"
            f"Verdict: {data['verdict']} ({data['beta_ratio']:.1f}% in {data['beta_band'][0]}-{data['beta_band'][1]} Hz)"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Modulation Frequency (Hz)")
        ax.legend()
        ax.grid(True)

        # Update UI
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Update file info
        filename = data['filepath'].split('/')[-1]
        self.file_label.config(
            text=f"File {self.current_plot + 1} of {len(self.plots_data)}: {filename}\n"
                 f"Threshold: {data['threshold']:.2f} (power) | "
                 f"Mean Beta: {np.mean(data['beta_power']):.2f}"
        )

        # Button states
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
    root.geometry("1100x700")
    viewer = PlotViewer(root)
    
    # Start with file dialog
    root.after(100, viewer.load_files)
    root.mainloop()


if __name__ == "__main__":
    run_analysis_ui()