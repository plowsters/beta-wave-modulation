import numpy as np
import soundfile as sf
import json
import os
from datetime import datetime
from scipy.interpolate import interp1d
import uuid

# Configuration
output_dir = "reference_bank"
os.makedirs(output_dir, exist_ok=True)
registry_path = os.path.join(output_dir, "reference_registry.json")
SAMPLE_RATE = 44100
NUM_SAMPLES = 40
FM_RANGE = np.arange(12.0, 20.5, 0.5)  # Half steps from 12 to 20 Hz

# Randomization ranges
def generate_random_duration(min_sec=60, max_sec=180):
    return float(np.random.uniform(min_sec, max_sec))

def generate_random_depth(min_depth=0.5, max_depth=1.0):
    return float(np.random.uniform(min_depth, max_depth))

def generate_random_fc(min_fc=100, max_fc=1000):
    return float(np.random.uniform(min_fc, max_fc))

# AM Tone Generator with randomized carrier
def generate_am_tone(f_c, f_m, duration, sr=SAMPLE_RATE, depth=1.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    modulator = (1 + depth * np.sin(2 * np.pi * f_m * t)) / (1 + depth)
    carrier = np.sin(2 * np.pi * f_c * t)
    return modulator * carrier

# Smoothly shifting modulation generator with randomized carrier
def generate_shifting_modulated_noise(f_c, duration, sr=SAMPLE_RATE, depth=1.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    times, freq_pairs = generate_random_shift_sequence(duration)

    f_mod = np.array([start for start, _ in freq_pairs] + [freq_pairs[-1][1]])
    f_interp = interp1d(times, f_mod, kind='quadratic', fill_value="extrapolate")
    f_mod_t = f_interp(t)

    phase = 2 * np.pi * np.cumsum(f_mod_t) / sr
    modulator = (1 + depth * np.sin(phase)) / (1 + depth)
    carrier = np.sin(2 * np.pi * f_c * t)
    return modulator * carrier, times, freq_pairs

def generate_random_shift_sequence(total_duration, min_segment=5.0, max_segment=30.0):
    time_points = [0.0]
    while time_points[-1] < total_duration:
        next_t = time_points[-1] + np.random.uniform(min_segment, max_segment)
        if next_t >= total_duration:
            break
        time_points.append(next_t)
    time_points.append(total_duration)

    freq_pairs = []
    for _ in range(len(time_points) - 1):
        start_f = np.random.uniform(12.0, 20.0)
        end_f = np.random.uniform(12.0, 20.0)
        freq_pairs.append((start_f, end_f))

    return time_points, freq_pairs

# Quality check for clipping, silence, or invalid values
def validate_audio(signal):
    if np.isnan(signal).any() or np.isinf(signal).any():
        return False, "Signal contains NaNs or Infs"
    peak = np.max(np.abs(signal))
    if peak > 1.0:
        return False, f"Clipping detected: peak = {peak}"
    if np.allclose(signal, 0.0, atol=1e-6):
        return False, "Signal is silent or all zeros"
    return True, "Valid signal"

# Load or create registry
if os.path.exists(registry_path):
    with open(registry_path, "r") as f:
        registry = json.load(f)
        if isinstance(registry, dict):
            registry = list(registry.values())
else:
    registry = []

# Batch generation
counter = 0
timestamp = datetime.now().isoformat()

for f_m in FM_RANGE:
    if counter >= NUM_SAMPLES:
        break

    # === AM TONE ===
    duration = generate_random_duration()
    depth = generate_random_depth()
    f_c = generate_random_fc()
    tone = generate_am_tone(f_c=f_c, f_m=f_m, duration=duration, depth=depth)
    valid, message = validate_audio(tone)
    if not valid:
        print(f"Skipping AM tone {f_m}Hz (reason: {message})")
        continue

    uid = uuid.uuid4().hex[:8]
    base_name = f"am_tone_{f_m:.1f}Hz_{uid}"
    audio_path = os.path.join(output_dir, f"{base_name}.flac")
    sf.write(audio_path, tone, SAMPLE_RATE)
    registry.append({
        "id": base_name,
        "type": "am_tone",
        "carrier_freq": f_c,
        "mod_freq": f_m,
        "depth": depth,
        "duration": duration,
        "sample_rate": SAMPLE_RATE,
        "timestamp": timestamp,
        "audio_file": audio_path
    })
    counter += 1

    if counter >= NUM_SAMPLES:
        break

    # === SHIFTING MODULATED NOISE ===
    duration = generate_random_duration()
    depth = generate_random_depth()
    f_c = generate_random_fc()
    noise, times, freq_pairs = generate_shifting_modulated_noise(f_c=f_c, duration=duration, depth=depth)
    valid, message = validate_audio(noise)
    if not valid:
        print(f"Skipping shifting noise (reason: {message})")
        continue

    uid = uuid.uuid4().hex[:8]
    base_name = f"shifting_noise_{uid}"
    audio_path = os.path.join(output_dir, f"{base_name}.flac")
    sf.write(audio_path, noise, SAMPLE_RATE)
    registry.append({
        "id": base_name,
        "type": "shifting_modulated_noise",
        "carrier_freq": f_c,
        "depth": depth,
        "duration": duration,
        "sample_rate": SAMPLE_RATE,
        "timestamp": timestamp,
        "audio_file": audio_path,
        "modulation_segments": [
            {
                "start_time": times[i],
                "end_time": times[i + 1],
                "start_freq": freq_pairs[i][0],
                "end_freq": freq_pairs[i][1]
            } for i in range(len(freq_pairs))
        ]
    })
    counter += 1

# Save updated registry
with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

print(f"{counter} new audio files generated and validated, added to reference_registry.json.")