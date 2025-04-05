import os
import essentia.standard as es
import numpy as np
from visualize import plot_waveform
import json

try:
    with open("/app/current_settings.json", "r") as f:
        CURRENT_SETTINGS = json.load(f)
except FileNotFoundError:
    CURRENT_SETTINGS = {}
    print("⚠️ current_settings.json not found, continuing without current settings.")

DATA_DIR = "/app/data"
OUTPUT_DIR = "/app/output"

TRACK_TYPE_MAP = {
    "bombo": "kick",
    "redo": "snare",
    "overmic": "overhead",
    "bass": "bass",
    "guitar": "guitar",
    "tonpiso": "tom",
    "ton": "tom",
    "hihat": "hihat",
    "carlos": "tenor",
    "pastor": "tenor",
    "director": "tenor",
    "wendy": "contra_alto",
    "milena": "contra_alto",
    "anita": "contra_alto",
    "accordion": "accordion",
    "piano": "piano",
}

TARGET_LUFS_BY_TYPE = {
    "soprano": -18.0,
    "tenor": -18.0,
    "contralto": -19.0,
    "director": -19.5,
    "pastor": -20.0,
    "guitar": -18.0,
    "piano": -18.0,
    "accordion": -18.0,
    "bass": -16.0,
    "kick": -14.0,
    "snare": -15.0,
    "toms": -15.0,
    "hihat": -18.0,
    "overhead": -16.0,
    "default": -18.0,
}


def mono_to_stereo(audio):
    return np.column_stack((audio, audio))


def deviation_to_db(value):
    """
    Converts band energy deviation to dB adjustment.
    A rough scale of 60 maps a deviation of 0.1 to +/- 6 dB.
    """
    scale = 60  # fine tune if necesary
    return round(value * scale, 1)


def detect_problem_frequencies(audio, sample_rate=44100):
    # frame and window the audio
    frame_size = 2048
    hop_size = 1024

    w = es.Windowing(type="hann")
    fft = es.FFT()  # outputs complex
    spectrum_mag = es.Spectrum()  # magnitude spectrum

    frame_gen = es.FrameGenerator(
        audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True
    )

    avg_spectrum = np.zeros(frame_size // 2 + 1)
    for frame in frame_gen:
        spec = spectrum_mag(w(frame))
        avg_spectrum += np.array(spec)

    avg_spectrum /= len(audio) / hop_size

    bin_hz = sample_rate / frame_size

    def freq_range_to_buns(start_hz, end_hz):
        return int(start_hz / bin_hz), int(end_hz / bin_hz)

    suggestions = []

    # Problem zones and labels
    zones = {"Muddiness": (150, 250), "Boxiness": (300, 500), "Harshness": (2000, 4000)}

    for label, (start_hz, end_hz) in zones.items():
        start_bin, end_bin = freq_range_to_buns(start_hz, end_hz)
        zone_energy = np.mean(avg_spectrum[start_bin:end_bin])
        global_avg = np.mean(avg_spectrum)

        if zone_energy > global_avg * 1.3:
            # the zone is 30% higher than average
            peak_bin = np.argmax(avg_spectrum[start_bin:end_bin]) + start_bin
            peak_freq = round(peak_bin * bin_hz)
            suggestions.append(f"Cut ~{peak_freq} Hz ({label.lower()} detected)")
    return suggestions


def band_energy_ratio_avg(
    audio, low_freq, high_freq, sample_rate=44100, frame_size=1024, hop_size=512
):
    """
    Calculate the average band energy ratio over all frames in the audio.

    Args:
        audio (numpy.ndarray): The input mono audio signal.
        low_freq (float): Lower bound of the frequency band.
        high_freq (float): Upper bound of the frequency band.
        sample_rate (int): Sample rate of the audio.
        frame_size (int): Frame size for windowing and FFT.
        hop_size (int): Hop size between frames.

    Returns:
        float: Average band energy ratio for the specified frequency range.
    """

    # Essentia algorithms
    window_algo = es.Windowing(type="hann")
    spectrum_algo = es.Spectrum(size=frame_size)

    # Frequency resolution (bin width)
    freq_resolution = sample_rate / frame_size

    # Get bin indices for the frequency band
    low_bin = int(low_freq / freq_resolution)
    high_bin = int(high_freq / freq_resolution)

    # Accumulators
    ratios = []

    # Process frames
    for start in range(0, len(audio) - frame_size, hop_size):
        frame = audio[start : start + frame_size]

        # Apply window and get the spectrum
        windowed_frame = window_algo(frame)

        frame = audio[start : start + frame_size]

        # Ensure even length
        if len(frame) % 2 != 0:
            frame = frame[:-1]
        spectrum = spectrum_algo(windowed_frame)

        # Energy in the target band
        band_energy = np.sum(spectrum[low_bin:high_bin] ** 2)

        # Total energy in the full spectrum
        total_energy = np.sum(spectrum**2)

        if total_energy == 0:
            ratio = 0.0
        else:
            ratio = band_energy / total_energy

        ratios.append(ratio)

    # Average ratio over all frames
    if len(ratios) == 0:
        return 0.0

    avg_ratio = np.mean(ratios)

    return avg_ratio


def generate_recommendations(
    audio, rms, centroid, ber_low, ber_mid, ber_high, track_type
):
    """Generate EQ and compression suggestions based on analysis and track type."""
    eq_suggestions = []
    compression_suggestions = {}

    # Basic EQ suggestions based on energy ratios
    if ber_low < 0.3:
        eq_suggestions.append("Boost low end (20Hz - 250Hz).")
    if ber_mid < 0.3:
        eq_suggestions.append("Boost mids (250Hz - 4kHz).")
    if ber_high > 0.5:
        eq_suggestions.append("Reduce harshness in highs (4kHz - 20kHz).")

    # Track-specific EQ logic
    if track_type == "kick":
        if ber_low < 0.5:
            eq_suggestions.append("Boost sub-bass (50-80Hz) for more punch.")
        if ber_high > 0.4:
            eq_suggestions.append("Cut highs above 5kHz to reduce click.")

    elif track_type == "snare":
        if ber_mid < 0.4:
            eq_suggestions.append("Boost mids around 1-3kHz for snap.")
        if ber_high > 0.6:
            eq_suggestions.append("Reduce highs above 8kHz to tame harshness.")

    elif track_type == "vocal":
        if centroid < 1500:
            eq_suggestions.append("Boost presence around 3kHz for vocal clarity.")
        if ber_high > 0.5:
            eq_suggestions.append("Use de-esser or cut 5-8kHz to tame sibilance.")

    elif track_type == "bass":
        if ber_low < 0.4:
            eq_suggestions.append("Boost 60-100Hz for more bass body.")
        if ber_mid > 0.5:
            eq_suggestions.append("Cut 300Hz muddiness.")

    elif track_type == "tom":
        if ber_low < 0.4:
            eq_suggestions.append("Boost 80Hz - 100Hz for body and punch.")
        if ber_mid > 0.5:
            eq_suggestions.append("Cut 250Hz - 400Hz to reduce boxiness.")
        if ber_high < 0.3:
            eq_suggestions.append("Boost 5kHz - 7kHz for attack and clarity.")

    elif track_type == "hihat":
        if ber_low > 0.2:
            eq_suggestions.append("Cut 200Hz - 400Hz to reduce bleed from other drums.")
        if ber_high < 0.5:
            eq_suggestions.append("Boost 8kHz - 12kHz for shimmer and brightness.")
        if ber_high > 0.7:
            eq_suggestions.append("Consider taming harshness above 12kHz.")

    elif track_type == "tenor":
        if ber_low < 0.3:
            eq_suggestions.append("Boost 150Hz - 250Hz for warmth and fullness.")
        if ber_mid < 0.4:
            eq_suggestions.append("Boost 2kHz - 5kHz for vocal presence and clarity.")
        if ber_high > 0.5:
            eq_suggestions.append(
                "Apply de-esser or reduce 5kHz - 8kHz to tame sibilance."
            )

    elif track_type == "contra_alto":
        if ber_low < 0.4:
            eq_suggestions.append("Boost 130Hz - 200Hz to add body and warmth.")
        if ber_mid < 0.4:
            eq_suggestions.append("Boost 1kHz - 3kHz for clarity and presence.")
        if ber_high > 0.6:
            eq_suggestions.append("Reduce 5kHz - 7kHz to control harshness.")

    elif track_type == "alto":
        if ber_low < 0.3:
            eq_suggestions.append("Boost 150Hz - 250Hz for warmth and body.")
        if ber_mid < 0.4:
            eq_suggestions.append("Boost 2kHz - 4kHz for articulation and clarity.")
        if ber_high > 0.5:
            eq_suggestions.append(
                "Reduce 5kHz - 8kHz or apply de-esser for sibilance control."
            )

    elif track_type == "accordion":
        if ber_low < 0.4:
            eq_suggestions.append("Boost 100Hz - 250Hz for warmth and bass body.")
        if ber_mid > 0.5:
            eq_suggestions.append("Cut 300Hz - 500Hz to reduce boxiness and muddiness.")
        if ber_mid < 0.3:
            eq_suggestions.append("Boost 1kHz - 4kHz for clarity and presence.")
        if ber_high > 0.6:
            eq_suggestions.append(
                "Reduce 5kHz - 8kHz to tame harshness or brittleness."
            )
        elif ber_high < 0.3:
            eq_suggestions.append("Boost 5kHz - 8kHz to add air and brightness.")
    elif track_type == "piano":
        if ber_low < 0.3:
            eq_suggestions.append("Boost 50Hz - 100Hz for low-end fullness.")
        if ber_mid > 0.5:
            eq_suggestions.append("Cut 200Hz - 400Hz to reduce muddiness or boxiness.")
        if ber_mid < 0.4:
            eq_suggestions.append("Boost 2kHz - 5kHz for attack and clarity.")
        if ber_high < 0.3:
            eq_suggestions.append("Boost 8kHz - 12kHz for air and brightness.")
        if ber_high > 0.6:
            eq_suggestions.append("Cut above 10kHz to reduce harshness or brittleness.")

    # Compression logic (basic crest factor)
    crest_factor = np.max(np.abs(audio)) / rms
    if crest_factor > 6:
        compression_suggestions = {
            "ratio": "4:1",
            "attack": "10 ms",
            "release": "80 ms",
            "threshold": "-8 dB",
        }
    else:
        compression_suggestions = {
            "ratio": "2:1",
            "attack": "5 ms",
            "release": "50 ms",
            "threshold": "-4 dB",
        }

    return eq_suggestions, compression_suggestions


def detect_noise_floor(audio, threshold_silence_db=60, threshold_noise_db=-50):
    # Calculate per-frame RMS value
    frame_size = 1024
    hop_size = 512
    rms_list = []

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i : i + frame_size]
        rms = np.sqrt(np.mean(np.square(frame)))
        if rms > 0:
            rms_db = 20 * np.log10(rms)
            rms_list.append(rms_db)

    if not rms_list:
        return ["⚠️ Could not compute noise floor (empty or zero audio)"]

    avg_rms = np.mean(rms_list)
    min_rms = np.min(rms_list)

    suggestions = []

    if avg_rms < threshold_silence_db:
        suggestions.append("⚠️ Track may be mostly silent or disconnected")

    if min_rms > threshold_noise_db:
        suggestions.append("⚠️ High noise floor detected - check mic, cable or gain")

    return suggestions


def suggest_gain_adjustment(lufs_value, target_lufs=-18.0):
    """
    Suggest gain adjustment in dB to match the target LUFS level.
    """
    delta = target_lufs - lufs_value
    if abs(delta) < 0.5:
        return f"Gain is within aceptable range (Current: {lufs_value:.1f} LUFS)"
    action = "Increase" if delta > 0 else "Reduce"
    return f"{action} gain by {abs(delta):.1f} dB to reach target loudness({target_lufs} LUFS)"


def analyze_track(file_path, output_dir, track_type):
    """Analyze a single audio file and output recommendations."""

    # Load audio as mono
    audio = es.MonoLoader(filename=file_path)()
    stereo_audio = mono_to_stereo(audio)

    # Check for Noise Issues
    noise_issues = detect_noise_floor(audio)

    # Extract features
    integrated_lufs = float(es.LoudnessEBUR128()(stereo_audio)[0])
    rms = es.RMS()(audio)

    target_lufs = TARGET_LUFS_BY_TYPE.get(
        track_type.lower(), TARGET_LUFS_BY_TYPE["default"]
    )
    gain_suggestions = suggest_gain_adjustment(integrated_lufs, target_lufs)

    # Band energy ratios (low, mid, high)
    ber_low = band_energy_ratio_avg(
        audio, low_freq=20, high_freq=250, sample_rate=44100
    )
    ber_mid = band_energy_ratio_avg(
        audio, low_freq=250, high_freq=4000, sample_rate=44100
    )
    ber_high = band_energy_ratio_avg(
        audio, low_freq=4000, high_freq=20000, sample_rate=44100
    )

    # Only use a frame, not full audio, and ensure it's even-sized
    frame_size = 1024
    frame = audio[:frame_size]

    if len(frame) % 2 != 0:
        frame = frame[:-1]

    window = es.Windowing(type="hann")
    spectrum_algo = es.Spectrum()

    windowed = window(frame)
    spectrum = spectrum_algo(windowed)
    centroid = es.Centroid()(spectrum)

    problem_freqs = detect_problem_frequencies(audio)

    # Plot waveform
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    waveform_plot = os.path.join(output_dir, f"{base_name}_waveform.png")
    plot_waveform(audio, waveform_plot)

    # Generate recommendations based on track type
    eq_suggestions, compression_suggestions = generate_recommendations(
        audio, rms, centroid, ber_low, ber_mid, ber_high, track_type
    )

    ber = es.BandEnergyRatio(inputSize=1024)
    low_ratio = ber(audio(20, 250))
    mid_ratio = ber(audio(250, 4000))
    high_ratio = ber(audio(4000, 12000))

    total = low_ratio + mid_ratio + high_ratio
    deviation = {
        "low": low_ratio / total,
        "mid": mid_ratio / total,
        "high": high_ratio / total,
    }

    track_name = os.path.basename(file_path)
    current = CURRENT_SETTINGS.get(track_name, {})

    # Handle the EQ adjustments based on what is given in the json
    current_eq = current.get("eq", {"low": 0, "mid": 0, "high": 0})
    eq_final = []
    for band in ["low", "mid", "high"]:
        suggested_db = deviation_to_db(deviation[band])
        current_db = current_eq.get(band, 0)
        delta_db = round(suggested_db - current_db, 1)

        if abs(delta_db) >= 1:
            action = "Boost" if delta_db > 0 else "Cut"
            eq_final.append(
                f"{action} {band} frequencies by {abs(delta_db)} dB"
                f"(current: {current_db:+} dB, Target: {suggested_db:+} dB)"
            )
    eq_suggestions = eq_final

    # Generate report
    report = {
        "file": file_path,
        "track_type": track_type,
        "rms": rms,
        "loudness": loudness,
        "centroid": centroid,
        "band_energy_low": ber_low,
        "band_energy_mid": ber_mid,
        "band_energy_high": ber_high,
        "eq_suggestions": eq_suggestions,
        "compression_suggestions": compression_suggestions,
        "waveform_plot": waveform_plot,
        "problem_frequencies": problem_freqs,
        "noise_floor_issues": noise_issues,
        "gain_suggestions": gain_suggestions,
    }

    # Output report
    print(f"\n🎛️ Analysis Report for {base_name} ({track_type})")
    for key, value in report.items():
        print(f"{key}: {value}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".wav")]

    if not files:
        print("⚠️ No WAV files found in /data")
        return

    print(f"🔍 Found {len(files)} WAV files to analyze...\n")

    for file_name in files:
        file_path = os.path.join(DATA_DIR, file_name)

        # Determine track type
        track_name = os.path.splitext(file_name)[0].lower()
        print(f"🔍 Looking up track type for: {track_name}")
        track_type = TRACK_TYPE_MAP.get(track_name, "unknown")

        if track_type == "unknown":
            print(
                f"⚠️ Track type for '{file_name}' not found in TRACK_TYPE_MAP. Skipping..."
            )
            continue

        analyze_track(file_path, OUTPUT_DIR, track_type)


if __name__ == "__main__":
    main()
