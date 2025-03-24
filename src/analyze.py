import os
import essentia.standard as es
import numpy as np
from visualize import plot_waveform

DATA_DIR = "/app/data"
OUTPUT_DIR = "/app/output"

def mono_to_stereo(audio):
  return np.column_stack((audio, audio))

def analyze_track(file_path, output_dir):
    audio = es.MonoLoader(filename=file_path)()

    # convert mono to stereo for laudnessEBUR128
    stereo_audio = mono_to_stereo(audio)

    loudness = es.LoudnessEBUR128()(stereo_audio)
    rms = es.RMS()(audio)

    spectrum = es.Spectrum()(audio)
    centroid = es.Centroid()(spectrum)
    complexity = es.SpectralComplexity()(spectrum)
    instrument = es.InstrumentRecognition()(audio)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    waveform_plot = os.path.join(output_dir, f"{base_name}_waveform.png")

    plot_waveform(audio, waveform_plot)

    eq_suggestions = []
    if centroid < 1000:
        eq_suggestions.append("Boost highs above 5kHz for clarity.")
    if rms < 0.05:
        eq_suggestions.append("Increase gain or apply compression.")

    compression = {
        "ratio": "3:1 to 5:1",
        "attack": "5-20 ms",
        "release": "50-100 ms",
        "threshold": "-6 dB",
    }

    report = {
        "file": file_path,
        "instrument": instrument,
        "rms": rms,
        "loudness": loudness,
        "spectral_centroid": centroid,
        "spectral_complexity": complexity,
        "eq_suggestions": eq_suggestions,
        "compression_suggestions": compression,
        "waveform_plot": waveform_plot
    }

    print(f"\n🎛️ Analysis Report for {base_name}")
    for k, v in report.items():
        print(f"{k}: {v}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.wav')]

    if not files:
        print("⚠️ No WAV files found in /data")
        return

    print(f"🔍 Found {len(files)} WAV files to analyze...\n")

    for file_name in files:
        analyze_track(os.path.join(DATA_DIR, file_name), OUTPUT_DIR)

if __name__ == "__main__":
    main()
