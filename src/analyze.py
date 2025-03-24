import os
import essentia.standard as es
import numpy as np
from visualize import plot_waveform

DATA_DIR = "/app/data"
OUTPUT_DIR = "/app/output"

TRACK_TYPE_MAP = {
  "Bonbo.wav": "kick",
  "Reo.wav": "snare",
  "OverMic.wav": "overhead",
  "Bajo.wav": "bass",
  "Guitar.wav": "guitar",
  "Ton de Piso.wav": "tom",
  "Ton.wav": "tom",
  "HiHat.wav": "hihat",
  "Carlos.wav": "tenor",
  "Pastor.wav": "tenor",
  "Director.wav": "tenor",
  "Wendy.wav": "contra_alto",
  "Mile.wav": "contra_alto",
  "Anita.wav": "contra_alto",
  "Accordion.wav": "accordion"
}

def mono_to_stereo(audio):
  return np.column_stack((audio, audio))

def generate_recommendations(audio, rms, centroid, ber_low, ber_mid, ber_high, track_type):
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
  if track_type == 'kick':
      if ber_low < 0.5:
          eq_suggestions.append("Boost sub-bass (50-80Hz) for more punch.")
      if ber_high > 0.4:
          eq_suggestions.append("Cut highs above 5kHz to reduce click.")

  elif track_type == 'snare':
      if ber_mid < 0.4:
          eq_suggestions.append("Boost mids around 1-3kHz for snap.")
      if ber_high > 0.6:
          eq_suggestions.append("Reduce highs above 8kHz to tame harshness.")

  elif track_type == 'vocal':
      if centroid < 1500:
          eq_suggestions.append("Boost presence around 3kHz for vocal clarity.")
      if ber_high > 0.5:
          eq_suggestions.append("Use de-esser or cut 5-8kHz to tame sibilance.")

  elif track_type == 'bass':
      if ber_low < 0.4:
          eq_suggestions.append("Boost 60-100Hz for more bass body.")
      if ber_mid > 0.5:
          eq_suggestions.append("Cut 300Hz muddiness.")

  elif track_type == 'tom':
    if ber_low < 0.4:
      eq_suggestions.append("Boost 80Hz - 100Hz for body and punch.")
    if ber_mid > 0.5:
      eq_suggestions.append("Cut 250Hz - 400Hz to reduce boxiness.")
    if ber_high < 0.3:
      eq_suggestions.append("Boost 5kHz - 7kHz for attack and clarity.")

  elif track_type == 'hihat':
    if ber_low > 0.2:
        eq_suggestions.append("Cut 200Hz - 400Hz to reduce bleed from other drums.")
    if ber_high < 0.5:
        eq_suggestions.append("Boost 8kHz - 12kHz for shimmer and brightness.")
    if ber_high > 0.7:
        eq_suggestions.append("Consider taming harshness above 12kHz.")

  elif track_type == 'tenor':
      if ber_low < 0.3:
          eq_suggestions.append("Boost 150Hz - 250Hz for warmth and fullness.")
      if ber_mid < 0.4:
          eq_suggestions.append("Boost 2kHz - 5kHz for vocal presence and clarity.")
      if ber_high > 0.5:
          eq_suggestions.append("Apply de-esser or reduce 5kHz - 8kHz to tame sibilance.")

  elif track_type == 'contra_alto':
      if ber_low < 0.4:
          eq_suggestions.append("Boost 130Hz - 200Hz to add body and warmth.")
      if ber_mid < 0.4:
          eq_suggestions.append("Boost 1kHz - 3kHz for clarity and presence.")
      if ber_high > 0.6:
          eq_suggestions.append("Reduce 5kHz - 7kHz to control harshness.")

  elif track_type == 'alto':
      if ber_low < 0.3:
          eq_suggestions.append("Boost 150Hz - 250Hz for warmth and body.")
      if ber_mid < 0.4:
          eq_suggestions.append("Boost 2kHz - 4kHz for articulation and clarity.")
      if ber_high > 0.5:
          eq_suggestions.append("Reduce 5kHz - 8kHz or apply de-esser for sibilance control.")

  elif track_type == 'accordion':
      if ber_low < 0.4:
          eq_suggestions.append("Boost 100Hz - 250Hz for warmth and bass body.")
      if ber_mid > 0.5:
          eq_suggestions.append("Cut 300Hz - 500Hz to reduce boxiness and muddiness.")
      if ber_mid < 0.3:
          eq_suggestions.append("Boost 1kHz - 4kHz for clarity and presence.")
      if ber_high > 0.6:
          eq_suggestions.append("Reduce 5kHz - 8kHz to tame harshness or brittleness.")
      elif ber_high < 0.3:
          eq_suggestions.append("Boost 5kHz - 8kHz to add air and brightness.")

  # Compression logic (basic crest factor)
  crest_factor = np.max(np.abs(audio)) / rms
  if crest_factor > 6:
      compression_suggestions = {
          "ratio": "4:1",
          "attack": "10 ms",
          "release": "80 ms",
          "threshold": "-8 dB"
      }
  else:
      compression_suggestions = {
          "ratio": "2:1",
          "attack": "5 ms",
          "release": "50 ms",
          "threshold": "-4 dB"
      }

  return eq_suggestions, compression_suggestions

def analyze_track(file_path, output_dir, track_type):
  """Analyze a single audio file and output recommendations."""

  # Load audio as mono
  audio = es.MonoLoader(filename=file_path)()
  stereo_audio = mono_to_stereo(audio)

  # Extract features
  loudness = es.LoudnessEBUR128()(stereo_audio)
  rms = es.RMS()(audio)

  # Band energy ratios (low, mid, high)
  ber_low = es.BandEnergyRatio(inputSize=1024)(audio, 20, 250)
  ber_mid = es.BandEnergyRatio(inputSize=1024)(audio, 250, 4000)
  ber_high = es.BandEnergyRatio(inputSize=1024)(audio, 4000, 20000)

  # Spectral features
  spectrum = es.Spectrum()(audio)
  centroid = es.Centroid()(spectrum)

  # Plot waveform
  base_name = os.path.splitext(os.path.basename(file_path))[0]
  waveform_plot = os.path.join(output_dir, f"{base_name}_waveform.png")
  plot_waveform(audio, waveform_plot)

  # Generate recommendations based on track type
  eq_suggestions, compression_suggestions = generate_recommendations(
      audio, rms, centroid, ber_low, ber_mid, ber_high, track_type
  )

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
      "waveform_plot": waveform_plot
  }

  # Output report
  print(f"\n🎛️ Analysis Report for {base_name} ({track_type})")
  for key, value in report.items():
      print(f"{key}: {value}")

def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.wav')]

  if not files:
      print("⚠️ No WAV files found in /data")
      return

  print(f"🔍 Found {len(files)} WAV files to analyze...\n")

  for file_name in files:
      file_path = os.path.join(DATA_DIR, file_name)

      # Determine track type
      track_type = TRACK_TYPE_MAP.get(file_name, "unknown")

      if track_type == "unknown":
          print(f"⚠️ Track type for '{file_name}' not found in TRACK_TYPE_MAP. Skipping...")
          continue

      analyze_track(file_path, OUTPUT_DIR, track_type)

if __name__ == "__main__":
    main()
