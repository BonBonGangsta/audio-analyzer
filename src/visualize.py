import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(data, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(data)), data)
    plt.title("Waveform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_band_energy_trends(ratios_by_band, output_path):
    plt.figure(figsize=(12, 5))

    for band, ratios in ratios_by_band.items():
        plt.plot(ratios, label=f"{band.capitalize()} Energy")

        plt.xlabel("Frame")
        plt.ylabel("Normalized Energy Ratio")
        plt.title("Per-Band Energy Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
