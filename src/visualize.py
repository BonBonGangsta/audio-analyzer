import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(data, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(data)), data)
    plt.title('Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
