import numpy as np
from scipy.signal import welch, butter, filtfilt

import matplotlib.pyplot as plt



def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    '''
    A bandpass Butterworth filter to filter the signal and denoise.

    Parameters:
    Inputs:
        signal (numpy array): Input signal.
        lowcut (float): Lower cutoff frequency.
        highcut (float): Higher cutoff frequency.
        fs (float): Sampling frequency of the signal.
        order (int): Order of the Butterworth filter.
    Outputs:
        filtered (numpy array): Filtered signal.
    '''
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered



def plot_psd(signal, fs, window=None, log=False, fmax=10000):
    """
    Plots the Power Spectral Density (PSD) of a given signal.

    Parameters:
    Inputs:
        signal (numpy array): Input signal.
        fs (float): Sampling frequency of the signal.
        window (list): Time window to be used for the PSD calculation, in seconds.
        fmax (int): Maximum frequency to be plotted.
    """
    if window is not None:
        signal = signal[int(window[0]*fs):int(window[1]*fs)]

    f, Pxx = welch(signal, fs)

    # Select the ROI
    f = f[f<=fmax]
    Pxx = Pxx[:len(f)]

    # Calculate the mean and 2*std of the PSD
    mean = np.mean(Pxx)
    std = np.std(Pxx)

    plt.figure(figsize=(20, 5))
    if log:
        plt.semilogy(f, Pxx)
    else:
        plt.plot(f, Pxx)
    plt.axhline(mean, color='g', linestyle='--', label='Mean')
    plt.axhline(mean + 2*std, color='r', linestyle='--', label='2*std')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD')
    plt.title('Power Spectral Density')
    plt.show()