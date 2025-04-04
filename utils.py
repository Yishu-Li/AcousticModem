import numpy as np
from scipy.signal import welch, butter, filtfilt, hilbert, medfilt
import scipy.fft as fft

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

def calc_fft(signal, fs, window=None, fmax=20000):
    """
    Calculates the Power Spectral Density (PSD) of a given signal.

    Parameters:
    Inputs:
        signal (numpy array): Input signal.
        fs (float): Sampling frequency of the signal.
        window (list): Time window to be used for the PSD calculation, in seconds.
        fmax (int): Maximum frequency to be plotted.
    Outputs:
        freqs (numpy array): Frequencies corresponding to the PSD values.
        F (numpy array): Power Spectral Density values.
    """
    if window is not None:
        signal = signal[int(window[0]*fs):int(window[1]*fs)]

    F = np.abs(fft.rfft(signal))
    freqs = fft.rfftfreq(len(signal), 1 / fs)

    return freqs, F


def plot_fft(signal, fs, window=None, log=False, fmax=20000):
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

    freqs, F = calc_fft(signal, fs, window=window)

    # Select the ROI
    freqs = freqs[freqs<=fmax]
    F = F[:len(freqs)]

    # Calculate the mean and 2*std of the PSD
    mean = np.mean(F)
    std = np.std(F)

    plt.figure(figsize=(20, 5))
    if log:
        plt.semilogy(freqs, F)
    else:
        plt.plot(freqs, F)
    plt.axhline(mean, color='g', linestyle='--', label='Mean')
    plt.axhline(mean + 2*std, color='r', linestyle='--', label='2*std')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD')
    plt.title('Power Spectral Density')
    plt.show()


def base_envelop(signal, fb, fs, smooth_kernel=0.5):
    """
    This function calculates the base envelope of the signal.

    Parameters:
    Inputs:
        signal (numpy array): Input signal.
        fb (float): Base frequency of the signal.
        fs (float): Sampling frequency of the signal.
        smooth_kernel (float): The kernel size for smoothing the envelope,
            the number between 0 and 1 indicates the fraction of T_SILENCE.
    Outputs:
        env (numpy array): Envelope of the signal.
    """
    # Band-pass the signal to find our ROI
    base_signal = bandpass_filter(signal, fb - 500, fb + 500, fs)

    # Calculate the envelope using Hilbert transform
    analytic = hilbert(base_signal)
    amplitude = np.abs(analytic)
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1  # Make sure the kernel size is odd
    envelope = medfilt(amplitude, smooth_kernel)

    return envelope