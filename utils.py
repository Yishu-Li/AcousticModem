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


def multi_bandpass_filter(signal, bands, band_width, fs, order=5):
    """
    Apply multiple bandpass filters to the signal and combine the results.

    Parameters:
    Inputs:
        signal (numpy array): Input signal.
        bands (list of tuples): List of (lowcut, highcut) frequency pairs.
        band_width (float): Bandwidth of the filter.
        fs (float): Sampling frequency of the signal.
        order (int): Order of the Butterworth filter.
    Outputs:
        filtered (numpy array): Combined filtered signal.
    """
    filtered_signal = np.zeros_like(signal)
    for band in bands:
        lowcut = band - band_width / 2
        highcut = band + band_width / 2
        filtered_signal += bandpass_filter(signal, lowcut, highcut, fs, order)
    return filtered_signal


def notch_filter(signal, notch_freq, fs, quality_factor=30):
    """
    A notch filter to remove a specific frequency from the signal.

    Parameters:
    Inputs:
        signal (numpy array): Input signal.
        notch_freq (float): Frequency to be removed.
        fs (float): Sampling frequency of the signal.
        quality_factor (float): Quality factor of the notch filter.
    Outputs:
        filtered (numpy array): Filtered signal.
    """
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = butter(2, [freq - freq / quality_factor, freq + freq / quality_factor], btype='bandstop')
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
    base_signal = bandpass_filter(signal, fb - 50, fb + 50, fs)

    # Calculate the envelope using Hilbert transform
    analytic = hilbert(base_signal)
    amplitude = np.abs(analytic)
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1  # Make sure the kernel size is odd
    envelope = medfilt(amplitude, smooth_kernel)

    return envelope