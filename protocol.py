import numpy as np
import scipy.fft as fft
from scipy.signal import welch
from utils import bandpass_filter, plot_psd, plot_fft, base_envelop, calc_fft, multi_bandpass_filter
import matplotlib.pyplot as plt
import time


# Constants
FS = 44100  # Standard sampling rate
TS = 0.2  # Duration of each signal segment
T_SILENCE = 0.1  # Duration of the silence between segments
BASE_FREQUENCY = 1000  # Base frequency for encoding the bits
BASE_FREQUENCY_SILENCE = 2000  # Base frequency for the entire signal
f_list = [4000, 5000, 6000, 7000, 8000, 9000, 10000]  # Frequencies to encode the bits
F_MAX = f_list[-1] + 1000  # Maximum frequency


def text_encoder(message: str, Gaussian_noise=False):
    """
    This function encodes a text message into an acoustical signal based
    on the ASCII and frequency encoding. The encoded message will have a
    binary representation of the ASCII code of each character in the message.

    Each character will be encoded into a 0.2-second signal segment and
    there will be a 0.1-second silence between each segment.

    Parameters:
    Inputs:
        message (str): The message that will be encoded.
        Gaussian_noise (bool): A flag that indicates whether to add Gaussian
            noise to the signal or not. Useful for testing the robustness of
            the protocol.
    Outputs:
        signal (np.array): The acoustical signal that represents the encoded
            message.
    """

    # Record the time needed for encoding
    start = time.time()

    binary = [
        format(ord(char), "07b") for char in message
    ]  # Keep 7-digits to make sure the encoding is consistent

    for i, c in enumerate(binary):
        print(f"Encoding character {i}: {c}")

    # Create the signal
    signal = []
    # Loop over each character
    for c in binary:
        seg = np.sin(2 * np.pi * BASE_FREQUENCY * np.arange(0, TS, 1 / FS))
        for i, bit in enumerate(c):
            if bit == "1":
                # Add a sine wave with the corresponding frequency in f_list
                seg += np.sin(2 * np.pi * f_list[i] * np.arange(0, TS, 1 / FS))

        # Concatenate the silence period
        silence = np.zeros(int(FS * T_SILENCE))
        seg = np.concatenate((seg, silence))

        signal = np.concatenate((signal, seg))

    # Add the silence frequency to the entire signal
    signal += np.sin(2 * np.pi * BASE_FREQUENCY_SILENCE * np.arange(0, len(signal) / FS, 1 / FS))

    # Add Gaussian noise
    if Gaussian_noise:
        signal += np.random.normal(0, 0.1, len(signal))

    # Print the time needed for encoding
    end = time.time()
    print(f"Time used for encoding: {end - start} seconds")

    return signal


def char_decoder(seg: np.array, if_plot=False, freq_ref=None, multi_bands=False):
    """
    This function decodes a signal segment into a character based on the
    frequency encoding.

    Parameters:
    Inputs:
        seg (np.array): The signal segment that will be decoded.
        if_plot (bool): A flag that indicates whether to plot the signal
            segment envelop or not.
        freq_ref (numpy array): The reference frequency for the decoding.
        multi_bands (bool): Whether to use multi-band frequency detection or not.
            The default value is False.
    Outputs:
        char (str): The character that the signal segment represents.
    """

    freqs_raw, seg_f_raw = welch(seg, FS)

    if not multi_bands:
        # Band-pass the signal to find our ROI
        seg = bandpass_filter(seg, f_list[0] - 500, f_list[-1] + 500, FS)
    else:
        # Try to band-pass all the frequencies in f_list
        seg = multi_bandpass_filter(seg, bands=f_list, band_width=200, fs=FS)

    # if if_plot:
    #     plot_psd(seg, FS, fmax=F_MAX)

    # Convert the signal segment to the frequency domain
    # freqs, seg_f = calc_fft(seg, FS)
    freqs, seg_f = welch(seg, FS)

    mean = np.mean(seg_f)
    std = np.std(seg_f)

    # Average the power around the frequencies in f_list to a list
    F_roi = []
    for f in f_list:
        # # Find the frequencies around f
        # band = np.arange(f - 100, f + 100)
        # freq_roi = np.where((freqs >= band[0]) & (freqs <= band[-1]))[0]
        # # Get the average power in the band
        # F_roi.append(np.mean(seg_f[freq_roi]))

        # Find the closest frequency in the segment
        idx = np.argmin(np.abs(freqs - f))
        F_roi.append(seg_f[idx])
    F_roi = np.array(F_roi)

    # Create a bar plot of the frequencies in f_list and their average power
    if if_plot:
        plt.figure(figsize=(10, 10))
        plt.subplot(3, 1, 1)
        plt.plot(seg)
        plt.title(f"Signal Segment: {seg.shape[0]} samples")
        plt.xlabel("Time [samples]")
        plt.ylabel("Amplitude")


        plt.subplot(3, 1, 2)
        plt.plot(freqs_raw, seg_f_raw, label='Raw')
        plt.plot(freqs, seg_f, label='Filtered')
        plt.axhline(mean, color='g', linestyle='--', label='Mean')
        plt.axhline(mean + 3 * std, color='r', linestyle='--', label='Mean + 3*std')
        plt.legend()
        plt.title("FFT of the segment")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power")
        plt.xlim(0, F_MAX)

        plt.subplot(3, 1, 3)
        plt.bar(np.arange(len(f_list)), F_roi)
        if freq_ref is not None:
            plt.scatter(np.arange(len(f_list)), freq_ref, color='r', label='Reference')
        else:
            plt.axhline(mean + 3 * std, color='r', linestyle='--', label='Mean + 3*std')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power")
        plt.title("Power of Frequencies")

        plt.tight_layout()
        plt.show()



    if freq_ref is None:
        # Find the value of the frequencies in f_list
        # that are above the 2*std threshold
        significant = F_roi > mean + 3 * std
        significant = significant.astype(int)
    else:
        # Find the frequencies above freq_ref
        significant = F_roi > freq_ref
    binary = ''.join(significant.astype(int).astype(str))
    print(f"Binary: {binary}")
    decoded_char = chr(int(binary, 2))

    return decoded_char


def onset_detector_offline(signal: np.array, threshold=0.5, smooth_coeff=0.5, plot_envelop=False):
    """
    This function detects the onsets of the signal segments based on the
    base frequency component in the segments and returns the segment. It
    first band-pass the signal to get the base frequency components, then
    gets the envelope of the signal and smooth it using median value filter
    and finally finds the onsets based on thresholding, where the threshold
    is a fraction of the maximum amplitude of the envelope.

    Parameters:
    Inputs:
        signal (np.array): The acoustical signal that will be segmented.
        threshold (float): The threshold for the onset detection, a fraction
            of the maximum amplitude of the smoothed envelope.
        smooth_coeff (float): The kernel size for smoothing the envelope,
            the number between 0 and 1 indicates the fraction of T_SILENCE.
        plot_envelop (bool): A flag that indicates whether to plot the
            envelope or not.
    Outputs:
        segments (list of np.array): A list of the detected segments.
    """

    # Get envelope
    smooth_kernel = int(smooth_coeff * FS * T_SILENCE)
    envelope = base_envelop(signal, fb=BASE_FREQUENCY, fs=FS, smooth_kernel=smooth_kernel)        

    # Double the envelope and find the onsets
    # HACK: This is a simple thresholding method that works for this case
    # perhaps a more sophisticated method is needed later
    # But maybe it is fine since it's smoothed and the threshold is parameterized
    binary = envelope > threshold * np.max(envelope)
    diff = np.diff(binary.astype(int))
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]

    if plot_envelop:
        plt.figure(figsize=(20, 5))
        plt.subplot(2, 1, 1)
        plt.plot(envelope)
        plt.title("Envelope")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(diff)
        plt.xlabel("Time [samples]")
        plt.ylabel("Binary Signal Diff")
        plt.tight_layout()
        plt.show()

    # Delete the unmatched onsets and offsets
    if onsets[0] > offsets[0]:
        pad = int(np.maximum(0, offsets[0] - TS * FS))
        onsets = np.insert(onsets, 0, pad)
        print(f"Padded index {pad} to the first onset")
    if onsets[-1] > offsets[-1]:
        onsets = onsets[:-1]

    # Get the segments
    segments = []
    for onset, offset in zip(onsets, offsets):
        segments.append(signal[onset:offset])

    print(f"Number of segments detected: {len(segments)}")

    return segments


def signal_reference(signal: np.array, multi_bands=False):
    """
    Calculate the reference frequency of the entire signal based on the power.

    Parameters:
    Inputs:
        signal (np.array): The acoustical signal that will be segmented.
        multi_bands (bool): Whether to use multi-band frequency detection or not.
            The default value is False.
    Outputs:
        freq_ref (numpy array): The reference frequency for the decoding.
    """

    if not multi_bands:
        # Band-pass the signal to find our ROI
        signal = bandpass_filter(signal, f_list[0] - 500, f_list[-1] + 500, FS)
    else:
        # Try to band-pass all the frequencies in f_list
        signal = multi_bandpass_filter(signal, bands=f_list, band_width=200, fs=FS)


    segment_size = int(T_SILENCE * FS)
    num_segments = len(signal) // segment_size
    F_roi_segments = np.zeros((num_segments, len(f_list)))

    for i in range(num_segments):
        segment = signal[i * segment_size:(i + 1) * segment_size]

        # Apply filtering
        if multi_bands:
            filtered_signal = multi_bandpass_filter(
                segment, bands=f_list, band_width=200, fs=FS
            )
        else:
            filtered_signal = bandpass_filter(
                segment, f_list[0] - 500, f_list[-1] + 500, FS
            )

        # Calculate Welch's power spectral density
        freqs, F = welch(filtered_signal, FS)

        # Average the power around the frequencies in f_list for this segment
        for f in f_list:

            idx = np.argmin(np.abs(freqs - f))
            F_roi_segments[i, f_list.index(f)] = np.mean(F[idx])

    # Convert to numpy array for easier manipulation
    F_roi_segments = np.array(F_roi_segments)

    # Calculate the mean and standard deviation across all segments
    F_mean = np.mean(F_roi_segments, axis=0)
    F_std = np.std(F_roi_segments, axis=0)

    freq_ref = F_mean
    return freq_ref




def text_decoder(signal: np.array, if_plot=False, plot_envelop=False, multi_bands=False):
    """
    This function decodes an acoustical signal into a text message based
    on the frequency encoding. The decoding process is done by decoding each
    signal segment into a character.

    Parameters:
    Inputs:
        signal (np.array): The acoustical signal that will be decoded.
    Outputs:
        decoded (str): The text message that the signal represents.
    """

    # Calculate the reference frequency of the entire signal
    freq_ref = signal_reference(signal, multi_bands=multi_bands)
    if if_plot:
        plt.figure(figsize=(10, 10))
        plt.bar(np.arange(len(f_list)), freq_ref)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power")
        plt.title("Reference Frequencies")
        plt.show()

    # Split the signal into segments
    segments = onset_detector_offline(signal, plot_envelop=plot_envelop)

    # Decode each segment
    decoded = ""
    for seg in segments:
        decoded += char_decoder(seg, if_plot=if_plot, multi_bands=multi_bands, freq_ref=freq_ref)

    return decoded


# For debug only
if __name__ == "__main__":
    signal = text_encoder("Hello, I want to join Paradromics!", Gaussian_noise=True)
    text = text_decoder(signal, if_plot=True, plot_envelop=True, multi_bands=True)
    print(f"Decoded text: {text}")
