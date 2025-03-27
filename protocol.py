import numpy as np
import scipy.fft as fft
from scipy.signal import hilbert, medfilt
from utils import bandpass_filter, plot_psd

# Constants
FS = 44100            # Standard sampling rate
TS = 0.2              # Duration of each signal segment
T_SILENCE = 0.1       # Duration of the silence between segments
BASE_FREQUENCY = 1000  # Base frequency for encoding the bits
f_list = [3000, 4000, 5000, 6000, 7000, 8000, 9000]  # Frequencies to encode the bits

def text_encoder(message: str, Gaussian_noise=False): 
    '''
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
    '''

    binary = [format(ord(char), '07b') for char in message] # Keep 7-digits to make sure the encoding is consistent
    
    # Create the signal
    signal = []
    # Loop over each character
    for c in binary:
        seg = np.sin(2*np.pi*BASE_FREQUENCY*np.arange(0, TS, 1/FS))
        for i, bit in enumerate(c):
            if bit == '1':
                # Add a sine wave with the corresponding frequency in f_list
                seg += np.sin(2*np.pi*f_list[i]*np.arange(0, TS, 1/FS)) 

        # Concatenate the silence period
        silence = np.zeros(int(FS*T_SILENCE))
        seg = np.concatenate((seg, silence))

        signal = np.concatenate((signal, seg))
    
    # Add Gaussian noise
    if Gaussian_noise:
        signal += np.random.normal(0, 0.1, len(signal))

    return signal



def char_decoder(seg: np.array, if_plot=False):
    '''
    This function decodes a signal segment into a character based on the
    frequency encoding. 

    Parameters:
    Inputs:
        seg (np.array): The signal segment that will be decoded.
    Outputs:
        char (str): The character that the signal segment represents.
    '''

    # Band-pass the signal to find our ROI
    seg = bandpass_filter(seg, f_list[0]-1000, f_list[-1]+1000, FS)
    if if_plot:
        plot_psd(seg, FS, fmax=10000)

    # Convert the signal segment to the frequency domain
    seg_f = np.abs(fft.rfft(seg))
    freqs = fft.rfftfreq(len(seg), 1/FS)

    # Calculate the baseline to reduce noise
    mean = np.mean(seg_f)
    std = np.std(seg_f)

    # Find the value of the frequencies in f_list
    # that are above the 2*std threshold
    significant = freqs[seg_f > mean + 2*std]
    significant = significant.astype(int)
    binary = list("0000000")
    for i, f in enumerate(f_list):
        # Test f and it's surrounding frequencies
        band = np.arange(f-50, f+50)
        if any(freq in significant for freq in band):
            binary[i] = '1'
    binary = ''.join(binary)
    decoded_char = chr(int(binary, 2))
    
    return decoded_char


def onset_detector_offline(signal: np.array, threshold=0.5, smooth_kernel=0.5):
    '''
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
        smooth_kernel (float): The kernel size for smoothing the envelope,
            the number between 0 and 1 indicates the fraction of T_SILENCE.
    Outputs:
        segments (list of np.array): A list of the detected segments.
    '''

    # Band-pass the signal to get the base frequency component and get envelope
    base_signal = bandpass_filter(signal, BASE_FREQUENCY-100, BASE_FREQUENCY+100, FS)
    analytic = hilbert(base_signal)
    amplitude = np.abs(analytic)
    smooth_kernel = int(smooth_kernel*FS*T_SILENCE)
    if smooth_kernel % 2 == 0:
        smooth_kernel += 1 # Make sure the kernel size is odd
    envelope = medfilt(amplitude, smooth_kernel)

    # Double the envelope and find the onsets
    # HACK: This is a simple thresholding method that works for this case
    # perhaps a more sophisticated method is needed later
    # But maybe it is fine since it's smoothed and the threshold is parameterized
    binary = envelope > threshold*np.max(envelope)
    diff = np.diff(binary.astype(int))
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]

    if len(onsets) != len(offsets):
        print('WARNING: Onsets and offsets do not match!')

        # Delete the unmatched onsets and offsets
        if onsets[0] > offsets[0]:
            onsets = onsets[1:]
        if onsets[-1] > offsets[-1]:
            offsets = offsets[:-1]
        
    # Get the segments
    segments = []
    for onset, offset in zip(onsets, offsets):
        segments.append(signal[onset:offset])

    return segments




def text_decoder(signal: np.array, if_plot=False):
    '''
    This function decodes an acoustical signal into a text message based
    on the frequency encoding. The decoding process is done by decoding each
    signal segment into a character.

    Parameters:
    Inputs:
        signal (np.array): The acoustical signal that will be decoded.
    Outputs:
        decoded (str): The text message that the signal represents.
    '''

    # Split the signal into segments
    segments = onset_detector_offline(signal)

    # Decode each segment
    decoded = ''
    for seg in segments:
        decoded += char_decoder(seg, if_plot=if_plot)
    
    return decoded



# For debug only
if __name__=='__main__':
    signal = text_encoder('I want to see if there is any bug in this protocol!', Gaussian_noise=True)
    c = char_decoder(signal[0:int(FS*0.1)])
    text = text_decoder(signal, if_plot=False)
    print(text)