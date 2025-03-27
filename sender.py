import pyaudio

def play_signal(signal):
    '''
    This function plays the input signal using PyAudio.

    Parameters:
    Inputs:
        signal (numpy array): Input signal.
    Outputs:
        None
    '''

    p = pyaudio.PyAudio()
    