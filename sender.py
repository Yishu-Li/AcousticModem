import pyaudio
from protocol import text_encoder, FS, TS
import numpy as np


def play_signal(signal):
    """
    This function plays the input signal using PyAudio.

    Parameters:
    Inputs:
        signal (numpy array): Input signal.
    Outputs:
        None
    """

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=FS,
        output=True,
    )

    signal = signal.astype(np.float32)

    # Write the signal to the strea
    stream.write(signal.tobytes())

    # Close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()


def main():
    """
    The main function of the sender. The user input a message
    to the sneder, and then the sender will encode the message
    to an acoustic signal and play it.
    """

    # Main loop
    while True:
        # Wait for the user to input a message
        message = input("Please input a message: ")

        # Encode the message to an acoustic signal
        signal = text_encoder(message)

        # Play the signal
        play_signal(signal)


if __name__ == "__main__":
    main()
