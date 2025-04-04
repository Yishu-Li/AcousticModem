import pyaudio
from protocol import text_encoder, FS
import numpy as np


class RealTimeSender:
    """
    The RealTimeSender class encodes text messages into acoustic signals
    and plays them in real-time. The sender uses a simple encoding scheme
    to convert characters into frequency-modulated signals. The sender is
    designed to work with the listener.py script in real-time.
    """

    def __init__(self):
        self.p = pyaudio.PyAudio()

    def start_sending(self):
        """Start the audio stream and begin sending audio data"""
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=FS,
            output=True,
        )

        # Main loop
        try:
            print("Sending acoustic signals... Press Ctrl+C to stop.")
            while True:
                # Wait for the user to input a message
                message = input("Please input a message: ")

                # Encode the message to an acoustic signal
                signal = text_encoder(message)

                # Write the signal to the strea
                self.stream.write(signal.tobytes())


        except KeyboardInterrupt:
            print("\nStopping sender...")
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()


def main():
    """
    The main function of the sender. The user input a message
    to the sender, and then the sender will encode the message
    to an acoustic signal and play it.
    """

    sender = RealTimeSender()
    sender.start_sending()



if __name__ == "__main__":
    main()
