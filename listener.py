import pyaudio
import numpy as np
from protocol import char_decoder, FS, BASE_FREQUENCY, TS, T_SILENCE
from scipy.signal import hilbert, medfilt
from utils import bandpass_filter, plot_psd
import time


class RealTimeListener:
    """
    The RealTimeListener class listens for acoustic signals in real-time
    and decodes them to text messages. The listener uses a simple envelope
    detection method to detect the onsets of signals and decode them into
    characters. The listener is designed to work with the sender.py script
    in real-time.
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.chunk_size = 1024
        self.buffer = np.array([])
        self.message = ""
        self.threshold = 0.5
        self.smooth_kernel = int(0.5 * FS * T_SILENCE)
        if self.smooth_kernel % 2 == 0:
            self.smooth_kernel += 1  # Make sure kernel size is odd
        self.in_segment = False
        self.segment_buffer = np.array([])
        
    def start_listening(self):
        """Start the audio stream and begin processing audio data"""
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=FS,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        print("Listening for acoustic signals... Press Ctrl+C to stop.")
        self.stream.start_stream()
        
        try:
            while self.stream.is_active():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping listener...")
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            print(f"Final decoded message: {self.message}")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Process incoming audio data"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to buffer
        self.buffer = np.append(self.buffer, audio_data)
        
        # Keep buffer at a reasonable size (3 segments worth of data)
        max_buffer_size = int(FS * (TS + T_SILENCE) * 3)
        if len(self.buffer) > max_buffer_size:
            self.buffer = self.buffer[-max_buffer_size:]
        
        # Process buffer to detect segments
        self.detect_and_decode_segments()
        
        return (in_data, pyaudio.paContinue)
    
    def detect_and_decode_segments(self):
        """Detect signal segments and decode them in real-time"""
        if len(self.buffer) < FS * (TS + T_SILENCE):
            return  # Not enough data yet
            
        # Band-pass filter to get base frequency component 
        base_signal = bandpass_filter(
            self.buffer, BASE_FREQUENCY - 100, BASE_FREQUENCY + 100, FS
        )
        analytic = hilbert(base_signal)
        amplitude = np.abs(analytic)
        envelope = medfilt(amplitude, self.smooth_kernel)
        
        # Check if we're in a segment or between segments
        if not self.in_segment:
            # Look for onset (start of segment)
            if np.max(envelope[-self.chunk_size:]) > self.threshold * np.max(envelope):
                self.in_segment = True
                self.segment_buffer = np.array([])
                print("Signal detected, decoding...")
        else:
            # Add data to segment buffer
            self.segment_buffer = np.append(self.segment_buffer, self.buffer[-self.chunk_size:])
            
            # Check if we've reached the end of a segment
            if len(self.segment_buffer) > FS * TS:
                # If the envelope has dropped, we've reached the end
                if np.max(envelope[-self.chunk_size:]) < self.threshold * np.max(envelope):
                    # Decode the segment
                    char = char_decoder(self.segment_buffer)
                    self.message += char
                    print(f"Decoded: '{char}' | Message so far: '{self.message}'")
                    
                    # Reset for next segment
                    self.in_segment = False
                    self.segment_buffer = np.array([])


def main():
    listener = RealTimeListener()
    listener.start_listening()


if __name__ == "__main__":
    main()

