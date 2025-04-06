import pyaudio
import numpy as np
from protocol import char_decoder, FS, BASE_FREQUENCY, TS, T_SILENCE, F_MAX, f_list
from utils import bandpass_filter, base_envelop, calc_fft
from scipy.signal import welch
import time
import matplotlib.pyplot as plt


class RealTimeListener:
    """
    The RealTimeListener class listens for acoustic signals in real-time
    and decodes them to text messages. The listener uses a simple envelope
    detection method to detect the onsets of signals and decode them into
    characters. The listener is designed to work with the sender.py script
    in real-time.
    """
    def __init__(self, freq_thresh=3, env_thresh=50):
        """
        Initialize the RealTimeListener class.
        Parameters:
        env_thresh (int): The threshold for envelope detection. Number of stds
            The default value is 50.
        """
        self.p = pyaudio.PyAudio()
        self.freq_thresh = freq_thresh
        self.env_thresh = env_thresh
        self.chunk_size = int(T_SILENCE * FS)
        self.buffer = np.array([])
        self.message = ""
        self.smooth_kernel = int(0.5 * FS * T_SILENCE)
        if self.smooth_kernel % 2 == 0:
            self.smooth_kernel += 1  # Make sure kernel size is odd
        self.in_segment = False
        self.segment_buffer = np.array([])

        # Initialize the UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI window"""
        plt.ion()  # Enable interactive mode
        self.fig, (self.envelop_plot, self.psd_plot) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Top subplot for additional visualization
        self.envelop_plot.set_title('Onset Detection')
        self.envelop_plot.set_xlabel('Time')
        self.envelop_plot.set_ylabel('Envelop Amplitude')
        # self.envelop_plot.set_ylim(0, 0.03)
        self.envelop_line, = self.envelop_plot.plot([], [])
        
        # Bottom subplot for PSD
        self.psd_line, = self.psd_plot.plot([], [])
        self.psd_plot.set_title('Real-time Power Spectral Density')
        self.psd_plot.set_xlabel('Frequency (Hz)')
        self.psd_plot.set_ylabel('Power/Frequency')
        self.psd_plot.set_xlim(0, F_MAX)
        
        self.fig.tight_layout()
        plt.show(block=False)
    
    def update_plots(self):
        """Update the PSD plot with current buffer data"""
        # Start to plot after we have enough data
        if len(self.buffer) < self.chunk_size:
            return

        # Plot the envelope of the signal
        envelope = base_envelop(
            self.buffer, BASE_FREQUENCY, FS, smooth_kernel=self.smooth_kernel
        )
        time = np.arange(len(envelope)) / FS

        self.envelop_line.set_data(time, envelope)
        
        # Clear previous line first to avoid cluttering
        if hasattr(self, '_env_line') and self._env_line:
            self._env_line.remove()
        self._env_line = self.envelop_plot.axhline(self.env_ref, color='red', linestyle='--', label='Env Ref')
        
        self.envelop_plot.relim()
        self.envelop_plot.autoscale_view()

            
        # Calculate frequency domain
        signal = bandpass_filter(
            self.buffer, f_list[0]-2000, f_list[-1]+2000, FS
        )
        # signal = self.buffer
        freqs, F = welch(signal, FS)
        freqs = freqs[freqs<=F_MAX]
        F = F - self.freq_mean
        F = F[:len(freqs)]
        
        # Update plot
        self.psd_line.set_data(freqs, F)
        
        # Clear previous line first to avoid cluttering
        if hasattr(self, '_freq_line') and self._freq_line:
            self._freq_line.remove()
        self._freq_line = self.psd_plot.axhline(self.freq_ref, color='red', linestyle='--', label='Freq Ref')
        
        self.psd_plot.relim()
        self.psd_plot.autoscale_view()
        self.fig.canvas.draw_idle()
        plt.pause(0.01)
        self.fig.canvas.flush_events()


    def record_reference(self):
        """Record a reference signal for envelope detection"""
        print("Recording reference signal for envelope detection...")

        # Open a separate stream for reference measurement
        ref_stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=FS,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        # Record 1 second of data for baseline
        baseline_buffer = np.array([])
        for _ in range(int(FS // self.chunk_size)):
            raw_data = ref_stream.read(self.chunk_size)
            baseline_buffer = np.append(baseline_buffer, np.frombuffer(raw_data, dtype=np.float32))

        ref_stream.close()

        # Calculate the envelope baseline
        baseline_envelope = base_envelop(
            baseline_buffer, BASE_FREQUENCY, FS, smooth_kernel=self.smooth_kernel
        )
        env_mean = np.mean(baseline_envelope)
        env_std = np.std(baseline_envelope)

        self.env_ref = env_mean + self.env_thresh * env_std

        # Calculate the frequency domain baseline
        filtered_signal = bandpass_filter(
            baseline_buffer, f_list[0]-2000, f_list[-1]+2000, FS
        )
        # filtered_signal = baseline_buffer
        freqs, F = welch(filtered_signal, FS)

        roi = (freqs<=f_list[-1]+1000) & (freqs>=f_list[0]-1000)
        freqs = freqs[roi]
        F = F[roi]
        mean = np.mean(F)
        std = np.std(F)
        self.freq_mean = mean
        self.freq_ref = mean + self.freq_thresh * std

        
    def start_listening(self):
        """Start the audio stream and begin processing audio data"""
        # Record a reference signal for envelope detection
        self.record_reference()

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
                # Update PSD plot
                self.update_plots()
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
        envelope = base_envelop(
            self.buffer, BASE_FREQUENCY, FS, smooth_kernel=self.smooth_kernel
        )
        
        # Check if we're in a segment or between segments
        if not self.in_segment:
            # Look for onset (start of segment)
            if np.max(envelope[-self.chunk_size:]) > self.env_ref:
                self.in_segment = True
                self.segment_buffer = np.array([])
                print("Signal detected, decoding...")
        else:
            # Add data to segment buffer
            self.segment_buffer = np.append(self.segment_buffer, self.buffer[-self.chunk_size:])
            
            # If the envelope has dropped, we've reached the end
            if np.max(envelope[-self.chunk_size:]) < self.env_ref:
                # Decode the segment
                char = char_decoder(self.segment_buffer)
                self.message += char
                print(f"Decoded: '{char}' | Message so far: '{self.message}'")
                
                # Reset for next segment
                self.in_segment = False
                self.segment_buffer = np.array([])


def main():
    listener = RealTimeListener(freq_thresh=5000, env_thresh=50
                                )
    listener.start_listening()


if __name__ == "__main__":
    main()

