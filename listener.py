import pyaudio
import numpy as np
from protocol import char_decoder, FS, BASE_FREQUENCY, TS, T_SILENCE, F_MAX, f_list
from utils import bandpass_filter, base_envelop, calc_fft, multi_bandpass_filter
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
    def __init__(self, freq_thresh=500, env_thresh=50, multi_bands=False):
        """
        Initialize the RealTimeListener class.
        Parameters:
        env_thresh (int): The threshold for envelope detection. Number of stds
            The default value is 50.
        freq_thresh (int): The threshold for frequency detection. Number of stds
        multi_bands (bool): Whether to use multi-band frequency detection or not.
            The default value is False.
        """
        self.p = pyaudio.PyAudio()
        self.freq_thresh = freq_thresh
        self.env_thresh = env_thresh
        self.multi_bands = multi_bands
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
        self.fig, ((self.envelop_plot, self.bar_plot), (self.psd_plot, _)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Top-left subplot for additional visualization
        self.envelop_plot.set_title('Onset Detection')
        self.envelop_plot.set_xlabel('Time')
        self.envelop_plot.set_ylabel('Envelop Amplitude')
        self.envelop_line, = self.envelop_plot.plot([], [])
        
        # Top-right subplot for bar plot
        self.bar_plot.set_title('Frequency Bar Plot')
        self.bar_plot.set_xlabel('Frequency Bands')
        self.bar_plot.set_ylabel('Amplitude')
        self.bar_plot.set_xlim(0, len(f_list) + 1)
        self.bar_plot.set_xticks(range(1, len(f_list) + 1))
        self.bar_plot.set_xticklabels([f"{f} Hz" for f in f_list], rotation=45)
        self.bar_bars = self.bar_plot.bar(range(1, len(f_list) + 1), [0] * len(f_list))
        
        # Bottom-left subplot for PSD
        self.psd_line, = self.psd_plot.plot([], [])
        self.psd_plot.set_title('Real-time Power Spectral Density')
        self.psd_plot.set_xlabel('Frequency (Hz)')
        self.psd_plot.set_ylabel('Power/Frequency')
        self.psd_plot.set_xlim(f_list[0] - 1000, f_list[-1] + 1000)
        
        self.fig.tight_layout()
        plt.show(block=False)
    
    def update_plots(self):
        """Update the plots with current buffer data"""
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
        # F = F - self.freq_mean
        F = F[:len(freqs)]
        
        # Update plot
        self.psd_line.set_data(freqs, F)
        
        self.psd_plot.relim()
        self.psd_plot.autoscale_view()



        # Update bar plot
        for i, bar in enumerate(self.bar_bars):
            band = np.arange(f_list[i] - 100, f_list[i] + 100)
            freq_roi = np.where((freqs >= band[0]) & (freqs <= band[-1]))[0]
            # Get the average power in the band
            F_roi = np.mean(F[freq_roi])
            bar.set_height(F_roi)
            bar.set_y(F_roi)
            # bar.set_y(F_roi - self.freq_mean[i])

            bar.set_color('blue' if F_roi > self.freq_ref[i] else 'red')
        

        # Clear previous line first to avoid cluttering
        if hasattr(self, '_bar_ref') and self._bar_ref:
            self._bar_ref.remove()
        self._bar_ref = self.bar_plot.scatter(
            np.arange(1, len(f_list) + 1), self.freq_ref, color='red', label='Freq Ref'
        )

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
        if self.multi_bands:
            filtered_signal = multi_bandpass_filter(
                baseline_buffer, bands=f_list, band_width=100, fs=FS
            )
        else:
            filtered_signal = bandpass_filter(
                baseline_buffer, f_list[0]-2000, f_list[-1]+2000, FS
            )
        # Split the baseline buffer into segments of TS length
        segment_size = int(T_SILENCE * FS)
        num_segments = len(baseline_buffer) // segment_size
        F_roi_segments = np.zeros((num_segments, len(f_list)))

        for i in range(num_segments):
            segment = baseline_buffer[i * segment_size:(i + 1) * segment_size]

            # Apply filtering
            if self.multi_bands:
                filtered_signal = multi_bandpass_filter(
                    segment, bands=f_list, band_width=200, fs=FS
                )
            else:
                filtered_signal = bandpass_filter(
                    segment, f_list[0] - 1000, f_list[-1] + 1000, FS
                )

            # Calculate Welch's power spectral density
            freqs, F = welch(filtered_signal, FS)

            # Average the power around the frequencies in f_list for this segment
            for f in f_list:
                band = np.arange(f - 100, f + 100)
                freq_roi = np.where((freqs >= band[0]) & (freqs <= band[-1]))[0]
                F_roi_segments[i, f_list.index(f)] = np.mean(F[freq_roi])

        # Convert to numpy array for easier manipulation
        F_roi_segments = np.array(F_roi_segments)

        # Calculate the mean and standard deviation across all segments
        F_mean = np.mean(F_roi_segments, axis=0)
        F_std = np.std(F_roi_segments, axis=0)
        
        self.freq_mean = F_mean
        self.freq_ref = F_mean + F_std * self.freq_thresh

        
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
                char = char_decoder(self.segment_buffer, freq_ref=self.freq_ref, multi_bands=self.multi_bands)
                self.message += char
                print(f"Decoded: '{char}' | Message so far: '{self.message}'")
                
                # Reset for next segment
                self.in_segment = False
                self.segment_buffer = np.array([])


def main():
    listener = RealTimeListener(freq_thresh=5000, env_thresh=50, multi_bands=True)
    listener.start_listening()


if __name__ == "__main__":
    main()

