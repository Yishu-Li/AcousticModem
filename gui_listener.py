import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox, QCheckBox)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from listener import RealTimeListener, FS, BASE_FREQUENCY
import pyaudio 

class MplCanvas(FigureCanvas):
    """Canvas for embedding matplotlib into Qt"""
    def __init__(self, parent=None, width=10, height=10, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        grid = self.fig.add_gridspec(2, 2)
        self.envelop_plot = self.fig.add_subplot(grid[0, 0])
        self.bar_plot = self.fig.add_subplot(grid[0, 1])
        self.psd_plot = self.fig.add_subplot(grid[1, :])
        
        # Setup plots
        self.envelop_plot.set_title('Onset Detection')
        self.envelop_plot.set_xlabel('Time')
        self.envelop_plot.set_ylabel('Envelope Amplitude')
        self.envelop_line, = self.envelop_plot.plot([], [])
        
        self.psd_plot.set_title('Real-time Power Spectral Density')
        self.psd_plot.set_xlabel('Frequency (Hz)')
        self.psd_plot.set_ylabel('Power/Frequency')
        self.psd_line, = self.psd_plot.plot([], [])
        
        self.bar_plot.set_title('Frequency Bar Plot')
        self.bar_plot.set_xlabel('Frequency Bands')
        self.bar_plot.set_ylabel('Amplitude')
        self.bar_bars = None
        self._bar_ref = None
        
        self.fig.tight_layout()
        
        super(MplCanvas, self).__init__(self.fig)

class AcousticModemGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.listener = None
        self.is_listening = False
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Acoustic Modem GUI')
        self.setGeometry(100, 100, 1000, 800)
        
        # Main widget
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Canvas for matplotlib
        self.canvas = MplCanvas(width=10, height=8)
        main_layout.addWidget(self.canvas)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Frequency threshold controls
        freq_group = QGroupBox("Frequency Threshold")
        freq_layout = QVBoxLayout()
        freq_group.setLayout(freq_layout)
        
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)  # Updated to use the enum class
        self.freq_slider.setMinimum(1)
        self.freq_slider.setMaximum(100)
        self.freq_slider.setValue(10)
        self.freq_slider.valueChanged.connect(self.update_freq_threshold)
        
        self.freq_label = QLabel(f"Threshold: {self.freq_slider.value()}")
        
        freq_layout.addWidget(self.freq_label)
        freq_layout.addWidget(self.freq_slider)
        
        # Envelope threshold controls
        env_group = QGroupBox("Envelope Threshold")
        env_layout = QVBoxLayout()
        env_group.setLayout(env_layout)
        
        self.env_slider = QSlider(Qt.Orientation.Horizontal)  # Updated to use the enum class
        self.env_slider.setMinimum(1)
        self.env_slider.setMaximum(200)
        self.env_slider.setValue(50)
        self.env_slider.valueChanged.connect(self.update_env_threshold)
        
        self.env_label = QLabel(f"Threshold: {self.env_slider.value()}")
        
        env_layout.addWidget(self.env_label)
        env_layout.addWidget(self.env_slider)
        
        # Multi-bands checkbox
        self.multi_bands_checkbox = QCheckBox("Use Multi-Bands")
        self.multi_bands_checkbox.setChecked(False)  # Default to False
        controls_layout.addWidget(self.multi_bands_checkbox)
        
        # Start/Stop button
        self.start_stop_button = QPushButton("Start Listening")
        self.start_stop_button.clicked.connect(self.toggle_listening)
        
        # Add controls to layout
        controls_layout.addWidget(freq_group)
        controls_layout.addWidget(env_group)
        controls_layout.addWidget(self.start_stop_button)
        
        main_layout.addLayout(controls_layout)
        
        # Decoded message and binary display
        message_group = QGroupBox("Decoded Message")
        message_layout = QVBoxLayout()
        message_group.setLayout(message_layout)
        
        self.message_label = QLabel("Message: ")
        self.binary_label = QLabel("Binary: ")
        
        message_layout.addWidget(self.message_label)
        message_layout.addWidget(self.binary_label)
        
        main_layout.addWidget(message_group)  # Add to the main layout
        
        self.setCentralWidget(main_widget)
        
    def update_freq_threshold(self):
        value = self.freq_slider.value()
        self.freq_label.setText(f"Threshold: {value}")
        if self.listener:
            self.listener.freq_thresh = value
            
    def update_env_threshold(self):
        value = self.env_slider.value()
        self.env_label.setText(f"Threshold: {value}")
        if self.listener:
            self.listener.env_thresh = value
    
    def toggle_listening(self):
        if not self.is_listening:
            # Start listening
            self.start_stop_button.setText("Stop Listening")
            self.is_listening = True
            self.start_listener()
        else:
            # Stop listening
            self.start_stop_button.setText("Start Listening")
            self.is_listening = False
            self.stop_listener()
    
    def start_listener(self):
        # Initialize listener with current slider values, checkbox state, and UI update callback
        self.listener = GUIRealTimeListener(
            freq_thresh=self.freq_slider.value(),
            env_thresh=self.env_slider.value(),
            canvas=self.canvas,
            multi_bands=self.multi_bands_checkbox.isChecked(),
            ui_callback=self.real_time_update  # Added callback for decoded updates
        )
        self.listener.start()
    
    # New: Add callback to update decoded message and binary fields
    def real_time_update(self, message):
        self.message_label.setText(f"Message: {message}")
        binary = format(ord(message[-1]), "07b") if message else "" 
        self.binary_label.setText(f"Binary: {binary}")
    
    def stop_listener(self):
        if self.listener:
            self.listener.stop()
            self.listener = None
    
    def closeEvent(self, event):
        self.stop_listener()
        event.accept()

class GUIRealTimeListener(RealTimeListener):
    """Modified RealTimeListener for GUI integration"""
    def __init__(self, freq_thresh=3, env_thresh=50, canvas=None, multi_bands=False, ui_callback=None):
        self.canvas = canvas
        self.running = True
        self.multi_bands = multi_bands
        self.ui_callback = ui_callback
        super().__init__(freq_thresh, env_thresh, multi_bands=multi_bands)
        # Initialize decoded outputs to avoid attribute errors
        self.message = ""
    
    def init_ui(self):
        # Override to use the canvas instead of creating new figures
        if self.canvas:
            self.envelop_plot = self.canvas.envelop_plot
            self.psd_plot = self.canvas.psd_plot
            self.bar_plot = self.canvas.bar_plot

            self.envelop_line = self.canvas.envelop_line
            self.psd_line = self.canvas.psd_line
            self.bar_bars = self.canvas.bar_bars
            self._bar_ref = self.canvas._bar_ref
            
            # Clear any existing threshold lines from previous runs
            self.clear_threshold_lines()
            
            # Initialize line references
            self._env_line = None
            self._freq_line = None
        else:
            # Fall back to original implementation if no canvas provided
            plt.ion()
            self.fig, (self.envelop_plot, self.psd_plot) = plt.subplots(2, 1, figsize=(10, 10))
            self.envelop_plot.set_title('Onset Detection')
            self.envelop_plot.set_xlabel('Time')
            self.envelop_plot.set_ylabel('Envelop Amplitude')
            self.envelop_line, = self.envelop_plot.plot([], [])
            
            self.psd_line, = self.psd_plot.plot([], [])
            self.psd_plot.set_title('Real-time Power Spectral Density')
            self.psd_plot.set_xlabel('Frequency (Hz)')
            self.psd_plot.set_ylabel('Power/Frequency')
            
            self.fig.tight_layout()
            plt.show(block=False)
    
    def clear_threshold_lines(self):
        """Clear all threshold lines from the plots"""
        for line in self.envelop_plot.get_lines():
            if line.get_linestyle() == '--':  # Identify threshold lines by linestyle
                line.remove()
        
        for line in self.psd_plot.get_lines():
            if line.get_linestyle() == '--':  # Identify threshold lines by linestyle
                line.remove()

        if self.bar_bars:
            for bar in self.bar_bars:
                bar.remove()
            self.bar_bars = None
    
    def update_plots(self):
        # Update the PSD plot with current buffer data
        if len(self.buffer) < self.chunk_size:
            return
            
        # Plot the envelope of the signal
        envelope = self.get_envelope()
        time = np.arange(len(envelope)) / FS
            
        self.envelop_line.set_data(time, envelope)
                
        # Remove previous threshold line and add a new one
        if hasattr(self, '_env_line') and self._env_line:
            self._env_line.remove()
        self._env_line = self.envelop_plot.axhline(self.env_ref, color='red', linestyle='--', label='Env Ref')
        
        self.envelop_plot.relim()
        self.envelop_plot.autoscale_view()
            
        # Calculate and plot frequency domain
        freqs, F = self.get_psd(self.multi_bands)
            
        self.psd_line.set_data(freqs, F)
        
        self.psd_plot.relim()
        self.psd_plot.autoscale_view()
        
        # Add/update bar plot
        from protocol import f_list
        if self.bar_plot and self.bar_bars is None:
            self.bar_plot.set_xlim(0, len(f_list) + 1)
            self.bar_plot.set_xticks(range(1, len(f_list) + 1))
            self.bar_plot.set_xticklabels([f"{f} Hz" for f in f_list], rotation=45)
            self.bar_bars = self.bar_plot.bar(range(1, len(f_list) + 1), [0]*len(f_list))
        
        if self.bar_bars:
            for i, bar in enumerate(self.bar_bars):
                band = np.arange(f_list[i] - 100, f_list[i] + 100)
                freq_roi = np.where((freqs >= band[0]) & (freqs <= band[-1]))[0]
                F_roi = np.mean(F[freq_roi]) if len(freq_roi) else 0
                bar.set_height(F_roi)
                bar.set_color('red' if F_roi > self.freq_ref[i] else 'blue')
            
            # Remove all existing scatter objects from the plot
            for coll in self.bar_plot.collections:
                coll.remove()
            self._bar_ref = None

            self._bar_ref = self.bar_plot.scatter(
                range(1, len(f_list) + 1), self.freq_ref, color='red'
            )

            y_max = np.maximum(np.nan_to_num(max(F)), np.nan_to_num(max(self.freq_ref))) * 1.2
            if y_max is None or y_max == 0:
                y_max = 1.0
            self.bar_plot.set_ylim(0, y_max)
            self.bar_plot.autoscale_view()
            
        # Different draw method when using Qt canvas
        if self.canvas:
            self.canvas.draw()
        else:
            self.fig.canvas.draw_idle()
            plt.pause(0.01)
            self.fig.canvas.flush_events()

        # Real-time update for decoded outputs:
        if self.ui_callback:
            self.ui_callback(self.message)
    
    def get_envelope(self):
        from utils import base_envelop
        return base_envelop(
            self.buffer, BASE_FREQUENCY, FS, smooth_kernel=self.smooth_kernel
        )
        
    def get_psd(self, multi_bands=False):
        from utils import bandpass_filter, multi_bandpass_filter
        from scipy.signal import welch
        from protocol import f_list, F_MAX
            
        if not multi_bands:
            signal = bandpass_filter(
                self.buffer, f_list[0]-1000, f_list[-1]+1000, FS
            )
        else:
            signal = multi_bandpass_filter(
                self.buffer, bands=f_list, band_width=200, fs=FS
            )
        freqs, F = welch(signal, FS)
        freqs = freqs[freqs<=F_MAX]
        # F = F - self.freq_mean
        F = F[:len(freqs)]
        return freqs, F
    
    def start_listening(self):
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
            
        print("Listening for acoustic signals...")
        self.stream.start_stream()
            
        while self.stream.is_active() and self.running:
            self.update_plots()
            QApplication.processEvents()  # Process Qt events
    
    def start(self):
        """Start listening in non-blocking mode"""
        import threading
        self.thread = threading.Thread(target=self.start_listening)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop listening"""
        self.running = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        print(f"Final decoded message: {self.message}")
        
        # Clear threshold lines when stopped
        if hasattr(self, 'canvas') and self.canvas:
            self.clear_threshold_lines()

def main():
    app = QApplication(sys.argv)
    window = AcousticModemGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
