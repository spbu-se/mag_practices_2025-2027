#!/usr/bin/env python3
"""
TASK 1: Analog Video Signal Generator
Generates PAL/NTSC baseband signals with FM modulation for SDR transmission.
Run: python3 task1_signal_generator.py --format PAL --duration 0.1 --output pal_signal.iq
"""

import numpy as np
import argparse
import sys
import os
from datetime import datetime

class AnalogVideoGenerator:
    def __init__(self, video_format="PAL"):
        """Initialize with real PAL/NTSC broadcast parameters"""
        self.format = video_format.upper()
        
        if self.format == "PAL":
            # PAL-B/G standard (Europe)
            self.line_duration = 64e-6          # 64µs per line
            self.lines_per_frame = 625          # Total lines
            self.active_lines = 576              # Visible lines
            self.field_rate = 50.0               # 50 Hz
            self.color_subcarrier = 4.43361875e6  # 4.43 MHz
            self.sync_level = -0.3
            self.black_level = 0.0
            self.white_level = 0.7
            self.bandwidth = 5.0e6
            
        elif self.format == "NTSC":
            # NTSC-M standard (USA/Japan)
            self.line_duration = 63.56e-6        # 63.56µs per line
            self.lines_per_frame = 525
            self.active_lines = 480
            self.field_rate = 59.94               # 59.94 Hz
            self.color_subcarrier = 3.579545e6    # 3.58 MHz
            self.sync_level = -0.286
            self.black_level = 0.0
            self.white_level = 0.714
            self.bandwidth = 4.2e6
            
        else:
            print(f"ERROR: Format '{self.format}' not supported. Use PAL or NTSC.")
            sys.exit(1)
            
        # SDR parameters
        self.sample_rate = 10e6                   # 10 MS/s
        self.fm_deviation = 5.0e6                 # 5 MHz FM deviation
        self.carrier_freq = 100e6                  # 100 MHz center frequency
        
        print(f"Initialized {self.format} generator")
        print(f"  Sample rate: {self.sample_rate/1e6:.1f} MS/s")
        print(f"  Frame rate: {self.field_rate:.2f} Hz")
        print(f"  Carrier: {self.carrier_freq/1e6:.1f} MHz")

    def generate_test_pattern(self, duration=0.1):
        """Generate standard test pattern with color bars"""
        print(f"Generating {self.format} test pattern ({duration}s)...")
        
        samples_per_line = int(self.line_duration * self.sample_rate)
        lines_total = int(duration / self.line_duration) + 100  # Add extra lines
        
        total_samples = lines_total * samples_per_line
        
        # Initialize signal with black level
        video_signal = np.full(total_samples, self.black_level, dtype=np.float32)
        
        for line in range(lines_total):
            line_start = line * samples_per_line
            
            # Add sync pulse (negative-going)
            sync_start = line_start
            sync_end = line_start + int(4.7e-6 * self.sample_rate)
            if sync_end < total_samples:
                video_signal[sync_start:sync_end] = self.sync_level
            
            # Add back porch after sync
            back_porch_start = sync_end
            back_porch_end = back_porch_start + int(5.8e-6 * self.sample_rate)
            if back_porch_end < total_samples:
                video_signal[back_porch_start:back_porch_end] = self.black_level
            
            # Add color burst during back porch
            burst_start = back_porch_start + int(1.5e-6 * self.sample_rate)
            burst_end = burst_start + int(10e-6 * self.sample_rate)
            if burst_end < total_samples:
                t = np.arange(burst_end - burst_start) / self.sample_rate
                color_burst = 0.2 * np.sin(2 * np.pi * self.color_subcarrier * t)
                video_signal[burst_start:burst_end] += color_burst
            
            # Add active video
            active_start = back_porch_end
            active_end = active_start + int(52e-6 * self.sample_rate)  # Active video duration
            if active_end < total_samples:
                active_length = active_end - active_start
                
                # Create pattern based on line number
                line_in_frame = line % self.lines_per_frame
                
                if line_in_frame < self.active_lines:
                    if line_in_frame < self.active_lines // 2:
                        # Top half: color bars
                        bar_width = active_length // 8
                        for bar in range(8):
                            bar_start = active_start + bar * bar_width
                            bar_end = bar_start + bar_width
                            bar_values = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
                            video_signal[bar_start:bar_end] = bar_values[bar]
                    else:
                        # Bottom half: gradient
                        gradient = np.linspace(0.1, 0.7, active_length)
                        video_signal[active_start:active_end] = gradient
        
        return video_signal[:int(duration * self.sample_rate)]
    
    def fm_modulate(self, baseband):
        """Apply broadcast FM modulation"""
        print(f"Applying FM modulation ({self.fm_deviation/1e6:.1f} MHz deviation)...")
        
        # Normalize baseband to [-1, 1]
        baseband_norm = baseband / np.max(np.abs(baseband))
        
        # Time vector
        t = np.arange(len(baseband_norm)) / self.sample_rate
        
        # FM modulation: phase = integral of frequency
        # frequency = carrier_freq + deviation * baseband
        phase = 2 * np.pi * (self.carrier_freq * t + 
                             self.fm_deviation * np.cumsum(baseband_norm) / self.sample_rate)
        
        # Generate complex baseband (IQ)
        i_component = np.cos(phase)
        q_component = np.sin(phase)
        fm_signal = i_component + 1j * q_component
        
        return fm_signal
    
    def save_iq_file(self, signal, filename):
        """Save IQ data in SDR-compatible format"""
        interleaved = np.zeros(2 * len(signal), dtype=np.float32)
        interleaved[0::2] = np.real(signal)
        interleaved[1::2] = np.imag(signal)
        
        os.makedirs('generated_signals', exist_ok=True)
        filepath = os.path.join('generated_signals', filename)
        interleaved.tofile(filepath)
        
        file_size_mb = len(interleaved) * 4 / 1e6
        print(f"Saved signal to {filepath}")
        print(f"  Samples: {len(signal)} complex")
        print(f"  Size: {file_size_mb:.2f} MB")
        
        return filepath
    
    def generate_and_save(self, duration, output_file):
        """Complete generation pipeline"""
        start_time = datetime.now()
        
        baseband = self.generate_test_pattern(duration)
        fm_signal = self.fm_modulate(baseband)
        filepath = self.save_iq_file(fm_signal, output_file)
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"\nGeneration complete in {elapsed:.2f} seconds")
        print(f"Output format: {self.format}")
        print(f"Duration: {duration} seconds")
        print(f"Sample rate: {self.sample_rate/1e6} MHz")
        
        return filepath

def main():
    parser = argparse.ArgumentParser(description="Generate analog video signals for SDR")
    parser.add_argument("--format", default="PAL", help="Video format: PAL or NTSC")
    parser.add_argument("--duration", type=float, default=0.1, 
                       help="Signal duration in seconds (default: 0.1)")
    parser.add_argument("--output", default="video_signal.iq", 
                       help="Output filename (default: video_signal.iq)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TASK 1: Analog Video Signal Generator")
    print("=" * 60)
    
    generator = AnalogVideoGenerator(args.format)
    generator.generate_and_save(args.duration, args.output)

if __name__ == "__main__":
    main()
