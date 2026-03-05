#!/usr/bin/env python3
"""
TASK 2: Analog Video Signal Detector and Decoder
Detects and decodes PAL/NTSC signals from IQ recordings.
Run: python3 task2_signal_detector.py --input generated_signals/video_signal.iq --format PAL
"""

import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime

class AnalogVideoDetector:
    def __init__(self, video_format="PAL", sample_rate=10e6):
        self.format = video_format.upper()
        self.sample_rate = sample_rate
        
        if self.format == "PAL":
            self.line_duration = 64e-6
            self.lines_per_frame = 625
            self.field_rate = 50.0
            self.color_subcarrier = 4.43361875e6
            self.sync_level = -0.3
            self.black_level = 0.0
            self.white_level = 0.7
        elif self.format == "NTSC":
            self.line_duration = 63.56e-6
            self.lines_per_frame = 525
            self.field_rate = 59.94
            self.color_subcarrier = 3.579545e6
            self.sync_level = -0.286
            self.black_level = 0.0
            self.white_level = 0.714
        else:
            print(f"ERROR: Format '{self.format}' not supported")
            sys.exit(1)
        
        print(f"Initialized {self.format} detector")
        print(f"  Sample rate: {self.sample_rate/1e6:.1f} MS/s")
        print(f"  Expected line rate: {1/self.line_duration:.0f} Hz")

    def load_iq_file(self, filename):
        print(f"Loading IQ file: {filename}")
        
        if not os.path.exists(filename):
            print(f"ERROR: File not found: {filename}")
            sys.exit(1)
        
        raw_data = np.fromfile(filename, dtype=np.float32)
        
        if len(raw_data) % 2 != 0:
            print("WARNING: IQ file has odd number of samples")
            raw_data = raw_data[:len(raw_data)-1]
        
        i_samples = raw_data[0::2]
        q_samples = raw_data[1::2]
        
        iq_signal = i_samples + 1j * q_samples
        
        # Normalize the signal
        iq_signal = iq_signal / np.max(np.abs(iq_signal))
        
        print(f"  Loaded {len(iq_signal)} complex samples")
        print(f"  Duration: {len(iq_signal)/self.sample_rate:.3f} seconds")
        
        return iq_signal
    
    def fm_demodulate(self, iq_signal):
        print("Demodulating FM signal...")
        
        # Calculate phase
        phase = np.angle(iq_signal)
        
        # Unwrap phase to avoid jumps
        phase_unwrapped = np.unwrap(phase)
        
        # Differentiate to get instantaneous frequency
        instantaneous_freq = np.diff(phase_unwrapped) * self.sample_rate / (2 * np.pi)
        
        # Remove carrier frequency (100 MHz)
        carrier_freq = 100e6
        baseband = instantaneous_freq - carrier_freq
        
        # Normalize to expected range [-0.4, 0.8]
        deviation = 5e6  # 5 MHz deviation
        baseband = baseband / deviation
        
        # Apply median filter to reduce noise
        baseband = signal.medfilt(baseband, kernel_size=5)
        
        print(f"  Demodulated signal range: [{np.min(baseband):.3f}, {np.max(baseband):.3f}]")
        print(f"  Signal mean: {np.mean(baseband):.3f}")
        print(f"  Signal std: {np.std(baseband):.3f}")
        
        return baseband
    
    def detect_sync_pulses(self, baseband):
        print("Detecting sync pulses...")
        
        # Sync pulses are the lowest values in the signal
        sync_threshold = np.percentile(baseband, 10)  # Bottom 10% are likely sync
        
        # Find all samples below threshold
        sync_positions = np.where(baseband < sync_threshold)[0]
        
        if len(sync_positions) == 0:
            print("WARNING: No sync pulses detected")
            return np.array([])
        
        # Group nearby sync samples
        min_sync_width = int(4e-6 * self.sample_rate)  # 4 µs minimum sync width
        sync_groups = []
        current_group = [sync_positions[0]]
        
        for i in range(1, len(sync_positions)):
            if sync_positions[i] - sync_positions[i-1] < min_sync_width:
                current_group.append(sync_positions[i])
            else:
                if len(current_group) > min_sync_width // 2:
                    # Take the start of the sync pulse (first sample in group)
                    sync_groups.append(current_group[0])
                current_group = [sync_positions[i]]
        
        # Add last group
        if len(current_group) > min_sync_width // 2:
            sync_groups.append(current_group[0])
        
        print(f"  Found {len(sync_groups)} sync pulses")
        print(f"  First few sync positions: {sync_groups[:5]}")
        
        return np.array(sync_groups)
    
    def extract_video_lines(self, baseband, sync_positions):
        print("Extracting video lines...")
        
        if len(sync_positions) < 2:
            print("ERROR: Not enough sync pulses for line extraction")
            return []
        
        samples_per_line = int(self.line_duration * self.sample_rate)
        lines = []
        
        # Calculate average line spacing
        line_spacings = np.diff(sync_positions)
        avg_line_spacing = np.median(line_spacings)
        print(f"  Average line spacing: {avg_line_spacing:.1f} samples")
        print(f"  Expected: {samples_per_line:.1f} samples")
        
        for i in range(len(sync_positions)-1):
            # Use sync pulse start as reference
            line_start = sync_positions[i]
            line_end = line_start + samples_per_line
            
            if line_end < len(baseband):
                # Extract full line including sync
                full_line = baseband[line_start:line_end]
                
                # Skip the sync and blanking portion (first ~12 µs)
                samples_to_skip = int(12e-6 * self.sample_rate)
                
                if samples_to_skip < len(full_line):
                    active_video = full_line[samples_to_skip:]
                    
                    # Normalize to 0-1 range for display
                    if np.max(active_video) > np.min(active_video):
                        active_video = (active_video - np.min(active_video)) / (np.max(active_video) - np.min(active_video))
                    
                    lines.append(active_video)
        
        print(f"  Extracted {len(lines)} video lines")
        
        if len(lines) > 0:
            print(f"  Line length: {len(lines[0])} samples")
        
        return lines
    
    def reconstruct_image(self, video_lines, output_prefix):
        if len(video_lines) == 0:
            print("ERROR: No video lines to reconstruct")
            return None
        
        # Pad all lines to same length
        max_length = max(len(line) for line in video_lines)
        image_height = len(video_lines)
        
        image = np.zeros((image_height, max_length))
        
        for i, line in enumerate(video_lines):
            image[i, :len(line)] = line
        
        # Create output directory if needed
        os.makedirs('detected_videos', exist_ok=True)
        output_file = os.path.join('detected_videos', f"{output_prefix}_reconstructed.png")
        
        # Save as image
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap='gray', aspect='auto', interpolation='nearest')
        plt.title(f'{self.format} Video Reconstruction ({image_height} lines)')
        plt.colorbar(label='Intensity')
        plt.xlabel('Sample Index')
        plt.ylabel('Line Number')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        # Also save raw data for analysis
        np.save(os.path.join('detected_videos', f"{output_prefix}_raw.npy"), image)
        
        print(f"Saved reconstructed image to {output_file}")
        print(f"  Image size: {image_height} x {max_length} pixels")
        
        return output_file
    
    def analyze_signal_quality(self, baseband, sync_positions):
        print("\nSignal quality analysis:")
        
        if len(sync_positions) > 1:
            sync_intervals = np.diff(sync_positions) / self.sample_rate
            avg_line_time = np.mean(sync_intervals) * 1e6
            std_line_time = np.std(sync_intervals) * 1e6
            
            print(f"  Average line time: {avg_line_time:.2f} µs")
            print(f"  Line time std dev: {std_line_time:.2f} µs")
            print(f"  Expected: {self.line_duration*1e6:.2f} µs")
            
            # Calculate lines per frame
            lines_detected = len(sync_positions)
            frames = lines_detected / self.lines_per_frame
            print(f"  Lines detected: {lines_detected}")
            print(f"  Frames detected: {frames:.2f}")
        
        # Signal statistics
        video_samples = baseband[np.abs(baseband) < 0.8]  # Remove extreme values
        if len(video_samples) > 0:
            print(f"  Video level range: [{np.min(video_samples):.3f}, {np.max(video_samples):.3f}]")
            print(f"  Video level mean: {np.mean(video_samples):.3f}")
            print(f"  Video level std: {np.std(video_samples):.3f}")
    
    def process_signal(self, input_file, output_prefix="detected"):
        start_time = datetime.now()
        
        print("=" * 60)
        print("TASK 2: Analog Video Signal Detector")
        print("=" * 60)
        
        iq_signal = self.load_iq_file(input_file)
        baseband = self.fm_demodulate(iq_signal)
        sync_positions = self.detect_sync_pulses(baseband)
        
        if len(sync_positions) > 5:  # Need at least a few sync pulses
            video_lines = self.extract_video_lines(baseband, sync_positions)
            
            if len(video_lines) > 5:
                image_file = self.reconstruct_image(video_lines, output_prefix)
                
                self.analyze_signal_quality(baseband, sync_positions)
                
                end_time = datetime.now()
                elapsed = (end_time - start_time).total_seconds()
                
                print(f"\nProcessing complete in {elapsed:.2f} seconds")
                print(f"Detected format: {self.format}")
                
                return {
                    'format': self.format,
                    'lines_detected': len(video_lines),
                    'sync_count': len(sync_positions),
                    'image_file': image_file
                }
            else:
                print("ERROR: Could not extract video lines")
                return None
        else:
            print(f"ERROR: Only {len(sync_positions)} sync pulses detected - need at least 5")
            return None

def main():
    parser = argparse.ArgumentParser(description="Detect and decode analog video signals")
    parser.add_argument("--input", required=True, help="Input IQ file (interleaved float32)")
    parser.add_argument("--format", default="PAL", help="Expected format: PAL or NTSC")
    parser.add_argument("--sample-rate", type=float, default=10e6, 
                       help="Sample rate in Hz (default: 10e6)")
    parser.add_argument("--output", default="detected_video", 
                       help="Output filename prefix")
    
    args = parser.parse_args()
    
    detector = AnalogVideoDetector(args.format, args.sample_rate)
    result = detector.process_signal(args.input, args.output)
    
    if result:
        print(f"\nSUCCESS: Signal detected and decoded")
        print(f"  Format: {result['format']}")
        print(f"  Lines: {result['lines_detected']}")
        print(f"  Sync pulses: {result['sync_count']}")
        
        if result['image_file']:
            print(f"  Image: {result['image_file']}")
    else:
        print("\nFAILURE: Signal decoding failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
