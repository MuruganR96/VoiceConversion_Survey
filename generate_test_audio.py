#!/usr/bin/env python3
"""
Generate synthetic test audio files for voice conversion testing
Creates male and female voice samples
"""

import numpy as np
import soundfile as sf
import os

def generate_synthetic_voice(duration=3.0, f0=150, sr=16000, output_path='test.wav'):
    """
    Generate synthetic voice-like audio

    Args:
        duration: Audio duration in seconds
        f0: Fundamental frequency (Hz) - 150Hz for male, 250Hz for female
        sr: Sample rate
        output_path: Output file path
    """
    t = np.linspace(0, duration, int(sr * duration))

    # Create harmonics (more realistic than simple sine)
    signal = np.zeros_like(t)

    # Add fundamental and harmonics
    harmonics = [1, 2, 3, 4, 5]  # First 5 harmonics
    amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1]  # Decreasing amplitude

    for harmonic, amplitude in zip(harmonics, amplitudes):
        # Add slight frequency modulation for more natural sound
        freq_modulation = 1 + 0.02 * np.sin(2 * np.pi * 3 * t)
        signal += amplitude * np.sin(2 * np.pi * f0 * harmonic * t * freq_modulation)

    # Add envelope (attack, sustain, release)
    envelope = np.ones_like(t)
    attack_samples = int(0.1 * sr)
    release_samples = int(0.1 * sr)

    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Release
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)

    signal = signal * envelope

    # Add slight noise for realism
    noise = np.random.normal(0, 0.01, len(signal))
    signal = signal + noise

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    # Save
    sf.write(output_path, signal, sr)
    print(f"Generated: {output_path} (F0={f0}Hz, duration={duration}s)")

    return signal, sr


def main():
    """Generate test audio files"""

    # Create test_audio directory if it doesn't exist
    os.makedirs('test_audio', exist_ok=True)

    print("Generating synthetic test audio files...")
    print("=" * 60)

    # Male voice (lower pitch)
    male_f0 = 120  # Hz (typical male: 85-180 Hz)
    generate_synthetic_voice(
        duration=3.0,
        f0=male_f0,
        sr=16000,
        output_path='test_audio/male_voice.wav'
    )

    # Female voice (higher pitch)
    female_f0 = 220  # Hz (typical female: 165-255 Hz)
    generate_synthetic_voice(
        duration=3.0,
        f0=female_f0,
        sr=16000,
        output_path='test_audio/female_voice.wav'
    )

    # Generate longer samples for stress testing
    generate_synthetic_voice(
        duration=10.0,
        f0=male_f0,
        sr=16000,
        output_path='test_audio/male_voice_long.wav'
    )

    generate_synthetic_voice(
        duration=10.0,
        f0=female_f0,
        sr=16000,
        output_path='test_audio/female_voice_long.wav'
    )

    print("=" * 60)
    print("Test audio generation complete!")
    print("\nGenerated files:")
    print("  - test_audio/male_voice.wav (3s, 120Hz)")
    print("  - test_audio/female_voice.wav (3s, 220Hz)")
    print("  - test_audio/male_voice_long.wav (10s, 120Hz)")
    print("  - test_audio/female_voice_long.wav (10s, 220Hz)")
    print("\nUse these files to test voice conversion approaches.")


if __name__ == "__main__":
    main()
