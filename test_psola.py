#!/usr/bin/env python3
"""
Test PSOLA for voice conversion
Measures: Memory, Latency, Quality
"""

import sys
import time
import tracemalloc
import psutil
import os
import numpy as np
import soundfile as sf

try:
    from psola import vocode
except ImportError:
    print("ERROR: psola not installed!")
    print("Install with: python3 -m pip install --user psola")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed!")
    print("Install with: python3 -m pip install --user librosa")
    sys.exit(1)


def convert_m2f_psola(audio, sr, semitone_shift=5):
    """Male to Female conversion using PSOLA"""

    # Shift pitch up by semitones
    pitch_ratio = 2 ** (semitone_shift / 12.0)
    output = vocode(audio, sample_rate=int(sr), constant_stretch=pitch_ratio)

    return output


def convert_f2m_psola(audio, sr, semitone_shift=-5):
    """Female to Male conversion using PSOLA"""

    # Shift pitch down by semitones
    pitch_ratio = 2 ** (semitone_shift / 12.0)
    output = vocode(audio, sample_rate=int(sr), constant_stretch=pitch_ratio)

    return output


def profile_memory(func, audio, sr):
    """Profile memory usage"""

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    tracemalloc.start()

    # Run conversion
    output = func(audio, sr)

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    return {
        'python_peak_mb': peak / 1024 / 1024,
        'system_delta_mb': mem_after - mem_before,
        'output': output
    }


def measure_latency(func, audio, sr, num_runs=10):
    """Measure conversion latency"""

    audio_duration = len(audio) / sr

    # Warmup
    _ = func(audio[:sr], sr)  # 1 second warmup

    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output = func(audio, sr)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)
    rtf = avg_time / audio_duration  # Real-time factor

    return {
        'latency_ms': avg_time * 1000,
        'latency_std_ms': std_time * 1000,
        'rtf': rtf,
        'audio_duration_s': audio_duration
    }


def evaluate_pitch_shift(original, converted, target_shift_semitones, sr):
    """Verify pitch shift accuracy"""

    # Estimate F0
    f0_orig, _, _ = librosa.pyin(original, sr=sr,
                                  fmin=librosa.note_to_hz('C2'),
                                  fmax=librosa.note_to_hz('C7'))
    f0_conv, _, _ = librosa.pyin(converted, sr=sr,
                                  fmin=librosa.note_to_hz('C2'),
                                  fmax=librosa.note_to_hz('C7'))

    # Remove NaN (unvoiced)
    f0_orig = f0_orig[~np.isnan(f0_orig)]
    f0_conv = f0_conv[~np.isnan(f0_conv)]

    if len(f0_orig) == 0 or len(f0_conv) == 0:
        return None

    # Median F0
    median_orig = np.median(f0_orig)
    median_conv = np.median(f0_conv)

    # Actual shift in semitones
    actual_shift = 12 * np.log2(median_conv / median_orig)
    error = abs(actual_shift - target_shift_semitones)

    return {
        'target_semitones': target_shift_semitones,
        'actual_semitones': actual_shift,
        'error_semitones': error,
        'original_f0_hz': median_orig,
        'converted_f0_hz': median_conv
    }


def test_conversion(input_file, conversion_type, output_file):
    """Test a single conversion"""

    print(f"\n{'='*60}")
    print(f"Testing: {conversion_type}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")

    # Load audio
    audio, sr = sf.read(input_file)
    print(f"Audio loaded: {len(audio)/sr:.2f}s, {sr}Hz")

    # Select conversion function
    if conversion_type == "M2F":
        conv_func = lambda a, s: convert_m2f_psola(a, s, semitone_shift=5)
        target_semitones = 5
    elif conversion_type == "F2M":
        conv_func = lambda a, s: convert_f2m_psola(a, s, semitone_shift=-5)
        target_semitones = -5
    else:
        raise ValueError(f"Unknown conversion type: {conversion_type}")

    # Memory profiling
    print("\n--- Memory Profiling ---")
    mem_stats = profile_memory(conv_func, audio, sr)
    print(f"Python peak memory: {mem_stats['python_peak_mb']:.2f} MB")
    print(f"System memory delta: {mem_stats['system_delta_mb']:.2f} MB")

    if mem_stats['system_delta_mb'] <= 2.0:
        print("✓ Memory constraint MET (≤2MB)")
    else:
        print(f"✗ Memory constraint EXCEEDED ({mem_stats['system_delta_mb']:.2f} MB > 2MB)")

    # Latency measurement
    print("\n--- Latency Measurement ---")
    lat_stats = measure_latency(conv_func, audio, sr)
    print(f"Average latency: {lat_stats['latency_ms']:.2f} ± {lat_stats['latency_std_ms']:.2f} ms")
    print(f"Real-time factor (RTF): {lat_stats['rtf']:.3f}")

    if lat_stats['rtf'] < 1.0:
        print(f"✓ Real-time capable (RTF < 1.0)")
    else:
        print(f"✗ Not real-time (RTF = {lat_stats['rtf']:.3f})")

    # Quality evaluation
    print("\n--- Quality Evaluation ---")
    converted = mem_stats['output']

    # Pitch accuracy
    pitch_stats = evaluate_pitch_shift(audio, converted, target_semitones, sr)
    if pitch_stats:
        print(f"Target pitch shift: {pitch_stats['target_semitones']:+.1f} semitones")
        print(f"Actual pitch shift: {pitch_stats['actual_semitones']:+.2f} semitones")
        print(f"Original F0: {pitch_stats['original_f0_hz']:.1f} Hz")
        print(f"Converted F0: {pitch_stats['converted_f0_hz']:.1f} Hz")
        print(f"Error: {pitch_stats['error_semitones']:.2f} semitones")
    else:
        print("Could not estimate pitch (possibly unvoiced signal)")

    # Save output
    sf.write(output_file, converted, sr)
    print(f"\n✓ Output saved: {output_file}")

    return {
        'conversion_type': conversion_type,
        'memory_mb': mem_stats['system_delta_mb'],
        'latency_ms': lat_stats['latency_ms'],
        'rtf': lat_stats['rtf'],
        'pitch_stats': pitch_stats
    }


def main():
    """Run PSOLA tests"""

    print("=" * 60)
    print("PSOLA - Voice Conversion Test")
    print("=" * 60)

    # Create results directory
    os.makedirs('results/psola', exist_ok=True)

    results = []

    # Test M2F conversion
    result_m2f = test_conversion(
        input_file='test_audio/male_voice.wav',
        conversion_type='M2F',
        output_file='results/psola/m2f_output.wav'
    )
    results.append(result_m2f)

    # Test F2M conversion
    result_f2m = test_conversion(
        input_file='test_audio/female_voice.wav',
        conversion_type='F2M',
        output_file='results/psola/f2m_output.wav'
    )
    results.append(result_f2m)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - PSOLA")
    print("=" * 60)

    for result in results:
        print(f"\n{result['conversion_type']}:")
        print(f"  Memory: {result['memory_mb']:.2f} MB")
        print(f"  Latency: {result['latency_ms']:.2f} ms")
        print(f"  RTF: {result['rtf']:.3f}")
        if result['pitch_stats']:
            print(f"  Pitch accuracy: {result['pitch_stats']['error_semitones']:.2f} semitones error")

    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)

    avg_memory = np.mean([r['memory_mb'] for r in results])
    avg_latency = np.mean([r['latency_ms'] for r in results])
    avg_rtf = np.mean([r['rtf'] for r in results])

    print(f"Average memory: {avg_memory:.2f} MB")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Average RTF: {avg_rtf:.3f}")

    print("\nConstraints Check:")
    print(f"  Memory ≤2MB: {'✓ PASS' if avg_memory <= 2.0 else '✗ FAIL'}")
    print(f"  Latency <100ms: {'✓ PASS' if avg_latency < 100 else '✗ FAIL'}")
    print(f"  RTF <1.0 (real-time): {'✓ PASS' if avg_rtf < 1.0 else '✗ FAIL'}")

    print("\nTest complete! Check results/psola/ for output files.")


if __name__ == "__main__":
    main()
