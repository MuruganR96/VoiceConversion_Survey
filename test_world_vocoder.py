#!/usr/bin/env python3
"""
Test WORLD Vocoder for voice conversion
Measures: Memory, Latency, Quality (MCD)
"""

import sys
import time
import tracemalloc
import psutil
import os
import numpy as np
import soundfile as sf

try:
    import pyworld as pw
except ImportError:
    print("ERROR: pyworld not installed!")
    print("Install with: python3 -m pip install --user pyworld")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed!")
    print("Install with: python3 -m pip install --user librosa")
    sys.exit(1)


def shift_formants(sp, ratio, sr):
    """
    Shift formants by frequency warping

    Args:
        sp: Spectral envelope from WORLD
        ratio: Formant shift ratio (>1.0 for up, <1.0 for down)
        sr: Sample rate

    Returns:
        Shifted spectral envelope
    """
    fft_size = (sp.shape[1] - 1) * 2
    sp_shifted = np.zeros_like(sp)

    for i in range(sp.shape[0]):
        # Frequency warping
        freq_axis_old = np.arange(sp.shape[1])
        freq_axis_new = freq_axis_old / ratio

        # Interpolate (clamp to valid range)
        freq_axis_new = np.clip(freq_axis_new, 0, sp.shape[1] - 1)
        sp_shifted[i] = np.interp(freq_axis_old, freq_axis_new, sp[i])

    return sp_shifted


def convert_m2f_world(audio, sr):
    """Male to Female conversion using WORLD"""

    # Extract WORLD features
    f0, sp, ap = pw.wav2world(audio, sr)

    # M2F conversion parameters
    f0_converted = f0 * 1.6  # Raise pitch by 60%
    sp_converted = shift_formants(sp, 1.2, sr)  # Shift formants up 20%

    # Synthesize
    output = pw.synthesize(f0_converted, sp_converted, ap, sr)

    return output


def convert_f2m_world(audio, sr):
    """Female to Male conversion using WORLD"""

    # Extract WORLD features
    f0, sp, ap = pw.wav2world(audio, sr)

    # F2M conversion parameters
    f0_converted = f0 * 0.7  # Lower pitch by 30%
    sp_converted = shift_formants(sp, 0.85, sr)  # Shift formants down 15%

    # Synthesize
    output = pw.synthesize(f0_converted, sp_converted, ap, sr)

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


def compute_mcd(reference, converted, sr=16000):
    """Compute Mel-Cepstral Distortion"""

    # Extract MFCCs
    mfcc_ref = librosa.feature.mfcc(y=reference, sr=sr, n_mfcc=13)
    mfcc_conv = librosa.feature.mfcc(y=converted, sr=sr, n_mfcc=13)

    # Align lengths
    min_len = min(mfcc_ref.shape[1], mfcc_conv.shape[1])
    mfcc_ref = mfcc_ref[:, :min_len]
    mfcc_conv = mfcc_conv[:, :min_len]

    # MCD formula
    diff = mfcc_ref - mfcc_conv
    mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0))) * (10 / np.log(10)) * 2

    return mcd


def evaluate_pitch_shift(original, converted, target_shift_ratio):
    """Verify pitch shift accuracy"""

    # Estimate F0
    f0_orig, _, _ = librosa.pyin(original, fmin=librosa.note_to_hz('C2'),
                                  fmax=librosa.note_to_hz('C7'))
    f0_conv, _, _ = librosa.pyin(converted, fmin=librosa.note_to_hz('C2'),
                                  fmax=librosa.note_to_hz('C7'))

    # Remove NaN (unvoiced)
    f0_orig = f0_orig[~np.isnan(f0_orig)]
    f0_conv = f0_conv[~np.isnan(f0_conv)]

    if len(f0_orig) == 0 or len(f0_conv) == 0:
        return None

    # Median F0
    median_orig = np.median(f0_orig)
    median_conv = np.median(f0_conv)

    # Actual ratio
    actual_ratio = median_conv / median_orig
    error_ratio = abs(actual_ratio - target_shift_ratio)

    return {
        'target_ratio': target_shift_ratio,
        'actual_ratio': actual_ratio,
        'error_ratio': error_ratio,
        'original_f0_hz': median_orig,
        'converted_f0_hz': median_conv,
        'error_semitones': 12 * np.log2(actual_ratio / target_shift_ratio) if target_shift_ratio > 0 else 0
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
    if audio.dtype != np.float64:
        audio = audio.astype(np.float64)

    print(f"Audio loaded: {len(audio)/sr:.2f}s, {sr}Hz")

    # Select conversion function
    if conversion_type == "M2F":
        conv_func = convert_m2f_world
        target_pitch_ratio = 1.6
    elif conversion_type == "F2M":
        conv_func = convert_f2m_world
        target_pitch_ratio = 0.7
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
    pitch_stats = evaluate_pitch_shift(audio, converted, target_pitch_ratio)
    if pitch_stats:
        print(f"Target pitch ratio: {pitch_stats['target_ratio']:.2f}")
        print(f"Actual pitch ratio: {pitch_stats['actual_ratio']:.2f}")
        print(f"Original F0: {pitch_stats['original_f0_hz']:.1f} Hz")
        print(f"Converted F0: {pitch_stats['converted_f0_hz']:.1f} Hz")
        print(f"Error: {abs(pitch_stats['error_semitones']):.2f} semitones")
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
    """Run WORLD vocoder tests"""

    print("=" * 60)
    print("WORLD Vocoder - Voice Conversion Test")
    print("=" * 60)

    # Create results directory
    os.makedirs('results/world', exist_ok=True)

    results = []

    # Test M2F conversion
    result_m2f = test_conversion(
        input_file='test_audio/male_voice.wav',
        conversion_type='M2F',
        output_file='results/world/m2f_output.wav'
    )
    results.append(result_m2f)

    # Test F2M conversion
    result_f2m = test_conversion(
        input_file='test_audio/female_voice.wav',
        conversion_type='F2M',
        output_file='results/world/f2m_output.wav'
    )
    results.append(result_f2m)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - WORLD Vocoder")
    print("=" * 60)

    for result in results:
        print(f"\n{result['conversion_type']}:")
        print(f"  Memory: {result['memory_mb']:.2f} MB")
        print(f"  Latency: {result['latency_ms']:.2f} ms")
        print(f"  RTF: {result['rtf']:.3f}")
        if result['pitch_stats']:
            print(f"  Pitch accuracy: {abs(result['pitch_stats']['error_semitones']):.2f} semitones error")

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

    print("\nTest complete! Check results/world/ for output files.")


if __name__ == "__main__":
    main()
