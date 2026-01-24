# Voice Conversion Testing Guide

This guide explains how to run and interpret the voice conversion tests in this repository.

---

## Quick Start

### 1. Install Dependencies

```bash
python3 -m pip install --user pyworld librosa soundfile numpy scipy psola psutil
```

### 2. Run All Tests

```bash
# Run comprehensive test suite
python3 run_all_tests.py
```

This will:
1. Test WORLD Vocoder (M2F and F2M)
2. Test PSOLA (M2F and F2M)
3. Generate results in `results/` directory
4. Create `results/TEST_RESULTS.md` report

### 3. Check Results

```bash
# View test report
cat results/TEST_RESULTS.md

# View detailed analysis
cat ACTUAL_TEST_RESULTS.md

# Listen to converted audio
# results/world/m2f_output.wav
# results/world/f2m_output.wav
# results/psola/m2f_output.wav
# results/psola/f2m_output.wav
```

---

## Individual Tests

### Test WORLD Vocoder Only

```bash
python3 test_world_vocoder.py
```

**What it measures**:
- Memory usage (Python peak and system delta)
- Latency (average over 10 runs)
- Real-time factor (RTF)
- Pitch shift accuracy

**Expected output**:
- M2F and F2M converted audio files
- Console output with metrics
- Files saved to `results/world/`

### Test PSOLA Only

```bash
python3 test_psola.py
```

**What it measures**:
- Same metrics as WORLD test
- Pitch shift accuracy

**Expected output**:
- M2F and F2M converted audio files
- Console output with metrics
- Files saved to `results/psola/`

### Generate New Test Audio

```bash
python3 generate_test_audio.py
```

**Generates**:
- `test_audio/male_voice.wav` (3s, 120Hz)
- `test_audio/female_voice.wav` (3s, 220Hz)
- `test_audio/male_voice_long.wav` (10s)
- `test_audio/female_voice_long.wav` (10s)

---

## Understanding the Metrics

### Memory Usage

**Python Peak Memory**: Memory used by Python objects (numpy arrays, etc.)
- WORLD: ~7MB (includes spectral envelope arrays)
- PSOLA: ~0.02MB (very lightweight)

**System Delta Memory**: Total process memory increase
- Includes Python interpreter overhead
- Not representative of embedded C/C++ deployment
- For 2MB target, refer to native C/C++ implementations

**Real Embedded Estimates**:
- WORLD (C++): ~1-1.5MB
- PSOLA (C): <500KB

### Latency

**Average Latency**: Mean conversion time over 10 runs
- Target: <100ms for real-time
- WORLD: ~117ms (slightly above target)
- PSOLA: ~21ms (excellent)

**Standard Deviation**: Consistency of timing
- Low std dev means predictable performance

### Real-Time Factor (RTF)

**Formula**: `RTF = processing_time / audio_duration`

**Interpretation**:
- RTF < 1.0: Real-time capable ✅
- RTF = 0.5: 2x faster than real-time
- RTF = 0.1: 10x faster than real-time

**Results**:
- WORLD: RTF ~0.04 (25x faster than real-time)
- PSOLA: RTF ~0.007 (140x faster than real-time)

Both are **highly real-time capable**.

### Pitch Accuracy

**Target vs Actual**: Semitone error

**WORLD Results**:
- M2F: 0.14 semitones error (excellent)
- F2M: 0.58 semitones error (good)

**PSOLA Results**:
- M2F/F2M: ~5 semitones error (CRITICAL BUG - pitch not shifting)

**Interpretation**:
- <1 semitone: Excellent
- 1-2 semitones: Good
- >2 semitones: Poor/Bug

---

## Test Results Interpretation

### Current Results (See ACTUAL_TEST_RESULTS.md)

#### WORLD Vocoder: ✅ RECOMMENDED

| Metric | M2F | F2M | Status |
|--------|-----|-----|--------|
| Memory | 17.77 MB | 4.84 MB | ⚠️ Python overhead |
| Latency | 123.14 ms | 112.21 ms | ⚠️ Slightly above 100ms |
| RTF | 0.041 | 0.037 | ✅ Real-time |
| Pitch Error | 0.14 st | 0.58 st | ✅ Excellent |

**Verdict**: Works excellently. For <2MB deployment, use native C++ WORLD library.

#### PSOLA: ⚠️ NEEDS FIX

| Metric | M2F | F2M | Status |
|--------|-----|-----|--------|
| Memory | 6.22 MB | 2.70 MB | ⚠️ Python overhead |
| Latency | 21.11 ms | 21.62 ms | ✅ Very fast |
| RTF | 0.007 | 0.007 | ✅ Real-time |
| Pitch Error | 5.20 st | 5.10 st | ❌ NOT WORKING |

**Verdict**: Critical bug - pitch not shifting. Needs debugging or alternative implementation.

---

## Troubleshooting

### Issue: "Module not found"

```bash
# Install missing module
python3 -m pip install --user MODULE_NAME
```

### Issue: "PSOLA pitch not shifting"

**Known issue**: The `psola` library may not be working correctly.

**Solutions**:
1. Try voice-gender-changer repository directly:
   ```bash
   cd implementations/voice-gender-changer
   python voice_changer.py ../../test_audio/male_voice.wav output.wav --semitones 5
   ```

2. Or implement TD-PSOLA from scratch (see GITHUB_REPOSITORIES.md)

### Issue: "Memory usage too high"

**This is expected in Python**. The memory measurements include:
- Python interpreter
- Numpy/librosa libraries
- Audio buffers

For true <2MB deployment:
- Use native C/C++ implementations
- WORLD: Use `implementations/World/` (build with cmake)
- PSOLA: Implement in C or use minimal library

### Issue: "Latency too high for WORLD"

**Current**: ~117ms
**Target**: <100ms

**Solutions**:
1. Use smaller FFT size in WORLD
2. Reduce sample rate (16kHz → 8kHz)
3. Optimize C++ implementation
4. Use faster CPU

---

## Using Real Voice Samples

### Download VCTK Dataset

```bash
# Download sample from VCTK
mkdir -p test_audio/real
cd test_audio/real

# Example: download a few samples (you can use wget or curl)
# VCTK: https://datashare.ed.ac.uk/handle/10283/3443
```

### Test with Real Audio

```python
# Modify test scripts to use real audio
# Replace 'test_audio/male_voice.wav' with 'test_audio/real/p225_001.wav'
```

---

## Advanced Testing

### Test with Different Parameters

**WORLD Vocoder - Adjust pitch shift**:
```python
# Edit test_world_vocoder.py
# Change: f0_converted = f0 * 1.6  # Try 1.4, 1.5, 1.7, 1.8
# Change: sp_converted = shift_formants(sp, 1.2, sr)  # Try 1.1, 1.15, 1.25
```

**PSOLA - Adjust semitones**:
```python
# Edit test_psola.py
# Change: semitone_shift=5  # Try 3, 4, 6, 7
```

### Test with Longer Audio

```bash
# Generate 30-second samples
python3 generate_test_audio.py  # Edit duration parameter

# Or use existing long samples
python3 test_world_vocoder.py  # Modify to use male_voice_long.wav
```

### Profile CPU Usage

```bash
# macOS
python3 -m cProfile -o profile.stats test_world_vocoder.py

# Analyze
python3 -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(10)"
```

---

## Testing Checklist

Before finalizing your approach:

- [ ] Run all tests successfully
- [ ] Listen to converted audio files
- [ ] Verify pitch shift is working correctly
- [ ] Check memory usage (account for Python overhead)
- [ ] Verify latency meets requirements
- [ ] Test with real voice samples (not just synthetic)
- [ ] Test with longer audio (10+ seconds)
- [ ] Profile CPU usage and identify bottlenecks
- [ ] Document any issues or limitations
- [ ] Compare subjective quality (MOS if possible)

---

## Next Steps After Testing

### If WORLD Vocoder is Chosen

1. **Build Native C++ Version**:
   ```bash
   cd implementations/World
   mkdir build && cd build
   cmake ..
   make
   ```

2. **Optimize Parameters**:
   - Fine-tune F0 and formant shift ratios
   - Test with various voice samples
   - Adjust for target voice characteristics

3. **Deploy**:
   - Integrate C++ library into your application
   - Measure actual embedded memory usage
   - Optimize for target platform (ARM, x86, etc.)

### If PSOLA is Chosen

1. **Fix Implementation**:
   - Debug current psola library
   - Or use voice-gender-changer directly
   - Or implement TD-PSOLA from scratch

2. **Optimize**:
   - Tune pitch shift parameters
   - Add formant shifting (currently missing)
   - Test quality improvements

3. **Deploy**:
   - Implement in C for minimal footprint
   - Should achieve <500KB easily

### If Neither Meets Requirements

1. **Try Quantized Neural Model**:
   - Setup TinyVC (in implementations/)
   - Train or download pretrained model
   - Apply INT8 quantization
   - Benchmark against DSP methods

2. **Hybrid Approach**:
   - Use WORLD for quality
   - Use PSOLA for speed
   - Switch based on requirements

---

## Directory Structure After Testing

```
.
├── test_audio/              # Test audio files
│   ├── male_voice.wav
│   ├── female_voice.wav
│   ├── male_voice_long.wav
│   └── female_voice_long.wav
│
├── results/                 # Test results
│   ├── world/
│   │   ├── m2f_output.wav
│   │   └── f2m_output.wav
│   ├── psola/
│   │   ├── m2f_output.wav
│   │   └── f2m_output.wav
│   ├── TEST_RESULTS.md
│   └── profile.stats (optional)
│
├── implementations/         # Cloned repositories
│   ├── World/
│   ├── voice-gender-changer/
│   └── tinyvc/
│
├── generate_test_audio.py   # Generate synthetic test samples
├── test_world_vocoder.py    # Test WORLD
├── test_psola.py            # Test PSOLA
├── run_all_tests.py         # Run all tests
│
├── ACTUAL_TEST_RESULTS.md   # Detailed analysis of test results
├── TESTING_GUIDE.md         # This file
│
├── README.md                # Project overview
├── VOICE_CONVERSION_TECHNICAL_REPORT.md
└── GITHUB_REPOSITORIES.md
```

---

## Questions?

**Q: Why is Python memory so high?**
A: Python interpreter overhead. For embedded deployment, use C/C++ implementations.

**Q: PSOLA not working?**
A: Known issue with `psola` library. Try voice-gender-changer repository or implement from scratch.

**Q: How to test with my own voice?**
A: Replace `test_audio/male_voice.wav` with your recording, then run tests.

**Q: Which method should I use?**
A: **WORLD Vocoder** - it works reliably and has excellent quality. For <2MB deployment, build the native C++ version.

---

**Last Updated**: January 24, 2026
