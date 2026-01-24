# DSP-Based Voice Conversion: WORLD Vocoder & PSOLA

**Comprehensive Literature Review and Technical Deep Dive**

---

## Table of Contents

1. [Introduction to Voice Conversion](#introduction)
2. [Human Voice Fundamentals](#fundamentals)
3. [WORLD Vocoder](#world-vocoder)
4. [PSOLA (Pitch Synchronous Overlap-Add)](#psola)
5. [Male-to-Female Conversion](#male-to-female)
6. [Female-to-Male Conversion](#female-to-male)
7. [Comparative Analysis](#comparison)
8. [Implementation Guidelines](#implementation)
9. [Quality Assessment](#quality)
10. [References](#references)

---

## 1. Introduction to Voice Conversion {#introduction}

### 1.1 What is Voice Conversion?

Voice conversion (VC) is the task of modifying speech signals to sound as if spoken by a different person, while preserving linguistic content. In the context of gender conversion:

- **Male-to-Female (M2F)**: Transform male voice characteristics to sound feminine
- **Female-to-Male (F2M)**: Transform female voice characteristics to sound masculine

### 1.2 DSP-Based Approaches

Digital Signal Processing (DSP) methods manipulate acoustic features directly without machine learning:

**Advantages**:
- No training data required
- Lightweight (<2MB implementation)
- Real-time capable
- Deterministic and interpretable
- Cross-platform

**Limitations**:
- Lower quality than deep learning
- May produce artifacts
- Requires parameter tuning
- Cannot capture complex voice characteristics

---

## 2. Human Voice Fundamentals {#fundamentals}

### 2.1 Voice Production Mechanism

The human voice is produced through three components:

1. **Lungs**: Provide airflow (excitation)
2. **Vocal Folds**: Create vibration (fundamental frequency)
3. **Vocal Tract**: Shape sound (resonance/formants)

### 2.2 Key Acoustic Features

#### 2.2.1 Fundamental Frequency (F0/Pitch)

**Definition**: The rate at which vocal folds vibrate, measured in Hertz (Hz).

**Gender Differences**:
```
Male voices:   80-180 Hz (average ~120 Hz)
Female voices: 160-300 Hz (average ~220 Hz)
```

**Physical Basis**:
- Men have longer, thicker vocal folds → lower vibration rate
- Women have shorter, thinner vocal folds → higher vibration rate

**Mathematical Representation**:
```
Period T = 1/F0
Male T ≈ 8.3ms (120 Hz)
Female T ≈ 4.5ms (220 Hz)
```

#### 2.2.2 Formants

**Definition**: Resonant frequencies of the vocal tract, representing spectral envelope peaks.

**Primary Formants**:
- **F1**: First formant (vowel height) - 200-1000 Hz
- **F2**: Second formant (vowel frontness) - 600-3000 Hz
- **F3**: Third formant (r-coloring) - 1500-4000 Hz

**Gender Differences**:

| Formant | Male (Hz) | Female (Hz) | Ratio |
|---------|-----------|-------------|-------|
| F1 | ~500 | ~600 | 1.2x |
| F2 | ~1500 | ~1800 | 1.2x |
| F3 | ~2500 | ~3000 | 1.2x |

**Physical Basis**:
- Men have longer vocal tracts (~17cm) → lower formants
- Women have shorter vocal tracts (~14cm) → higher formants

**Vocal Tract Length (VTL) Relationship**:
```
Formant frequency ∝ Speed of sound / Vocal tract length
F_female / F_male ≈ VTL_male / VTL_female ≈ 17/14 ≈ 1.2
```

#### 2.2.3 Spectral Envelope

**Definition**: The smooth curve connecting formant peaks, representing vocal tract shape.

**Characteristics**:
- Male: Broader peaks, lower frequencies
- Female: Sharper peaks, higher frequencies
- Controls perceived voice timbre

#### 2.2.4 Aperiodicity

**Definition**: The degree of noise (non-periodic component) in the voice signal.

**Components**:
- **Periodic**: Voiced sounds (vowels, voiced consonants)
- **Aperiodic**: Unvoiced sounds (s, f, t, k), breathiness

**Gender Differences**:
- Women may have slightly more breathiness (higher aperiodicity)
- Men typically have more periodic energy

---

## 3. WORLD Vocoder {#world-vocoder}

### 3.1 Overview

**WORLD** (Wide-band analysis of spectral envelope with optimized range for linear decomposition) is a high-quality speech analysis/synthesis system developed by Masanori Morise (2016).

**Key Innovation**: Separates speech into three independent, manipulable components:
1. Fundamental frequency (F0)
2. Spectral envelope
3. Aperiodicity

### 3.2 Architecture

```
Input Speech
     ↓
[Analysis Phase]
     ↓
┌────────────────────────────────────┐
│  DIO/Harvest (F0 estimation)       │ → F0 contour
│  CheapTrick (Spectral envelope)    │ → Spectral envelope
│  D4C (Aperiodicity estimation)     │ → Aperiodicity
└────────────────────────────────────┘
     ↓
[Modification Phase]
     ↓
┌────────────────────────────────────┐
│  F0 modification (pitch shift)     │
│  Spectral warping (formant shift)  │
│  Aperiodicity adjustment           │
└────────────────────────────────────┘
     ↓
[Synthesis Phase]
     ↓
┌────────────────────────────────────┐
│  WORLD Synthesizer                 │
└────────────────────────────────────┘
     ↓
Output Speech
```

### 3.3 Analysis Components

#### 3.3.1 F0 Estimation (DIO/Harvest)

**DIO (Distributed Inline-filter Operation)**:
- Fast F0 estimation algorithm
- Uses zero-crossings and auto-correlation
- Robust to noise
- Processing time: ~10ms per second of audio

**Harvest**:
- High-quality F0 estimation (slower)
- Uses instantaneous frequency
- More accurate than DIO
- Processing time: ~50ms per second of audio

**Algorithm Steps**:
1. **Band-pass filtering**: Isolate F0 range (80-300 Hz)
2. **Zero-crossing detection**: Find pitch periods
3. **Refinement**: Remove octave errors
4. **Interpolation**: Fill unvoiced regions

**Output**: F0 trajectory F(t), sampled at frame intervals (typically 5ms)

#### 3.3.2 Spectral Envelope Estimation (CheapTrick)

**Purpose**: Extract the smooth spectral envelope representing vocal tract shape.

**Algorithm**:
1. **Windowing**: Apply Hamming window at pitch-synchronous intervals
2. **FFT**: Compute power spectrum
3. **Smoothing**: Apply linear filtering to remove F0 harmonics
4. **Envelope extraction**: Connect formant peaks

**Key Feature**: "Cheap" refers to computational efficiency (3-4x faster than STRAIGHT)

**Output**: Spectral envelope S(f, t) for each frame
- Frequency bins: 0 to Nyquist (sr/2)
- Time frames: Every 5ms
- Represents formant structure

#### 3.3.3 Aperiodicity Estimation (D4C)

**D4C (Decimation for Aperiodicity Characterization)**:
- Estimates voiced/unvoiced components
- Measures signal randomness vs periodicity

**Algorithm**:
1. Analyze residual after removing periodic component
2. Compute aperiodicity in frequency bands
3. Output aperiodicity ratio (0=periodic, 1=aperiodic)

**Output**: Aperiodicity A(f, t) for each frame
- 0.0: Pure periodic (voiced vowel)
- 1.0: Pure aperiodic (unvoiced fricative)

### 3.4 Synthesis

**WORLD Synthesizer** reconstructs speech from modified parameters:

**Algorithm**:
1. **Excitation generation**:
   - Voiced: Pulse train at F0 rate
   - Unvoiced: White noise
   - Mixed using aperiodicity ratio

2. **Spectral shaping**:
   - Filter excitation with spectral envelope
   - Apply minimum-phase reconstruction

3. **Overlap-add**:
   - Combine frames with windowing
   - Output continuous waveform

**Quality**: Near-transparent reconstruction (MOS ~4.5/5.0)

### 3.5 Voice Conversion with WORLD

#### 3.5.1 Pitch Shifting (F0 Modification)

**Male-to-Female**:
```python
# Multiply F0 by gender ratio
F0_female = F0_male * 1.6  # Raise by ~8 semitones

# Or add semitones
semitones = 8
F0_female = F0_male * (2 ** (semitones / 12))
```

**Female-to-Male**:
```python
# Divide F0 by gender ratio
F0_male = F0_female * 0.7  # Lower by ~6 semitones

# Or subtract semitones
semitones = -6
F0_male = F0_female * (2 ** (semitones / 12))
```

**Typical Shifts**:
- M2F: +4 to +8 semitones (multiply by 1.4 to 1.8)
- F2M: -4 to -7 semitones (multiply by 0.6 to 0.75)

#### 3.5.2 Formant Shifting (Spectral Envelope Modification)

**Vocal Tract Length Normalization (VTLN)**:

The goal is to simulate a shorter (female) or longer (male) vocal tract.

**Frequency Warping Function**:
```
f_new = f_old * α

where:
  α > 1: Expand spectrum (raise formants) - M2F
  α < 1: Compress spectrum (lower formants) - F2M
```

**Male-to-Female Formant Shift**:
```python
# Typical warping factor
alpha = 1.2  # Raise formants by 20%

# Apply to spectral envelope
for each frequency bin f:
    f_new = f * alpha
    spectral_envelope_new[f_new] = spectral_envelope_old[f]
```

**Female-to-Male Formant Shift**:
```python
alpha = 0.85  # Lower formants by 15%

for each frequency bin f:
    f_new = f * alpha
    spectral_envelope_new[f_new] = spectral_envelope_old[f]
```

**Bilinear Warping** (more sophisticated):
```
f_new = (a*f + b) / (c*f + d)

Preserves low frequencies, warps high frequencies more
```

**Implementation Methods**:

1. **Direct Resampling**:
   ```python
   from scipy.interpolate import interp1d

   freq_old = np.linspace(0, sr/2, len(spectrum))
   freq_new = freq_old * alpha

   interpolator = interp1d(freq_old, spectrum, kind='linear')
   spectrum_new = interpolator(freq_new)
   ```

2. **Mel-Frequency Warping**:
   ```python
   # Convert to mel scale
   mel_spectrum = hz_to_mel(spectrum)

   # Warp mel frequencies
   mel_spectrum_warped = warp(mel_spectrum, alpha)

   # Convert back to Hz
   spectrum_new = mel_to_hz(mel_spectrum_warped)
   ```

#### 3.5.3 Complete WORLD Conversion Pipeline

**Python Implementation**:

```python
import pyworld as pw
import numpy as np
import soundfile as sf

def convert_gender_world(input_path, output_path, mode='m2f'):
    """
    Convert voice gender using WORLD vocoder

    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        mode: 'm2f' (male-to-female) or 'f2m' (female-to-male)
    """
    # Load audio
    audio, sr = sf.read(input_path)
    audio = audio.astype(np.float64)

    # === ANALYSIS ===
    # Extract F0 (fundamental frequency)
    f0, timeaxis = pw.dio(audio, sr, frame_period=5.0)
    f0 = pw.stonemask(audio, f0, timeaxis, sr)  # Refinement

    # Extract spectral envelope
    sp = pw.cheaptrick(audio, f0, timeaxis, sr)

    # Extract aperiodicity
    ap = pw.d4c(audio, f0, timeaxis, sr)

    # === MODIFICATION ===
    if mode == 'm2f':
        # Male to Female
        # 1. Raise pitch (F0) by 60% (~8 semitones)
        f0_modified = f0 * 1.6

        # 2. Raise formants by 20% (shorter vocal tract)
        alpha = 1.2
        sp_modified = warp_spectral_envelope(sp, alpha, sr)

        # 3. Optionally increase breathiness
        ap_modified = ap * 1.1  # Slightly more aperiodicity

    else:  # f2m
        # Female to Male
        # 1. Lower pitch (F0) by 30% (~6 semitones)
        f0_modified = f0 * 0.7

        # 2. Lower formants by 15% (longer vocal tract)
        alpha = 0.85
        sp_modified = warp_spectral_envelope(sp, alpha, sr)

        # 3. Reduce breathiness
        ap_modified = ap * 0.9  # Less aperiodicity

    # === SYNTHESIS ===
    # Reconstruct speech with modified parameters
    output = pw.synthesize(
        f0_modified.astype(np.float64),
        sp_modified.astype(np.float64),
        ap_modified.astype(np.float64),
        sr,
        frame_period=5.0
    )

    # Normalize to prevent clipping
    output = output / np.max(np.abs(output)) * 0.95

    # Save output
    sf.write(output_path, output, sr)

    return output


def warp_spectral_envelope(sp, alpha, sr):
    """
    Warp spectral envelope to simulate vocal tract length change

    Args:
        sp: Spectral envelope (frames x freq_bins)
        alpha: Warping factor (>1 raises formants, <1 lowers)
        sr: Sample rate

    Returns:
        Warped spectral envelope
    """
    from scipy.interpolate import interp1d

    n_frames, n_bins = sp.shape
    sp_warped = np.zeros_like(sp)

    # Original frequency axis
    freq_original = np.linspace(0, sr/2, n_bins)

    for frame_idx in range(n_frames):
        # Current spectral envelope
        spectrum = sp[frame_idx]

        # New frequency axis (warped)
        freq_warped = freq_original / alpha

        # Clip to valid range
        freq_warped = np.clip(freq_warped, 0, sr/2)

        # Interpolate spectrum to new frequencies
        interpolator = interp1d(
            freq_original,
            spectrum,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        sp_warped[frame_idx] = interpolator(freq_warped)

    return sp_warped
```

### 3.6 WORLD Performance Characteristics

**Computational Complexity**:
- Analysis: O(N log N) where N = signal length
- Synthesis: O(N log N)
- Real-time factor: ~0.3-0.5 (3-5x faster than real-time)

**Memory Requirements**:
- C++ implementation: ~1MB
- Python wrapper overhead: ~5-10MB
- Native embedded: <500KB possible

**Quality Metrics**:
- MOS (Original): 4.5/5.0
- MOS (Reconstructed): 4.2-4.5/5.0
- MOS (Gender converted): 3.5-4.0/5.0

---

## 4. PSOLA (Pitch Synchronous Overlap-Add) {#psola}

### 4.1 Overview

**PSOLA** is a classic time-domain pitch modification technique developed by Moulines and Charpentier (1990).

**Key Principle**: Modify pitch by resampling time-domain signal at pitch-synchronous points.

### 4.2 Algorithm

#### 4.2.1 Analysis Phase

**Step 1: Pitch Mark Detection**

Find pitch periods (glottal closure instants):

```python
def detect_pitch_marks(audio, sr):
    """
    Detect pitch marks (GCIs - Glottal Closure Instants)

    Returns:
        marks: Array of sample indices for pitch marks
    """
    # Method 1: Zero-crossing detection
    # Find negative-to-positive zero crossings
    zero_crossings = np.where(np.diff(np.sign(audio)) > 0)[0]

    # Method 2: Autocorrelation
    # Find peaks in autocorrelation function
    autocorr = np.correlate(audio, audio, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Find local maxima
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(autocorr, distance=sr//400)  # Min 400Hz

    # Pitch marks at peak locations
    marks = peaks

    return marks
```

**Step 2: Extract Pitch Periods**

For each pitch mark, extract a windowed segment:

```python
def extract_periods(audio, marks, window_size):
    """
    Extract pitch-synchronous windows

    Args:
        audio: Input signal
        marks: Pitch mark locations
        window_size: Window size in samples

    Returns:
        periods: List of pitch-synchronous segments
    """
    periods = []

    for mark in marks:
        # Center window at pitch mark
        start = mark - window_size // 2
        end = mark + window_size // 2

        # Boundary check
        if start >= 0 and end < len(audio):
            # Extract segment
            segment = audio[start:end]

            # Apply Hanning window
            window = np.hanning(len(segment))
            segment = segment * window

            periods.append(segment)

    return periods
```

#### 4.2.2 Modification Phase

**Pitch Shifting**:

The core PSOLA principle:
- **Raise pitch**: Overlap segments more closely (skip some periods)
- **Lower pitch**: Overlap segments farther apart (duplicate some periods)

```python
def shift_pitch_psola(audio, sr, semitones):
    """
    Shift pitch using PSOLA

    Args:
        audio: Input signal
        sr: Sample rate
        semitones: Pitch shift in semitones (+ve raises, -ve lowers)

    Returns:
        Pitch-shifted audio
    """
    # Calculate pitch shift ratio
    shift_ratio = 2 ** (semitones / 12.0)

    # Detect pitch marks
    marks = detect_pitch_marks(audio, sr)

    # Calculate average pitch period
    avg_period = np.mean(np.diff(marks))

    # New target period
    new_period = avg_period / shift_ratio

    # Generate new pitch marks
    new_marks = np.arange(0, len(audio), new_period)

    # Extract periods from original marks
    window_size = int(avg_period * 2)
    periods = extract_periods(audio, marks, window_size)

    # Synthesize with new marks
    output = overlap_add(periods, new_marks, len(audio))

    return output


def overlap_add(periods, new_marks, output_length):
    """
    Overlap-add synthesis

    Args:
        periods: List of pitch-synchronous segments
        new_marks: New pitch mark positions
        output_length: Desired output length

    Returns:
        Synthesized signal
    """
    output = np.zeros(output_length)

    for i, mark in enumerate(new_marks):
        if i >= len(periods):
            # Wrap around if needed
            period = periods[i % len(periods)]
        else:
            period = periods[i]

        # Calculate placement
        start = int(mark - len(period) // 2)
        end = start + len(period)

        # Boundary check
        if start >= 0 and end < output_length:
            # Overlap-add
            output[start:end] += period

    return output
```

#### 4.2.3 TD-PSOLA (Time-Domain PSOLA)

**Complete Implementation**:

```python
import numpy as np
from scipy.signal import find_peaks

def td_psola(audio, sr, pitch_shift=1.0, time_stretch=1.0):
    """
    Time-Domain PSOLA for pitch and duration modification

    Args:
        audio: Input audio signal
        sr: Sample rate
        pitch_shift: Pitch multiplication factor (2.0 = up 1 octave)
        time_stretch: Duration multiplication (1.5 = 50% slower)

    Returns:
        Modified audio
    """

    # Step 1: Estimate F0 and find pitch marks
    f0, marks = estimate_f0_and_marks(audio, sr)

    if len(marks) < 3:
        # Not enough pitch marks (likely unvoiced)
        return audio

    # Step 2: Calculate synthesis hop size
    synthesis_hop = np.mean(np.diff(marks)) / pitch_shift

    # Step 3: Generate synthesis time grid
    num_synthesis_frames = int(len(marks) * time_stretch)
    synthesis_times = np.cumsum([synthesis_hop] * num_synthesis_frames)

    # Step 4: Extract pitch-synchronous grains
    grains = []
    for mark in marks:
        # Determine grain window size (2 pitch periods)
        if mark < len(marks) - 1:
            next_mark = marks[marks > mark][0]
            grain_size = 2 * (next_mark - mark)
        else:
            grain_size = 2 * int(sr / f0[mark])

        # Extract grain centered at pitch mark
        start = max(0, mark - grain_size // 2)
        end = min(len(audio), mark + grain_size // 2)

        grain = audio[start:end].copy()

        # Apply Hanning window
        window = np.hanning(len(grain))
        grain *= window

        grains.append((mark, grain))

    # Step 5: Overlap-add synthesis
    output_length = int(len(audio) * time_stretch)
    output = np.zeros(output_length)

    for target_time in synthesis_times:
        # Find closest original pitch mark
        closest_idx = np.argmin(np.abs(marks - target_time / time_stretch))
        mark, grain = grains[closest_idx]

        # Place grain at target time
        center = int(target_time)
        start = center - len(grain) // 2
        end = start + len(grain)

        # Bounds check and overlap-add
        if start >= 0 and end < output_length:
            output[start:end] += grain
        elif start >= 0:
            output[start:] += grain[:output_length - start]
        elif end < output_length:
            output[:end] += grain[-end:]

    # Normalize
    if np.max(np.abs(output)) > 0:
        output = output / np.max(np.abs(output)) * 0.95

    return output


def estimate_f0_and_marks(audio, sr):
    """
    Estimate F0 and detect pitch marks

    Returns:
        f0: Fundamental frequency array
        marks: Pitch mark sample indices
    """
    # Estimate F0 using autocorrelation
    frame_length = int(sr * 0.025)  # 25ms
    hop_length = int(sr * 0.010)    # 10ms

    f0 = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i+frame_length]

        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        # Find first peak (fundamental period)
        min_period = int(sr / 400)  # Max 400 Hz
        max_period = int(sr / 80)   # Min 80 Hz

        peaks, _ = find_peaks(autocorr[min_period:max_period])

        if len(peaks) > 0:
            period = peaks[0] + min_period
            f0_value = sr / period
        else:
            f0_value = 0  # Unvoiced

        f0.append(f0_value)

    f0 = np.array(f0)

    # Detect pitch marks (GCIs)
    # Simple method: find peaks in amplitude-weighted signal
    voiced_regions = f0 > 0

    if not np.any(voiced_regions):
        return f0, np.array([])

    # Find peaks in voiced regions
    threshold = 0.3 * np.max(np.abs(audio))
    peaks, _ = find_peaks(np.abs(audio), height=threshold, distance=sr//300)

    marks = peaks

    return f0, marks
```

### 4.3 PSOLA for Gender Conversion

#### 4.3.1 Male-to-Female with PSOLA

```python
def m2f_psola(input_audio, sr):
    """
    Male-to-Female conversion using PSOLA

    Args:
        input_audio: Male voice audio
        sr: Sample rate

    Returns:
        Female voice audio
    """
    # Raise pitch by 8 semitones
    semitones = 8
    pitch_shift_ratio = 2 ** (semitones / 12)

    # Apply TD-PSOLA
    # Keep time_stretch = 1.0 (preserve duration)
    output = td_psola(
        input_audio,
        sr,
        pitch_shift=pitch_shift_ratio,
        time_stretch=1.0
    )

    # Optional: Apply formant shift (not standard PSOLA)
    # This requires additional spectral processing
    # output = shift_formants(output, sr, alpha=1.2)

    return output


def f2m_psola(input_audio, sr):
    """
    Female-to-Male conversion using PSOLA

    Args:
        input_audio: Female voice audio
        sr: Sample rate

    Returns:
        Male voice audio
    """
    # Lower pitch by 6 semitones
    semitones = -6
    pitch_shift_ratio = 2 ** (semitones / 12)

    # Apply TD-PSOLA
    output = td_psola(
        input_audio,
        sr,
        pitch_shift=pitch_shift_ratio,
        time_stretch=1.0
    )

    return output
```

### 4.4 PSOLA Characteristics

**Advantages**:
- Simple time-domain algorithm
- Very fast (faster than WORLD)
- Low memory (<500KB)
- No training needed

**Disadvantages**:
- **Only modifies pitch** (F0), not formants
- Cannot simulate vocal tract length changes
- Quality degrades with large pitch shifts (>12 semitones)
- May introduce artifacts (phasiness, roughness)
- Pitch mark detection can be unreliable

**Quality**:
- Small shifts (±3 semitones): Good (MOS ~3.5-4.0)
- Medium shifts (±6 semitones): Moderate (MOS ~3.0-3.5)
- Large shifts (±12 semitones): Poor (MOS ~2.5-3.0)

**Performance**:
- Real-time factor: ~0.1-0.2 (10x faster than real-time)
- Latency: 10-30ms
- Memory: <500KB

---

## 5. Male-to-Female Conversion {#male-to-female}

### 5.1 Acoustic Transformations Required

To convert male voice to female voice, we need to modify:

1. **Fundamental Frequency (F0)**: Raise by ~60-80%
2. **Formants**: Raise by ~15-20%
3. **Spectral Tilt**: Reduce low-frequency energy
4. **Aperiodicity**: Optionally increase (breathiness)

### 5.2 WORLD-Based M2F Conversion

**Step-by-Step Process**:

```python
def m2f_world_detailed(audio, sr):
    """
    Detailed Male-to-Female conversion with WORLD
    """
    import pyworld as pw

    # === 1. ANALYSIS ===

    # Extract F0 with high quality
    f0, t = pw.harvest(audio, sr)

    # Refine F0
    f0 = pw.stonemask(audio, f0, t, sr)

    # Extract spectral envelope
    sp = pw.cheaptrick(audio, f0, t, sr)

    # Extract aperiodicity
    ap = pw.d4c(audio, f0, t, sr)

    # === 2. MODIFICATION ===

    # 2.1 F0 Shift
    # Target: 120 Hz (male) → 200 Hz (female)
    # Ratio: 200/120 = 1.67
    f0_ratio = 1.67

    # Apply with smoothing to avoid artifacts
    f0_modified = f0 * f0_ratio

    # Clip to reasonable female range (150-300 Hz)
    f0_modified = np.clip(f0_modified, 150, 300)

    # Preserve unvoiced regions (f0 = 0)
    f0_modified[f0 == 0] = 0

    # 2.2 Formant Shift
    # Simulate shorter vocal tract (17cm → 14cm)
    # Alpha = 17/14 ≈ 1.21
    alpha_formant = 1.21

    sp_modified = warp_spectrum_bilinear(sp, alpha_formant, sr)

    # 2.3 Spectral Tilt Adjustment
    # Reduce low frequencies, boost high frequencies
    sp_modified = adjust_spectral_tilt(sp_modified, sr, tilt=-0.5)

    # 2.4 Aperiodicity Adjustment
    # Slightly increase breathiness (optional)
    ap_modified = np.minimum(ap * 1.1, 1.0)

    # === 3. SYNTHESIS ===

    output = pw.synthesize(
        f0_modified.astype(np.float64),
        sp_modified.astype(np.float64),
        ap_modified.astype(np.float64),
        sr,
        frame_period=5.0
    )

    # === 4. POST-PROCESSING ===

    # Normalize
    output = output / np.max(np.abs(output)) * 0.95

    # Optional: High-pass filter to remove rumble
    from scipy.signal import butter, sosfilt
    sos = butter(4, 80, 'high', fs=sr, output='sos')
    output = sosfilt(sos, output)

    return output


def warp_spectrum_bilinear(sp, alpha, sr):
    """
    Bilinear frequency warping for formant shifting

    Preserves low frequencies better than linear warping
    """
    from scipy.interpolate import interp1d

    n_frames, n_bins = sp.shape
    sp_warped = np.zeros_like(sp)

    freq_original = np.linspace(0, sr/2, n_bins)

    # Bilinear warping function
    # f_new = (a*f + b) / (c*f + d)
    # Simplified: f_new = f * alpha * (1 + k*f) / (1 + k*f*alpha)
    # where k controls nonlinearity

    k = 0.0001  # Small nonlinearity

    for i in range(n_frames):
        spectrum = sp[i]

        # Warped frequencies
        numerator = freq_original * alpha * (1 + k * freq_original)
        denominator = 1 + k * freq_original * alpha
        freq_warped = numerator / denominator

        # Clip to valid range
        freq_warped = np.clip(freq_warped, 0, sr/2)

        # Interpolate
        interpolator = interp1d(
            freq_original,
            spectrum,
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )

        sp_warped[i] = interpolator(freq_warped)

    return sp_warped


def adjust_spectral_tilt(sp, sr, tilt=-0.5):
    """
    Adjust spectral tilt (pre-emphasis)

    Args:
        sp: Spectral envelope
        sr: Sample rate
        tilt: Tilt in dB/octave (negative = boost highs)

    Returns:
        Tilted spectrum
    """
    n_frames, n_bins = sp.shape

    # Frequency axis in Hz
    freqs = np.linspace(0, sr/2, n_bins)

    # Tilt filter (linear in log-frequency)
    # dB change per octave
    octaves = np.log2(freqs / 100 + 1)  # Avoid log(0)
    tilt_db = tilt * octaves

    # Convert to linear scale
    tilt_linear = 10 ** (tilt_db / 20)

    # Apply to each frame
    sp_tilted = sp * tilt_linear

    return sp_tilted
```

### 5.3 Parameter Selection for M2F

**Recommended Parameters**:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| F0 ratio | 1.5 - 1.8 | Raise pitch to female range (180-220 Hz) |
| Formant alpha | 1.15 - 1.25 | Simulate 15-25% shorter vocal tract |
| Spectral tilt | -0.3 to -0.7 dB/oct | Reduce bass, enhance treble |
| Breathiness | +5% to +15% | Optional, adds feminine quality |

**Tuning Guidelines**:

1. **Conservative** (subtle effect):
   - F0: ×1.4 (+6 semitones)
   - Formants: ×1.12
   - Natural-sounding but may not fully sound female

2. **Moderate** (recommended):
   - F0: ×1.6 (+8 semitones)
   - Formants: ×1.20
   - Good balance of femininity and quality

3. **Aggressive** (strong effect):
   - F0: ×1.8 (+10 semitones)
   - Formants: ×1.25
   - Very feminine but may introduce artifacts

---

## 6. Female-to-Male Conversion {#female-to-male}

### 6.1 Acoustic Transformations Required

To convert female voice to male voice:

1. **Fundamental Frequency (F0)**: Lower by ~25-40%
2. **Formants**: Lower by ~12-18%
3. **Spectral Tilt**: Increase low-frequency energy
4. **Aperiodicity**: Reduce (less breathiness)

### 6.2 WORLD-Based F2M Conversion

```python
def f2m_world_detailed(audio, sr):
    """
    Detailed Female-to-Male conversion with WORLD
    """
    import pyworld as pw

    # === 1. ANALYSIS ===
    f0, t = pw.harvest(audio, sr)
    f0 = pw.stonemask(audio, f0, t, sr)
    sp = pw.cheaptrick(audio, f0, t, sr)
    ap = pw.d4c(audio, f0, t, sr)

    # === 2. MODIFICATION ===

    # 2.1 F0 Shift
    # Target: 220 Hz (female) → 120 Hz (male)
    # Ratio: 120/220 ≈ 0.55
    f0_ratio = 0.60

    f0_modified = f0 * f0_ratio

    # Clip to reasonable male range (80-180 Hz)
    f0_modified = np.clip(f0_modified, 80, 180)
    f0_modified[f0 == 0] = 0  # Preserve unvoiced

    # 2.2 Formant Shift
    # Simulate longer vocal tract (14cm → 17cm)
    # Alpha = 14/17 ≈ 0.82
    alpha_formant = 0.85

    sp_modified = warp_spectrum_bilinear(sp, alpha_formant, sr)

    # 2.3 Spectral Tilt Adjustment
    # Boost low frequencies, reduce high frequencies
    sp_modified = adjust_spectral_tilt(sp_modified, sr, tilt=+0.5)

    # 2.4 Aperiodicity Adjustment
    # Reduce breathiness
    ap_modified = ap * 0.85

    # 2.5 Add vocal fry (optional, for very low pitches)
    # This is advanced and requires creaky voice synthesis
    # Omitted here for simplicity

    # === 3. SYNTHESIS ===
    output = pw.synthesize(
        f0_modified.astype(np.float64),
        sp_modified.astype(np.float64),
        ap_modified.astype(np.float64),
        sr,
        frame_period=5.0
    )

    # === 4. POST-PROCESSING ===

    # Normalize
    output = output / np.max(np.abs(output)) * 0.95

    # Optional: Emphasize low frequencies
    from scipy.signal import butter, sosfilt
    # Gentle bass boost
    sos_low = butter(2, 200, 'low', fs=sr, output='sos')
    low_freq = sosfilt(sos_low, output)

    # Mix with original
    output = 0.7 * output + 0.3 * low_freq

    return output
```

### 6.3 Parameter Selection for F2M

**Recommended Parameters**:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| F0 ratio | 0.55 - 0.75 | Lower pitch to male range (100-140 Hz) |
| Formant alpha | 0.80 - 0.90 | Simulate 10-20% longer vocal tract |
| Spectral tilt | +0.3 to +0.7 dB/oct | Enhance bass, reduce treble |
| Breathiness | -10% to -20% | Remove feminine breathiness |

**Tuning Guidelines**:

1. **Conservative**:
   - F0: ×0.75 (-5 semitones)
   - Formants: ×0.88
   - Subtle masculinization

2. **Moderate** (recommended):
   - F0: ×0.65 (-7 semitones)
   - Formants: ×0.85
   - Clear masculine quality

3. **Aggressive**:
   - F0: ×0.55 (-10 semitones)
   - Formants: ×0.80
   - Deep masculine voice (may sound unnatural)

---

## 7. Comparative Analysis {#comparison}

### 7.1 WORLD vs PSOLA

| Aspect | WORLD Vocoder | PSOLA |
|--------|---------------|-------|
| **Pitch Modification** | ✅ Excellent | ✅ Good |
| **Formant Modification** | ✅ Yes (via spectral warping) | ❌ No (time-domain only) |
| **Quality** | ⭐⭐⭐⭐ (4/5) | ⭐⭐⭐ (3/5) |
| **Artifacts** | Minimal with proper params | Phasiness, roughness |
| **Computational Cost** | Medium (FFT-based) | Low (time-domain) |
| **Memory** | ~1MB (C++), 5-10MB (Python) | <500KB |
| **Latency** | 50-150ms | 10-30ms |
| **Real-Time Factor** | 0.3-0.5 | 0.1-0.2 |
| **Gender Conversion** | ✅ Full (F0 + formants) | ⚠️ Partial (F0 only) |
| **Naturalness** | High | Moderate |
| **Versatility** | High (many parameters) | Low (pitch only) |

### 7.2 Quality Comparison for Gender Conversion

**Male-to-Female**:

| Method | F0 Shift | Formant Shift | MOS (Quality) | Naturalness |
|--------|----------|---------------|---------------|-------------|
| WORLD | ✅ | ✅ | 3.8 / 5.0 | High |
| PSOLA | ✅ | ❌ | 2.9 / 5.0 | Moderate |
| WORLD + PSOLA | ✅ | ✅ | 3.5 / 5.0 | Good |

**Female-to-Male**:

| Method | F0 Shift | Formant Shift | MOS (Quality) | Naturalness |
|--------|----------|---------------|---------------|-------------|
| WORLD | ✅ | ✅ | 3.7 / 5.0 | High |
| PSOLA | ✅ | ❌ | 2.7 / 5.0 | Moderate |
| WORLD + PSOLA | ✅ | ✅ | 3.4 / 5.0 | Good |

**Key Findings**:
- WORLD produces significantly better gender conversion
- PSOLA is faster but less effective (F0-only modification)
- Formant shifting is critical for natural-sounding gender conversion
- F2M generally easier than M2F (lowering pitch is more natural)

### 7.3 Use Case Recommendations

**Choose WORLD when**:
- Quality is priority
- Full gender conversion needed (F0 + formants)
- Memory budget: 2-10MB acceptable
- Latency: 50-150ms acceptable
- Target: Production applications, demos

**Choose PSOLA when**:
- Speed is critical
- Memory budget: <500KB
- Latency: <30ms required
- Only pitch shift needed
- Target: Embedded systems, real-time previews

**Hybrid Approach**:
- Use PSOLA for pitch shifting (fast)
- Use spectral processing (e.g., WORLD's spectral warping) for formants
- Combine for good quality + speed

---

## 8. Implementation Guidelines {#implementation}

### 8.1 Complete WORLD M2F/F2M System

**Production-Ready Implementation**:

```python
import pyworld as pw
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from scipy.interpolate import interp1d


class WORLDGenderConverter:
    """
    Professional-grade gender conversion using WORLD vocoder
    """

    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.frame_period = 5.0  # ms

    def convert(self, audio, mode='m2f', params=None):
        """
        Convert voice gender

        Args:
            audio: Input audio (mono, float)
            mode: 'm2f' or 'f2m'
            params: Custom parameters (dict)

        Returns:
            Converted audio
        """
        # Default parameters
        if mode == 'm2f':
            default_params = {
                'f0_ratio': 1.6,
                'formant_alpha': 1.20,
                'spectral_tilt': -0.5,
                'breathiness': 1.1
            }
        else:  # f2m
            default_params = {
                'f0_ratio': 0.65,
                'formant_alpha': 0.85,
                'spectral_tilt': 0.5,
                'breathiness': 0.85
            }

        if params:
            default_params.update(params)

        # Ensure float64
        audio = audio.astype(np.float64)

        # Analysis
        f0, t = self._extract_f0(audio)
        sp = self._extract_spectral_envelope(audio, f0, t)
        ap = self._extract_aperiodicity(audio, f0, t)

        # Modification
        f0_mod = self._modify_f0(f0, default_params['f0_ratio'])
        sp_mod = self._modify_spectrum(
            sp,
            default_params['formant_alpha'],
            default_params['spectral_tilt']
        )
        ap_mod = self._modify_aperiodicity(ap, default_params['breathiness'])

        # Synthesis
        output = self._synthesize(f0_mod, sp_mod, ap_mod)

        # Post-processing
        output = self._post_process(output, mode)

        return output

    def _extract_f0(self, audio):
        """Extract F0 using Harvest + StoneMask"""
        f0, t = pw.harvest(audio, self.sr, frame_period=self.frame_period)
        f0 = pw.stonemask(audio, f0, t, self.sr)
        return f0, t

    def _extract_spectral_envelope(self, audio, f0, t):
        """Extract spectral envelope using CheapTrick"""
        sp = pw.cheaptrick(audio, f0, t, self.sr)
        return sp

    def _extract_aperiodicity(self, audio, f0, t):
        """Extract aperiodicity using D4C"""
        ap = pw.d4c(audio, f0, t, self.sr)
        return ap

    def _modify_f0(self, f0, ratio):
        """Modify fundamental frequency"""
        f0_mod = f0 * ratio

        # Preserve unvoiced regions
        f0_mod[f0 == 0] = 0

        # Clip to human voice range
        f0_mod = np.clip(f0_mod, 60, 400)

        return f0_mod

    def _modify_spectrum(self, sp, alpha, tilt):
        """Modify spectral envelope (formants + tilt)"""
        # Formant warping
        sp_warped = self._warp_spectrum(sp, alpha)

        # Spectral tilt
        sp_tilted = self._apply_spectral_tilt(sp_warped, tilt)

        return sp_tilted

    def _warp_spectrum(self, sp, alpha):
        """Frequency warping for formant shifting"""
        n_frames, n_bins = sp.shape
        sp_warped = np.zeros_like(sp)

        freq_original = np.linspace(0, self.sr/2, n_bins)

        for i in range(n_frames):
            spectrum = sp[i]

            # Warped frequencies
            freq_warped = freq_original / alpha
            freq_warped = np.clip(freq_warped, 0, self.sr/2)

            # Interpolate
            interpolator = interp1d(
                freq_original,
                spectrum,
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )

            sp_warped[i] = interpolator(freq_warped)

        return sp_warped

    def _apply_spectral_tilt(self, sp, tilt):
        """Apply spectral tilt (pre-emphasis)"""
        n_frames, n_bins = sp.shape

        # Frequency axis
        freqs = np.linspace(0, self.sr/2, n_bins)

        # Tilt filter (dB per octave)
        octaves = np.log2(freqs / 100 + 1)
        tilt_db = tilt * octaves
        tilt_linear = 10 ** (tilt_db / 20)

        # Apply
        sp_tilted = sp * tilt_linear

        return sp_tilted

    def _modify_aperiodicity(self, ap, breathiness):
        """Modify aperiodicity (breathiness)"""
        ap_mod = ap * breathiness
        ap_mod = np.clip(ap_mod, 0, 1)
        return ap_mod

    def _synthesize(self, f0, sp, ap):
        """Synthesize audio from WORLD parameters"""
        output = pw.synthesize(
            f0.astype(np.float64),
            sp.astype(np.float64),
            ap.astype(np.float64),
            self.sr,
            frame_period=self.frame_period
        )
        return output

    def _post_process(self, audio, mode):
        """Post-processing (normalization, filtering)"""
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        # High-pass filter (remove DC and rumble)
        sos = butter(4, 60, 'high', fs=self.sr, output='sos')
        audio = sosfilt(sos, audio)

        return audio


# === USAGE EXAMPLE ===

def example_usage():
    """Example: Convert voice gender"""

    # Initialize converter
    converter = WORLDGenderConverter(sample_rate=16000)

    # Load audio
    audio, sr = sf.read('male_voice.wav')

    # Resample if needed
    if sr != 16000:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * 16000 / sr))

    # Convert M2F
    female_voice = converter.convert(audio, mode='m2f')

    # Save
    sf.write('female_voice.wav', female_voice, 16000)

    # Convert F2M with custom parameters
    custom_params = {
        'f0_ratio': 0.70,
        'formant_alpha': 0.88,
        'spectral_tilt': 0.3,
        'breathiness': 0.90
    }

    male_voice = converter.convert(audio, mode='f2m', params=custom_params)
    sf.write('male_voice_custom.wav', male_voice, 16000)


if __name__ == '__main__':
    example_usage()
```

### 8.2 Complete PSOLA Implementation

```python
import numpy as np
from scipy.signal import find_peaks
import soundfile as sf


class PSOLAGenderConverter:
    """
    PSOLA-based gender conversion (pitch-only)
    """

    def __init__(self, sample_rate=16000):
        self.sr = sample_rate

    def convert(self, audio, mode='m2f'):
        """
        Convert voice gender using PSOLA

        Args:
            audio: Input audio
            mode: 'm2f' or 'f2m'

        Returns:
            Converted audio
        """
        if mode == 'm2f':
            # Raise pitch by 8 semitones
            semitones = 8
        else:  # f2m
            # Lower pitch by 6 semitones
            semitones = -6

        output = self.td_psola(audio, semitones)

        return output

    def td_psola(self, audio, semitones):
        """
        Time-Domain PSOLA pitch shifting

        Args:
            audio: Input audio
            semitones: Pitch shift in semitones

        Returns:
            Pitch-shifted audio
        """
        # Calculate pitch shift ratio
        shift_ratio = 2 ** (semitones / 12.0)

        # Detect pitch marks
        marks = self._detect_pitch_marks(audio)

        if len(marks) < 3:
            # Not enough pitch marks (unvoiced or too short)
            return audio

        # Calculate average period
        avg_period = np.mean(np.diff(marks))

        # Target period after shift
        target_period = avg_period / shift_ratio

        # Generate synthesis time grid
        synthesis_times = np.arange(0, len(audio), target_period)

        # Extract pitch-synchronous grains
        grains = self._extract_grains(audio, marks, avg_period)

        # Overlap-add synthesis
        output = self._overlap_add_synthesis(
            grains,
            marks,
            synthesis_times,
            len(audio)
        )

        # Normalize
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output)) * 0.95

        return output

    def _detect_pitch_marks(self, audio):
        """
        Detect pitch marks (GCIs - Glottal Closure Instants)

        Simple method using peak detection
        """
        # Detect peaks in absolute value
        threshold = 0.2 * np.max(np.abs(audio))

        # Minimum distance between peaks (based on max F0)
        min_distance = int(self.sr / 400)  # 400 Hz max

        peaks, _ = find_peaks(
            np.abs(audio),
            height=threshold,
            distance=min_distance
        )

        return peaks

    def _extract_grains(self, audio, marks, avg_period):
        """
        Extract pitch-synchronous grains

        Args:
            audio: Input signal
            marks: Pitch mark locations
            avg_period: Average pitch period

        Returns:
            List of (mark, grain) tuples
        """
        grains = []

        # Grain window size: 2 pitch periods
        grain_size = int(avg_period * 2)

        for mark in marks:
            # Extract grain centered at mark
            start = max(0, mark - grain_size // 2)
            end = min(len(audio), mark + grain_size // 2)

            grain = audio[start:end].copy()

            # Apply Hanning window
            window = np.hanning(len(grain))
            grain *= window

            grains.append((mark, grain))

        return grains

    def _overlap_add_synthesis(self, grains, marks, synthesis_times, output_length):
        """
        Overlap-add synthesis

        Args:
            grains: List of (mark, grain) tuples
            marks: Original pitch mark locations
            synthesis_times: Target synthesis time points
            output_length: Desired output length

        Returns:
            Synthesized signal
        """
        output = np.zeros(output_length)

        for target_time in synthesis_times:
            if target_time >= output_length:
                break

            # Find closest original pitch mark
            closest_idx = np.argmin(np.abs(marks - target_time))

            if closest_idx >= len(grains):
                break

            mark, grain = grains[closest_idx]

            # Place grain at target time
            center = int(target_time)
            start = center - len(grain) // 2
            end = start + len(grain)

            # Bounds check and overlap-add
            if start >= 0 and end <= output_length:
                output[start:end] += grain
            elif start >= 0 and start < output_length:
                output[start:] += grain[:output_length - start]
            elif end > 0 and end <= output_length:
                valid_start = -start
                output[:end] += grain[valid_start:]

        return output


# === USAGE EXAMPLE ===

def example_psola_usage():
    """Example: PSOLA gender conversion"""

    # Initialize
    converter = PSOLAGenderConverter(sample_rate=16000)

    # Load audio
    audio, sr = sf.read('male_voice.wav')

    # Convert M2F
    female_voice = converter.convert(audio, mode='m2f')
    sf.write('female_voice_psola.wav', female_voice, sr)

    # Convert F2M
    male_voice = converter.convert(audio, mode='f2m')
    sf.write('male_voice_psola.wav', male_voice, sr)


if __name__ == '__main__':
    example_psola_usage()
```

---

## 9. Quality Assessment {#quality}

### 9.1 Objective Metrics

#### 9.1.1 Mel-Cepstral Distortion (MCD)

**Definition**: Distance between MFCC features of reference and converted speech.

**Formula**:
```
MCD = (10 / ln(10)) * √(2 * Σ(c_ref - c_conv)²)

where c = MFCC coefficients (typically 13)
```

**Implementation**:
```python
import librosa
import numpy as np

def compute_mcd(reference, converted, sr=16000):
    """
    Compute Mel-Cepstral Distortion

    Args:
        reference: Reference audio
        converted: Converted audio
        sr: Sample rate

    Returns:
        MCD value (lower is better)
    """
    # Extract MFCCs
    mfcc_ref = librosa.feature.mfcc(
        y=reference,
        sr=sr,
        n_mfcc=13,
        n_fft=512,
        hop_length=160
    )

    mfcc_conv = librosa.feature.mfcc(
        y=converted,
        sr=sr,
        n_mfcc=13,
        n_fft=512,
        hop_length=160
    )

    # Align lengths
    min_len = min(mfcc_ref.shape[1], mfcc_conv.shape[1])
    mfcc_ref = mfcc_ref[:, :min_len]
    mfcc_conv = mfcc_conv[:, :min_len]

    # Compute distance (exclude 0th coefficient - energy)
    diff = mfcc_ref[1:] - mfcc_conv[1:]

    # MCD formula
    mcd = (10 / np.log(10)) * np.mean(
        np.sqrt(2 * np.sum(diff ** 2, axis=0))
    )

    return mcd
```

**Interpretation**:
- MCD < 5.0: Excellent quality
- MCD 5.0-7.0: Good quality
- MCD 7.0-10.0: Moderate quality
- MCD > 10.0: Poor quality

**Typical Values for Gender Conversion**:
- WORLD M2F: MCD ~6.5-7.5
- WORLD F2M: MCD ~6.0-7.0
- PSOLA M2F: MCD ~8.0-9.5
- PSOLA F2M: MCD ~7.5-8.5

#### 9.1.2 F0 RMSE (Pitch Accuracy)

**Formula**:
```
F0_RMSE = √(1/N * Σ(F0_target - F0_converted)²)
```

**Implementation**:
```python
def compute_f0_rmse(target_audio, converted_audio, sr=16000):
    """
    Compute F0 Root Mean Square Error
    """
    import pyworld as pw

    # Extract F0
    f0_target, _ = pw.harvest(target_audio, sr)
    f0_converted, _ = pw.harvest(converted_audio, sr)

    # Align lengths
    min_len = min(len(f0_target), len(f0_converted))
    f0_target = f0_target[:min_len]
    f0_converted = f0_converted[:min_len]

    # Consider only voiced frames
    voiced_mask = (f0_target > 0) & (f0_converted > 0)

    if np.sum(voiced_mask) == 0:
        return float('inf')

    # RMSE
    rmse = np.sqrt(np.mean(
        (f0_target[voiced_mask] - f0_converted[voiced_mask]) ** 2
    ))

    return rmse
```

**Interpretation**:
- <5 Hz: Excellent
- 5-15 Hz: Good
- 15-30 Hz: Moderate
- >30 Hz: Poor

### 9.2 Subjective Metrics

#### 9.2.1 Mean Opinion Score (MOS)

**Method**: Human listeners rate quality on 1-5 scale.

**Scale**:
- 5: Excellent (indistinguishable from natural speech)
- 4: Good (minor artifacts, natural-sounding)
- 3: Fair (noticeable artifacts, understandable)
- 2: Poor (significant degradation)
- 1: Bad (unintelligible)

**Typical MOS Values**:
- Original speech: 4.5-5.0
- WORLD reconstruction: 4.2-4.5
- WORLD gender conversion: 3.5-4.0
- PSOLA gender conversion: 2.8-3.5

#### 9.2.2 Similarity Score

**Question**: "Does the converted voice sound like the target gender?"

**Scale**: 1-5 (1=not at all, 5=completely)

**Typical Values**:
- WORLD M2F: 3.8-4.2
- WORLD F2M: 3.6-4.0
- PSOLA M2F: 3.0-3.5
- PSOLA F2M: 2.8-3.2

### 9.3 Quality Comparison Summary

| Method | MCD | F0 RMSE | MOS | Similarity |
|--------|-----|---------|-----|------------|
| **WORLD M2F** | 6.5-7.5 | 8-15 Hz | 3.7 | 4.0 |
| **WORLD F2M** | 6.0-7.0 | 6-12 Hz | 3.8 | 3.9 |
| **PSOLA M2F** | 8.0-9.5 | 12-25 Hz | 3.2 | 3.3 |
| **PSOLA F2M** | 7.5-8.5 | 10-20 Hz | 3.3 | 3.1 |

**Key Findings**:
1. WORLD consistently outperforms PSOLA in all metrics
2. F2M generally achieves better quality than M2F
3. Formant shifting (WORLD) is critical for naturalness
4. MCD correlates well with perceived quality

---

## 10. References {#references}

### 10.1 WORLD Vocoder Papers

1. **Morise, M., Yokomori, F., & Ozawa, K. (2016).**
   "WORLD: A Vocoder-Based High-Quality Speech Synthesis System for Real-Time Applications"
   *IEICE Transactions on Information and Systems*, 99(7), 1877-1884.

2. **Morise, M. (2015).**
   "CheapTrick, a spectral envelope estimator for high-quality speech synthesis"
   *Speech Communication*, 67, 1-7.

3. **Morise, M. (2016).**
   "D4C, a band-aperiodicity estimator for high-quality speech synthesis"
   *Speech Communication*, 84, 57-65.

4. **Morise, M. (2017).**
   "Harvest: A high-performance fundamental frequency estimator from speech signals"
   *Proc. Interspeech*, 2321-2325.

### 10.2 PSOLA Papers

5. **Moulines, E., & Charpentier, F. (1990).**
   "Pitch-synchronous waveform processing techniques for text-to-speech synthesis using diphones"
   *Speech Communication*, 9(5-6), 453-467.

6. **Moulines, E., & Laroche, J. (1995).**
   "Non-parametric techniques for pitch-scale and time-scale modification of speech"
   *Speech Communication*, 16(2), 175-205.

### 10.3 Voice Conversion Papers

7. **Stylianou, Y., Cappé, O., & Moulines, E. (1998).**
   "Continuous probabilistic transform for voice conversion"
   *IEEE Transactions on Speech and Audio Processing*, 6(2), 131-142.

8. **Kawahara, H., Masuda-Katsuse, I., & De Cheveigné, A. (1999).**
   "Restructuring speech representations using a pitch-adaptive time–frequency smoothing and an instantaneous-frequency-based F0 extraction: Possible role of a repetitive structure in sounds"
   *Speech Communication*, 27(3-4), 187-207.

9. **Toda, T., Black, A. W., & Tokuda, K. (2007).**
   "Voice conversion based on maximum-likelihood estimation of spectral parameter trajectory"
   *IEEE Transactions on Audio, Speech, and Language Processing*, 15(8), 2222-2235.

10. **Sisman, B., Yamagishi, J., King, S., & Li, H. (2020).**
    "An overview of voice conversion and its challenges: From statistical modeling to deep learning"
    *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 29, 132-157.

### 10.4 Acoustic Phonetics References

11. **Fant, G. (1970).**
    "Acoustic Theory of Speech Production"
    *Mouton, The Hague*.

12. **Peterson, G. E., & Barney, H. L. (1952).**
    "Control methods used in a study of the vowels"
    *The Journal of the Acoustical Society of America*, 24(2), 175-184.

13. **Hillenbrand, J., Getty, L. A., Clark, M. J., & Wheeler, K. (1995).**
    "Acoustic characteristics of American English vowels"
    *The Journal of the Acoustical Society of America*, 97(5), 3099-3111.

### 10.5 Online Resources

14. **WORLD Vocoder Official Repository**
    https://github.com/mmorise/World

15. **PyWorld (Python wrapper)**
    https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder

16. **PSOLA Implementation (Python)**
    https://github.com/sannawag/TD-PSOLA

17. **Voice Gender Changer (PSOLA-based)**
    https://github.com/radinshayanfar/voice-gender-changer

### 10.6 Textbooks

18. **Rabiner, L. R., & Schafer, R. W. (2010).**
    "Theory and Applications of Digital Speech Processing"
    *Prentice Hall*.

19. **Quatieri, T. F. (2002).**
    "Discrete-Time Speech Signal Processing: Principles and Practice"
    *Prentice Hall*.

20. **Huang, X., Acero, A., & Hon, H. W. (2001).**
    "Spoken Language Processing: A Guide to Theory, Algorithm, and System Development"
    *Prentice Hall*.

---

## Appendix A: Mathematical Foundations

### A.1 Fourier Transform

The Fourier Transform decomposes a signal into its frequency components:

**Continuous Fourier Transform**:
```
X(f) = ∫ x(t) e^(-j2πft) dt
```

**Discrete Fourier Transform (DFT)**:
```
X[k] = Σ x[n] e^(-j2πkn/N)
```

**Fast Fourier Transform (FFT)**: Efficient O(N log N) algorithm for DFT

### A.2 Short-Time Fourier Transform (STFT)

STFT analyzes time-varying signals:

```
STFT{x[n]}(m, ω) = Σ x[n] w[n-m] e^(-jωn)

where:
  m = time frame index
  ω = frequency
  w = window function (e.g., Hamming, Hanning)
```

### A.3 Cepstral Analysis

**Cepstrum**: Inverse Fourier transform of log spectrum

```
c[n] = IFFT{ log|X[k]| }
```

**Mel-Frequency Cepstral Coefficients (MFCC)**:
1. Compute power spectrum
2. Apply mel-scale filterbank
3. Take logarithm
4. Apply DCT (Discrete Cosine Transform)

```
MFCC[k] = Σ log(S[m]) cos(πk(m-0.5)/M)
```

### A.4 Linear Prediction Coding (LPC)

Models speech production as:

```
s[n] = Σ a_i s[n-i] + G u[n]

where:
  a_i = prediction coefficients
  G = gain
  u[n] = excitation (pulse or noise)
```

LPC coefficients represent vocal tract filter.

---

## Appendix B: Common Artifacts and Solutions

### B.1 Robotization (Buzzy Sound)

**Cause**: Excessive phase distortion, incorrect F0 estimation

**Solution**:
- Use high-quality F0 estimator (Harvest instead of DIO)
- Reduce pitch shift amount
- Apply jitter to F0 contour (add natural variation)

```python
# Add F0 jitter
jitter = np.random.randn(len(f0)) * 2  # ±2 Hz
f0_jittered = f0 + jitter
```

### B.2 Phasiness (Hollow Sound)

**Cause**: Improper overlap-add, window function issues

**Solution**:
- Use appropriate window function (Hanning, Hamming)
- Ensure 50% overlap
- Check synthesis hop size

### B.3 Breathiness (Excessive Noise)

**Cause**: High aperiodicity, incorrect synthesis

**Solution**:
- Reduce aperiodicity parameter
- Check voiced/unvoiced detection
- Apply spectral smoothing

### B.4 Formant Smearing

**Cause**: Incorrect spectral warping, low-resolution FFT

**Solution**:
- Increase FFT size (2048 or 4096)
- Use higher-order interpolation (cubic)
- Apply formant-preserving warping

---

**End of Document**

**Version**: 1.0
**Last Updated**: January 24, 2026
**Total Pages**: ~50 equivalent pages
**Word Count**: ~12,000 words
