# Voice Conversion Technical Report
## Male-to-Female (M2F) & Female-to-Male (F2M) Voice Conversion for Edge Deployment

**Target Constraints**: CPU-based, ≤2MB memory footprint, Real-time capable

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [DSP-Based Approaches](#dsp-based-approaches)
3. [ML/DL-Based Approaches](#ml-dl-based-approaches)
4. [Recommended GitHub Repositories](#recommended-github-repositories)
5. [Comparative Analysis](#comparative-analysis)
6. [Testing Methodology](#testing-methodology)
7. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

This report evaluates voice conversion techniques for gender transformation (M2F/F2M) optimized for edge devices with strict memory constraints (≤2MB) and CPU-only execution. Two primary approaches are analyzed:

1. **DSP-Based**: Signal processing techniques (PSOLA, WORLD vocoder)
2. **ML/DL-Based**: Lightweight neural networks with quantization

### Key Findings
- **DSP methods** require minimal memory (<1MB) and are CPU-efficient but offer moderate quality
- **Quantized neural models** can achieve <2MB with INT8 quantization while maintaining higher quality
- Real-time processing is achievable with both approaches on modern CPUs

---

## DSP-Based Approaches

### 1. PSOLA (Pitch Synchronous Overlap-Add)

#### Overview
PSOLA is a classic time-domain pitch modification technique that maintains formant structure while shifting fundamental frequency (F0).

#### Technical Details
- **Pitch Shifting**: Detects pitch using algorithms like SIFT (Simplified Inverse Filter Tracking) or YIN
- **Time Stretching**: Windows speech into pitch-synchronized grains
- **Formant Preservation**: Resamples to shift formants independently of pitch

#### Gender Conversion Strategy
- **M2F Conversion**:
  - Pitch shift: +3 to +5 semitones (+50-80 Hz typical)
  - Formant shift: +15-20% (simulates shorter vocal tract)

- **F2M Conversion**:
  - Pitch shift: -3 to -5 semitones (-50-80 Hz typical)
  - Formant shift: -15-20% (simulates longer vocal tract)

#### Implementation Characteristics
- **Memory**: <500KB (algorithm only)
- **Latency**: 10-30ms typical
- **CPU Usage**: Low (single-core real-time on modest CPUs)
- **Quality**: Moderate (artifacts possible with large shifts)

#### Advantages
✓ Minimal memory footprint
✓ No training data required
✓ Deterministic and controllable
✓ Fast processing

#### Disadvantages
✗ Limited naturalness for large pitch shifts
✗ Possible artifacts (phasiness, metallic sound)
✗ Doesn't model speaker characteristics

---

### 2. WORLD Vocoder

#### Overview
WORLD is a high-quality vocoder system designed for real-time speech synthesis and modification. It decomposes speech into three parameters:
- **F0 (Fundamental Frequency)**: Pitch contour
- **SP (Spectral Envelope)**: Formant structure
- **AP (Aperiodicity)**: Noise characteristics

#### Technical Details
WORLD provides independent control over pitch and spectral characteristics, making it ideal for voice conversion.

#### Gender Conversion Strategy
- **F0 Modification**: Scale fundamental frequency
  - M2F: Multiply F0 by 1.5-1.8
  - F2M: Multiply F0 by 0.6-0.75

- **Spectral Envelope Warping**: Shift formants
  - M2F: Compress frequency axis (higher formants)
  - F2M: Expand frequency axis (lower formants)

- **Aperiodicity**: Adjust breathiness/roughness

#### Implementation Characteristics
- **Memory**: ~1MB (vocoder + parameters)
- **Latency**: 20-50ms
- **CPU Usage**: Moderate (optimized C++ implementation)
- **Quality**: Good (higher than PSOLA)

#### Advantages
✓ Independent parameter control
✓ Higher quality than PSOLA
✓ Well-documented and maintained
✓ Real-time capable

#### Disadvantages
✗ Slightly higher latency than PSOLA
✗ Requires parameter tuning
✗ Still produces non-human artifacts with extreme modifications

---

### 3. Phase Vocoder with Formant Shifting

#### Overview
Frequency-domain approach using STFT (Short-Time Fourier Transform) to independently modify pitch and formants.

#### Technical Details
- Decomposes speech into time-frequency representation
- Shifts frequency bins for formant modification
- Resamples for pitch modification

#### Implementation Characteristics
- **Memory**: ~800KB
- **Latency**: 30-60ms (depends on FFT size)
- **CPU Usage**: Moderate
- **Quality**: Good for moderate shifts

#### Gender Conversion Parameters
- **FFT Size**: 1024-2048 (trade-off: quality vs latency)
- **Hop Size**: 256-512 samples
- **Formant Shift**: 1.15-1.25 for M2F, 0.8-0.85 for F2M

---

## ML/DL-Based Approaches

### 1. LLVC (Low-Latency Voice Conversion)

#### Overview
LLVC is a state-of-the-art lightweight voice conversion model optimized for real-time CPU inference with minimal resources.

#### Technical Specifications
- **Latency**: <20ms at 16kHz
- **Speed**: 2.8x faster than real-time on consumer CPU
- **Architecture**: Generative adversarial network + knowledge distillation
- **Model Size**: Can be quantized to <2MB

#### Architecture Details
- **Encoder**: Lightweight convolutional feature extractor
- **Bottleneck**: Compressed speaker representation
- **Decoder**: Fast waveform generator
- **Training**: Knowledge distillation from larger teacher model

#### Quantization Strategy for ≤2MB Constraint
1. **INT8 Quantization**: 4x memory reduction
   - Original FP32: ~8MB → INT8: ~2MB

2. **Mixed Precision**: Critical layers in INT8, others in lower precision

3. **Weight Pruning**: Remove <5% least important weights

#### Implementation Characteristics
- **Memory (Quantized)**: ~2MB
- **Latency**: 15-20ms
- **CPU Usage**: Low (single-core capable)
- **Quality**: High (near state-of-the-art)

#### Advantages
✓ Excellent quality-to-size ratio
✓ Real-time on CPU
✓ Learns speaker characteristics
✓ Natural output

#### Disadvantages
✗ Requires training data
✗ Fixed speaker pairs (less flexible than DSP)
✗ Quantization may reduce quality slightly

---

### 2. TinyVC (Lightweight Voice Conversion)

#### Overview
Specifically designed as a minimal voice conversion system with pitch shifting capabilities for gender transformation.

#### Technical Specifications
- **Model Size**: Small (can be <2MB with quantization)
- **Pitch Shifting**: Semitone-based control
  - 12 semitones = 1 octave shift
  - M2F: +4 to +7 semitones typical
  - F2M: -4 to -7 semitones typical

#### Architecture
- Compact encoder-decoder with attention
- Built-in pitch shift module
- Optimized for CPU inference

#### Memory Optimization Options
- `--no-chunking`: Higher quality but more RAM
- Chunked processing: Lower memory, suitable for edge devices

#### Implementation Characteristics
- **Memory**: 1-2MB (quantized)
- **Latency**: 30-50ms
- **CPU Usage**: Low-moderate
- **Quality**: Good

#### Advantages
✓ Simple to use
✓ Built-in pitch control
✓ Lightweight architecture
✓ Active development

#### Disadvantages
✗ Quality lower than LLVC
✗ Limited documentation
✗ May require fine-tuning

---

### 3. Quantized SoftVC VITS (Compressed)

#### Overview
SoftVC VITS is a high-quality voice conversion system. For edge deployment, aggressive compression is required.

#### Original Specifications
- **Model Size**: 40-100MB (too large for constraint)
- **Quality**: State-of-the-art
- **Latency**: 50-100ms

#### Compression Strategy for ≤2MB
This requires extreme compression which significantly impacts quality:

1. **INT8 Quantization**: 4x reduction
2. **Knowledge Distillation**: Train tiny student model (10-20% of original)
3. **Architecture Pruning**: Remove attention layers, reduce channels
4. **Vocabulary Reduction**: Limit speaker embedding dimensions

#### Compressed Characteristics
- **Memory**: ~2MB (heavily compressed)
- **Latency**: 40-80ms
- **CPU Usage**: Moderate
- **Quality**: Moderate (significant degradation from compression)

#### Advantages
✓ Based on proven high-quality architecture
✓ Can achieve 2MB target
✓ Automatic pitch prediction

#### Disadvantages
✗ Severe quality loss from compression
✗ Complex compression pipeline
✗ Higher latency than LLVC
✗ **Not recommended** for strict 2MB constraint

---

### 4. Model Compression Techniques for Edge Deployment

#### INT8 Quantization
- **Memory Reduction**: 4x (FP32 → INT8)
- **Speed Improvement**: 2-4x on CPUs with SIMD
- **Quality Impact**: Minimal with quantization-aware training
- **Tools**: ONNX Runtime, Intel Neural Compressor, TensorFlow Lite

#### Dynamic Quantization
- Weights stored as INT8
- Activations computed in FP32
- **Memory**: ~3-4x reduction
- **Speed**: 2-3x improvement

#### Quantization-Aware Training (QAT)
- Simulate quantization during training
- Model adapts to lower precision
- Best accuracy for compressed models

#### Example: Voice Model Compression Pipeline
```
Original FP32 Model (8MB)
    ↓ [Prune 30% weights]
Pruned Model (5.6MB)
    ↓ [INT8 Quantization]
Quantized Model (1.4MB)
    ↓ [Knowledge Distillation]
Final Tiny Model (1.8MB)
```

---

## Recommended GitHub Repositories

### DSP-Based Implementations

#### 1. **WORLD Vocoder** (Official)
- **Repository**: `mmorise/World`
- **Language**: C++
- **License**: Modified BSD
- **Status**: Actively maintained
- **Memory**: ~1MB
- **Platform**: Cross-platform (Windows, Linux, macOS)

**Features**:
- High-quality speech analysis/synthesis
- Independent F0, spectral, aperiodicity control
- Real-time capable
- Extensive documentation

**Usage for Gender Conversion**:
```cpp
// Extract parameters
F0, SP, AP = WORLD_Analyze(audio);

// M2F conversion
F0_converted = F0 * 1.6;  // Raise pitch
SP_converted = FormantShift(SP, +1.2);  // Shift formants up

// F2M conversion
F0_converted = F0 * 0.7;  // Lower pitch
SP_converted = FormantShift(SP, -1.15);  // Shift formants down

// Synthesize
output = WORLD_Synthesize(F0_converted, SP_converted, AP);
```

**Deployment Readiness**: ⭐⭐⭐⭐⭐ (Excellent for edge)

---

#### 2. **voice-gender-changer**
- **Repository**: `radinshayanfar/voice-gender-changer`
- **Language**: Python
- **License**: MIT
- **Dependencies**: librosa, psola, soundfile

**Features**:
- Simple PSOLA-based pitch shifting
- Pitch estimation using librosa.pyin
- Command-line interface
- Lightweight

**Usage**:
```python
# M2F: shift pitch up by 4 semitones
python voice_changer.py input.wav output.wav --semitones 4

# F2M: shift pitch down by 4 semitones
python voice_changer.py input.wav output.wav --semitones -4
```

**Deployment Readiness**: ⭐⭐⭐⭐ (Good, Python dependency is overhead)

---

#### 3. **Voice-Gender-Prediction-and-Conversion**
- **Repository**: `KomalBabariya/Voice-Gender-Prediction-and-Conversion`
- **Language**: Python
- **Features**: Gender detection + conversion

**Components**:
- Gender prediction (male/female classification)
- Source-filter modification for conversion
- Based on vocal tract modeling

**Deployment Readiness**: ⭐⭐⭐ (Research code, needs optimization)

---

### ML/DL-Based Implementations

#### 1. **Low-Latency Voice Conversion (LLVC)**
- **Paper**: arXiv:2311.00873 (2023)
- **Repository**: Search for "LLVC voice conversion" or "Koe AI voice conversion"
- **Status**: Research implementation available

**Key Features**:
- **<20ms latency** at 16kHz
- **2.8x real-time** on consumer CPU
- GAN + knowledge distillation architecture
- Lowest resource usage of open-source VC models

**Quantization for Edge**:
- Apply INT8 quantization using ONNX Runtime
- Target: 1.5-2MB model size
- Expected quality: High

**Deployment Readiness**: ⭐⭐⭐⭐⭐ (Best ML option for edge)

---

#### 2. **TinyVC**
- **Repository**: `uthree/tinyvc`
- **Language**: Python (PyTorch)
- **License**: Open source

**Features**:
- Lightweight voice conversion
- Built-in pitch shifting (semitone control)
- `--no-chunking` mode for quality vs. memory trade-off

**Usage**:
```bash
# M2F conversion (+5 semitones)
python tinyvc.py --input male.wav --output female.wav --pitch 5

# F2M conversion (-5 semitones)
python tinyvc.py --input female.wav --output male.wav --pitch -5
```

**Quantization Path**:
```bash
# Export to ONNX
python export_onnx.py --model tinyvc.pth --output tinyvc.onnx

# Quantize with ONNX Runtime
python quantize.py --model tinyvc.onnx --output tinyvc_int8.onnx
```

**Deployment Readiness**: ⭐⭐⭐⭐ (Good with quantization)

---

#### 3. **Awesome Voice Conversion**
- **Repository**: `JeffC0628/awesome-voice-conversion`
- **Type**: Curated list

**Contents**:
- Comprehensive list of VC papers and repos
- Low-latency and real-time VC resources
- CPU-optimized implementations
- Links to LLVC and other lightweight models

**Use**: Reference for discovering latest methods

---

### Supporting Tools for Compression

#### 1. **ONNX Runtime with Quantization**
- **Repository**: `microsoft/onnxruntime`
- **Purpose**: Cross-platform inference with quantization

**Capabilities**:
- Dynamic/static INT8 quantization
- CPU-optimized inference
- Minimal dependencies for edge deployment

---

#### 2. **Intel Neural Compressor**
- **Repository**: `intel/neural-compressor`
- **Purpose**: Model compression toolkit

**Features**:
- Automated quantization
- Pruning and knowledge distillation
- Optimization for Intel CPUs
- Support for PyTorch, TensorFlow, ONNX

---

## Comparative Analysis

### DSP vs. ML/DL Trade-offs

| Aspect | DSP (PSOLA/WORLD) | ML/DL (LLVC/TinyVC) |
|--------|-------------------|---------------------|
| **Memory** | <1MB | 1.5-2MB (quantized) |
| **Quality** | Moderate | High |
| **Latency** | 10-50ms | 15-50ms |
| **Training Data** | None required | Required |
| **Flexibility** | High (parameter control) | Low (fixed speakers) |
| **Naturalness** | Moderate (artifacts) | High |
| **CPU Usage** | Very Low | Low-Moderate |
| **Deployment Complexity** | Simple | Moderate |
| **Robustness** | High | Moderate |

### Recommended Approach by Use Case

#### 1. **Extreme Resource Constraint (<1MB, minimal CPU)**
→ **PSOLA** (voice-gender-changer)
- Simplest implementation
- Lowest memory
- Acceptable quality for basic use

#### 2. **Balanced Constraint (~1-2MB, moderate quality)**
→ **WORLD Vocoder**
- Better quality than PSOLA
- Still lightweight
- More control over conversion

#### 3. **Quality Priority (2MB budget, modern CPU)**
→ **Quantized LLVC**
- Best quality
- Real-time capable
- Natural output
- Requires training/fine-tuning

#### 4. **Hybrid Approach (Recommended)**
→ **WORLD + TinyVC**
- Use WORLD for initial processing
- Use TinyVC for refinement
- Total memory: ~1.8MB
- Best quality-to-resource ratio

---

## Testing Methodology

### Performance Metrics

#### 1. **Memory Profiling**
Measure actual runtime memory usage:

```python
import tracemalloc
import psutil
import os

def profile_memory(func, audio_input):
    """Profile peak memory usage"""

    # System memory before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Python object memory tracking
    tracemalloc.start()

    # Run conversion
    output = func(audio_input)

    # Peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # System memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    return {
        'python_peak_mb': peak / 1024 / 1024,
        'system_delta_mb': mem_after - mem_before,
        'output': output
    }
```

**Target**: Peak memory ≤ 2MB

---

#### 2. **Latency Measurement**
Real-time factor (RTF) and absolute latency:

```python
import time
import numpy as np

def measure_latency(conversion_func, audio, sample_rate=16000):
    """Measure conversion latency and RTF"""

    audio_duration = len(audio) / sample_rate

    # Warmup
    _ = conversion_func(audio[:1000])

    # Measure
    times = []
    for _ in range(10):  # Average over 10 runs
        start = time.perf_counter()
        output = conversion_func(audio)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    rtf = avg_time / audio_duration  # <1.0 = real-time capable

    return {
        'latency_ms': avg_time * 1000,
        'rtf': rtf,
        'audio_duration_s': audio_duration
    }
```

**Target**: RTF < 1.0 (real-time), latency < 100ms

---

#### 3. **Quality Assessment**

##### Objective Metrics

**Mel-Cepstral Distortion (MCD)**:
```python
import librosa
import numpy as np

def compute_mcd(ref_audio, conv_audio, sr=16000):
    """Compute Mel-Cepstral Distortion"""

    # Extract MFCCs
    mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=13)
    mfcc_conv = librosa.feature.mfcc(y=conv_audio, sr=sr, n_mfcc=13)

    # Align lengths
    min_len = min(mfcc_ref.shape[1], mfcc_conv.shape[1])
    mfcc_ref = mfcc_ref[:, :min_len]
    mfcc_conv = mfcc_conv[:, :min_len]

    # MCD formula
    diff = mfcc_ref - mfcc_conv
    mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0))) * (10 / np.log(10)) * 2

    return mcd
```

**Interpretation**:
- MCD < 5.0: Excellent
- MCD 5.0-7.0: Good
- MCD 7.0-10.0: Moderate
- MCD > 10.0: Poor

**Pitch Accuracy**:
```python
def evaluate_pitch_shift(original, converted, target_shift_semitones):
    """Verify pitch shift accuracy"""

    # Estimate F0
    f0_orig, _, _ = librosa.pyin(original, fmin=80, fmax=400)
    f0_conv, _, _ = librosa.pyin(converted, fmin=80, fmax=400)

    # Remove NaN (unvoiced)
    f0_orig = f0_orig[~np.isnan(f0_orig)]
    f0_conv = f0_conv[~np.isnan(f0_conv)]

    # Median F0
    median_orig = np.median(f0_orig)
    median_conv = np.median(f0_conv)

    # Actual shift
    actual_shift = 12 * np.log2(median_conv / median_orig)
    error = abs(actual_shift - target_shift_semitones)

    return {
        'target_semitones': target_shift_semitones,
        'actual_semitones': actual_shift,
        'error_semitones': error,
        'original_f0_hz': median_orig,
        'converted_f0_hz': median_conv
    }
```

##### Subjective Metrics

**Mean Opinion Score (MOS)**:
- Collect ratings from 10+ listeners
- Scale: 1 (Bad) to 5 (Excellent)
- Evaluate: Naturalness, Speaker similarity, Quality

**Test Protocol**:
```
1. Prepare 20 test utterances (10 M2F, 10 F2M)
2. Convert using each method
3. Randomize presentation order
4. Listeners rate each conversion (1-5)
5. Compute mean and 95% confidence intervals
```

---

#### 4. **CPU Profiling**

```python
import cProfile
import pstats

def profile_cpu(func, audio):
    """Profile CPU usage and hotspots"""

    profiler = cProfile.Profile()
    profiler.enable()

    output = func(audio)

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

    return output
```

**Analyze**:
- Identify bottlenecks
- Optimize hot paths
- Verify single-core capability

---

### Test Dataset

#### Recommended Datasets

1. **VCTK Corpus** (Recommended)
   - 110 speakers (male/female)
   - ~400 utterances per speaker
   - Clean, high-quality recordings
   - Diverse accents

2. **LibriSpeech**
   - Large-scale dataset
   - Various speakers
   - Good for robustness testing

3. **Custom Recording**
   - Record 50 utterances
   - Controlled conditions
   - Representative of target use case

#### Test Set Composition
```
Training/Fine-tuning: 80% (ML models only)
Validation: 10%
Testing: 10% (20 M2F + 20 F2M conversions)
```

---

### Benchmark Protocol

```python
# benchmark.py

import json
from pathlib import Path

class VoiceConversionBenchmark:
    def __init__(self, test_audio_dir, output_dir):
        self.test_audio_dir = Path(test_audio_dir)
        self.output_dir = Path(output_dir)
        self.results = {}

    def test_method(self, method_name, conversion_func, gender_pair):
        """Test a single conversion method"""

        results = {
            'method': method_name,
            'gender_pair': gender_pair,
            'metrics': {
                'memory': [],
                'latency': [],
                'rtf': [],
                'mcd': [],
                'pitch_accuracy': []
            }
        }

        # Get test files
        test_files = list(self.test_audio_dir.glob(f"{gender_pair}/*.wav"))

        for audio_file in test_files:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=16000)

            # Memory profiling
            mem_stats = profile_memory(conversion_func, audio)
            results['metrics']['memory'].append(mem_stats['system_delta_mb'])

            # Latency
            lat_stats = measure_latency(conversion_func, audio, sr)
            results['metrics']['latency'].append(lat_stats['latency_ms'])
            results['metrics']['rtf'].append(lat_stats['rtf'])

            # Quality (if reference available)
            ref_file = self.test_audio_dir / f"{gender_pair}_ref" / audio_file.name
            if ref_file.exists():
                ref_audio, _ = librosa.load(ref_file, sr=16000)
                mcd = compute_mcd(ref_audio, mem_stats['output'], sr)
                results['metrics']['mcd'].append(mcd)

        # Aggregate statistics
        for metric in results['metrics']:
            values = results['metrics'][metric]
            if values:
                results['metrics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        self.results[method_name] = results
        return results

    def save_results(self):
        """Save benchmark results"""
        output_file = self.output_dir / 'benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to {output_file}")

    def generate_report(self):
        """Generate markdown report"""
        report = "# Voice Conversion Benchmark Results\n\n"

        for method_name, results in self.results.items():
            report += f"## {method_name}\n\n"
            report += f"**Gender Pair**: {results['gender_pair']}\n\n"
            report += "| Metric | Mean | Std | Min | Max |\n"
            report += "|--------|------|-----|-----|-----|\n"

            for metric, stats in results['metrics'].items():
                if isinstance(stats, dict):
                    report += f"| {metric} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n"

            report += "\n"

        report_file = self.output_dir / 'BENCHMARK_REPORT.md'
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"Report generated: {report_file}")

# Usage
if __name__ == "__main__":
    benchmark = VoiceConversionBenchmark(
        test_audio_dir="./test_audio",
        output_dir="./results"
    )

    # Test each method
    # benchmark.test_method("PSOLA", psola_convert, "M2F")
    # benchmark.test_method("WORLD", world_convert, "M2F")
    # benchmark.test_method("LLVC_INT8", llvc_convert, "M2F")
    # benchmark.test_method("TinyVC", tinyvc_convert, "M2F")

    benchmark.save_results()
    benchmark.generate_report()
```

---

## Implementation Roadmap

### Phase 1: DSP Baseline (Week 1)

#### Tasks
1. **Setup WORLD Vocoder**
   ```bash
   git clone https://github.com/mmorise/World.git
   cd World
   mkdir build && cd build
   cmake ..
   make
   ```

2. **Implement M2F/F2M Conversion**
   - Create parameter modification functions
   - Tune F0 and formant shift ratios
   - Test on sample audio

3. **Setup PSOLA (Python)**
   ```bash
   pip install psola librosa soundfile
   git clone https://github.com/radinshayanfar/voice-gender-changer.git
   ```

4. **Baseline Testing**
   - Run on 10 test samples
   - Measure memory and latency
   - Document quality (subjective listening)

**Deliverable**: Working DSP baseline with performance metrics

---

### Phase 2: ML Model Setup (Week 2)

#### Tasks
1. **TinyVC Installation**
   ```bash
   git clone https://github.com/uthree/tinyvc.git
   cd tinyvc
   pip install -r requirements.txt
   ```

2. **Model Training/Fine-tuning**
   - Prepare dataset (VCTK subset)
   - Train or download pretrained model
   - Test initial conversion quality

3. **LLVC Setup** (if available)
   - Search for implementation
   - Install dependencies
   - Run inference tests

**Deliverable**: Working ML models (pre-quantization)

---

### Phase 3: Model Compression (Week 3)

#### Tasks
1. **Export to ONNX**
   ```python
   import torch
   import onnx

   # Export TinyVC
   dummy_input = torch.randn(1, 80, 100)  # Example
   torch.onnx.export(model, dummy_input, "tinyvc.onnx")
   ```

2. **INT8 Quantization**
   ```python
   from onnxruntime.quantization import quantize_dynamic

   quantize_dynamic(
       model_input="tinyvc.onnx",
       model_output="tinyvc_int8.onnx",
       weight_type=QuantType.QInt8
   )
   ```

3. **Memory Validation**
   - Load quantized model
   - Verify size ≤ 2MB
   - Measure inference memory

4. **Quality Testing**
   - Compare quantized vs. original
   - Compute MCD degradation
   - Subjective listening tests

**Deliverable**: Quantized models meeting 2MB constraint

---

### Phase 4: Comprehensive Benchmarking (Week 4)

#### Tasks
1. **Prepare Test Suite**
   - 40 test utterances (20 M2F, 20 F2M)
   - Reference conversions (if available)

2. **Run Benchmark Protocol**
   - Memory profiling (all methods)
   - Latency measurement (all methods)
   - Quality metrics (MCD, pitch accuracy)

3. **Generate Reports**
   - Automated benchmark script
   - Comparison tables
   - Visualization (charts)

4. **MOS Collection** (optional)
   - Setup listening test
   - Collect 10+ ratings per sample
   - Statistical analysis

**Deliverable**: Complete benchmark report with recommendations

---

### Phase 5: Optimization & Deployment Prep (Week 5)

#### Tasks
1. **Code Optimization**
   - Profile CPU hotspots
   - Optimize critical paths
   - Parallelize where possible

2. **Edge Deployment Package**
   - Create standalone binaries (C++/WORLD)
   - Python deployment scripts
   - Docker container (optional)

3. **Documentation**
   - API documentation
   - Deployment guide
   - Usage examples

4. **Final Validation**
   - Test on target hardware (if available)
   - Verify all constraints met
   - Edge case testing

**Deliverable**: Production-ready voice conversion system

---

## Expected Results Summary

### DSP Methods (WORLD Vocoder - Recommended)
- **Memory**: <1MB ✓
- **Latency**: 20-40ms ✓
- **RTF**: 0.3-0.5 (3-2x real-time) ✓
- **MCD**: 6.5-8.0 (moderate quality)
- **Deployment**: Immediate

### ML Methods (Quantized LLVC - Best Quality)
- **Memory**: ~2MB ✓
- **Latency**: 15-25ms ✓
- **RTF**: 0.35-0.6 (2.8-1.6x real-time) ✓
- **MCD**: 4.5-6.0 (good-excellent quality)
- **Deployment**: Requires quantization pipeline

### Hybrid Recommendation
**Primary**: WORLD Vocoder for deployment
**Secondary**: Quantized TinyVC for quality-critical applications
**Fallback**: PSOLA for ultra-constrained scenarios

---

## References & Resources

### Papers
1. Low-latency Real-time Voice Conversion on CPU (arXiv:2311.00873)
2. WORLD: A Vocoder-Based High-Quality Speech Synthesis System (IEICE 2016)
3. Voice Conversion Using Pitch Shifting with PSOLA and Re-Sampling

### GitHub Repositories
- WORLD: https://github.com/mmorise/World
- TinyVC: https://github.com/uthree/tinyvc
- voice-gender-changer: https://github.com/radinshayanfar/voice-gender-changer
- Awesome Voice Conversion: https://github.com/JeffC0628/awesome-voice-conversion
- Intel Neural Compressor: https://github.com/intel/neural-compressor

### Tools
- ONNX Runtime: https://onnxruntime.ai/
- Librosa: https://librosa.org/
- PyTorch: https://pytorch.org/

---

## Conclusion

For edge deployment with ≤2MB memory and CPU-only constraints:

1. **Best Overall**: WORLD Vocoder
   - Proven, reliable, lightweight
   - Good quality-to-resource ratio
   - Immediate deployment readiness

2. **Best Quality** (if 2MB acceptable): Quantized LLVC or TinyVC
   - Higher quality output
   - Still real-time capable
   - Requires model compression workflow

3. **Simplest**: PSOLA (voice-gender-changer)
   - Minimal dependencies
   - Lowest memory (<500KB)
   - Acceptable for basic applications

**Recommended Path**: Start with WORLD for baseline, then explore quantized neural models if quality requirements demand it.

---

**Document Version**: 1.0
**Date**: January 24, 2026
**Author**: Voice Conversion Technical Analysis
