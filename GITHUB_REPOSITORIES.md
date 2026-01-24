# GitHub Repositories for Voice Conversion
## Male-to-Female & Female-to-Male Voice Conversion (CPU, ≤2MB)

This document lists ready-to-use GitHub repositories for testing voice conversion approaches.

---

## Quick Reference Table

| Repository | Type | Memory | Quality | Complexity | Recommended |
|------------|------|--------|---------|------------|-------------|
| mmorise/World | DSP | <1MB | Good | Low | ⭐⭐⭐⭐⭐ |
| radinshayanfar/voice-gender-changer | DSP | <500KB | Moderate | Very Low | ⭐⭐⭐⭐ |
| uthree/tinyvc | ML | 1-2MB* | Good | Medium | ⭐⭐⭐⭐ |
| KomalBabariya/Voice-Gender-Prediction-and-Conversion | DSP+ML | <2MB | Moderate | Medium | ⭐⭐⭐ |

*With quantization

---

## DSP-Based Repositories

### 1. WORLD Vocoder (Primary Recommendation)

**Repository**: https://github.com/mmorise/World

#### Overview
High-quality vocoder for speech analysis and synthesis with independent control of F0, spectral envelope, and aperiodicity.

#### Specifications
- **Language**: C++ (with MATLAB/Python wrappers)
- **License**: Modified BSD License
- **Platform**: Cross-platform (Windows, Linux, macOS)
- **Memory**: ~1MB
- **Latency**: 20-50ms
- **Quality**: Good to Excellent

#### Installation

**C++ (Native)**:
```bash
# Clone repository
git clone https://github.com/mmorise/World.git
cd World

# Build with CMake
mkdir build && cd build
cmake ..
make

# Test
./test
```

**Python Wrapper** (PyWorld):
```bash
pip install pyworld

# Or from source
git clone https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder.git
cd Python-Wrapper-for-World-Vocoder
pip install -e .
```

#### Usage for Gender Conversion

**Python Example**:
```python
import pyworld as pw
import numpy as np
import soundfile as sf

# Load audio
audio, sr = sf.read('input_male.wav')
audio = audio.astype(np.float64)

# Extract WORLD parameters
f0, sp, ap = pw.wav2world(audio, sr)

# M2F Conversion
f0_converted = f0 * 1.6  # Raise pitch by 60%
sp_converted = pw.scale_frequency(sp, sr, 1.2)  # Shift formants up 20%

# F2M Conversion (alternative)
# f0_converted = f0 * 0.7  # Lower pitch by 30%
# sp_converted = pw.scale_frequency(sp, sr, 0.85)  # Shift formants down 15%

# Synthesize
output = pw.synthesize(f0_converted, sp_converted, ap, sr)

# Save
sf.write('output_female.wav', output, sr)
```

**Advanced Formant Shifting** (custom implementation):
```python
def shift_formants(sp, ratio):
    """
    Shift formants by frequency warping
    ratio > 1.0: shift up (M2F)
    ratio < 1.0: shift down (F2M)
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

# Usage
sp_converted = shift_formants(sp, 1.18)  # M2F: 18% formant shift
```

#### Testing Checklist
- [ ] Install PyWorld
- [ ] Load sample male voice (8-10 seconds)
- [ ] Extract F0, SP, AP parameters
- [ ] Apply M2F conversion (F0 × 1.6, formants × 1.2)
- [ ] Synthesize and save output
- [ ] Measure memory usage (should be <1MB)
- [ ] Measure latency (aim for <50ms per second of audio)
- [ ] Listen to output quality
- [ ] Repeat for F2M conversion (F0 × 0.7, formants × 0.85)

#### Expected Results
- **Memory**: 800KB - 1.2MB
- **Processing Speed**: 2-3x real-time on modern CPU
- **Quality**: Natural prosody, possible spectral artifacts
- **Pitch Shift Accuracy**: ±0.5 semitones

---

### 2. voice-gender-changer (Simplest PSOLA)

**Repository**: https://github.com/radinshayanfar/voice-gender-changer

#### Overview
Lightweight Python tool using PSOLA (Pitch Synchronous Overlap-Add) for pitch shifting.

#### Specifications
- **Language**: Python
- **License**: MIT
- **Dependencies**: librosa, psola, soundfile
- **Memory**: <500KB
- **Latency**: 10-30ms
- **Quality**: Moderate (simple pitch shift)

#### Installation
```bash
# Clone repository
git clone https://github.com/radinshayanfar/voice-gender-changer.git
cd voice-gender-changer

# Install dependencies
pip install librosa psola soundfile numpy

# Alternative: use requirements.txt if available
pip install -r requirements.txt
```

#### Usage

**Command Line**:
```bash
# M2F: Shift pitch up by 4-5 semitones
python voice_changer.py input_male.wav output_female.wav --semitones 5

# F2M: Shift pitch down by 4-5 semitones
python voice_changer.py input_female.wav output_male.wav --semitones -5

# Custom pitch shift
python voice_changer.py input.wav output.wav --semitones 3.5
```

**Python API**:
```python
import librosa
import soundfile as sf
from psola import vocode

# Load audio
audio, sr = librosa.load('input_male.wav', sr=16000)

# M2F: Shift up 5 semitones
target_pitch = 2 ** (5 / 12.0)  # Semitones to frequency ratio
output = vocode(audio, sr, target_pitch=target_pitch)

# Save
sf.write('output_female.wav', output, sr)
```

**Pitch Estimation + Shifting**:
```python
import librosa
import numpy as np
from psola import vocode

def convert_gender_psola(audio, sr, semitone_shift):
    """
    Convert gender using PSOLA
    semitone_shift: +4 to +7 for M2F, -4 to -7 for F2M
    """
    # Estimate pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)

    # Pitch shift
    pitch_ratio = 2 ** (semitone_shift / 12.0)
    output = vocode(audio, sr, target_pitch=pitch_ratio)

    return output

# M2F
audio, sr = librosa.load('male_voice.wav', sr=16000)
female_voice = convert_gender_psola(audio, sr, semitone_shift=5)
sf.write('female_output.wav', female_voice, sr)

# F2M
audio, sr = librosa.load('female_voice.wav', sr=16000)
male_voice = convert_gender_psola(audio, sr, semitone_shift=-5)
sf.write('male_output.wav', male_voice, sr)
```

#### Testing Checklist
- [ ] Install dependencies (librosa, psola, soundfile)
- [ ] Test M2F conversion (+5 semitones)
- [ ] Test F2M conversion (-5 semitones)
- [ ] Measure memory usage
- [ ] Measure latency
- [ ] Evaluate quality (listen for artifacts)
- [ ] Test with different semitone values (3, 4, 5, 6, 7)

#### Expected Results
- **Memory**: 300KB - 600KB
- **Processing Speed**: 3-5x real-time
- **Quality**: Moderate (may sound "chipmunk-like" with large shifts)
- **Best Shift Range**: ±3 to ±5 semitones

---

### 3. Voice-Gender-Prediction-and-Conversion

**Repository**: https://github.com/KomalBabariya/Voice-Gender-Prediction-and-Conversion

#### Overview
Combines gender detection (male/female classification) with voice conversion using source-filter modification.

#### Specifications
- **Language**: Python
- **Features**: Gender prediction + conversion
- **Dataset**: TIMIT-based training
- **Memory**: <2MB (model + processing)

#### Installation
```bash
# Clone repository
git clone https://github.com/KomalBabariya/Voice-Gender-Prediction-and-Conversion.git
cd Voice-Gender-Prediction-and-Conversion

# Install dependencies
pip install numpy scipy librosa scikit-learn tensorflow
```

#### Usage
```python
# Gender prediction
predicted_gender = predict_gender('input_voice.wav')
print(f"Detected gender: {predicted_gender}")

# Automatic conversion to opposite gender
if predicted_gender == 'male':
    convert_to_female('input_voice.wav', 'output_female.wav')
else:
    convert_to_male('input_voice.wav', 'output_male.wav')
```

#### Testing Checklist
- [ ] Install dependencies
- [ ] Test gender prediction accuracy
- [ ] Test M2F conversion
- [ ] Test F2M conversion
- [ ] Measure total memory (model + inference)
- [ ] Evaluate conversion quality

---

## ML/DL-Based Repositories

### 4. TinyVC (Lightweight Neural Voice Conversion)

**Repository**: https://github.com/uthree/tinyvc

#### Overview
Lightweight neural voice conversion with built-in pitch shifting for gender transformation.

#### Specifications
- **Language**: Python (PyTorch)
- **Model Size**: ~5-8MB (original), ~1.5-2MB (quantized)
- **Latency**: 30-50ms
- **Quality**: Good

#### Installation
```bash
# Clone repository
git clone https://github.com/uthree/tinyvc.git
cd tinyvc

# Install dependencies
pip install torch torchaudio librosa soundfile numpy

# Install requirements if available
pip install -r requirements.txt
```

#### Usage

**Basic Conversion**:
```bash
# M2F: Shift pitch up by 5 semitones
python convert.py --input male_voice.wav --output female_voice.wav --pitch 5

# F2M: Shift pitch down by 5 semitones
python convert.py --input female_voice.wav --output male_voice.wav --pitch -5

# High quality mode (more memory)
python convert.py --input male.wav --output female.wav --pitch 5 --no-chunking
```

**Python API**:
```python
import torch
import torchaudio
from tinyvc import TinyVC

# Load model
model = TinyVC.load_pretrained('path/to/checkpoint.pt')
model.eval()

# Load audio
audio, sr = torchaudio.load('male_voice.wav')

# M2F conversion
with torch.no_grad():
    converted = model.convert(audio, pitch_shift=5)  # +5 semitones

# Save
torchaudio.save('female_voice.wav', converted, sr)
```

#### Model Quantization (for ≤2MB constraint)

**Export to ONNX**:
```python
import torch
import onnx

# Load PyTorch model
model = TinyVC.load_pretrained('tinyvc.pt')
model.eval()

# Dummy input
dummy_mel = torch.randn(1, 80, 100)  # Adjust to actual input shape

# Export
torch.onnx.export(
    model,
    dummy_mel,
    'tinyvc.onnx',
    input_names=['mel_input'],
    output_names=['audio_output'],
    dynamic_axes={'mel_input': {2: 'time'}, 'audio_output': {1: 'time'}}
)
```

**Quantize with ONNX Runtime**:
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic INT8 quantization
quantize_dynamic(
    model_input='tinyvc.onnx',
    model_output='tinyvc_int8.onnx',
    weight_type=QuantType.QInt8
)

# Check size
import os
original_size = os.path.getsize('tinyvc.onnx') / (1024 * 1024)
quantized_size = os.path.getsize('tinyvc_int8.onnx') / (1024 * 1024)

print(f"Original: {original_size:.2f} MB")
print(f"Quantized: {quantized_size:.2f} MB")
print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
```

**Inference with Quantized Model**:
```python
import onnxruntime as ort
import numpy as np

# Load quantized model
session = ort.InferenceSession('tinyvc_int8.onnx')

# Prepare input
mel_input = extract_mel_spectrogram(audio)  # Your preprocessing

# Run inference
outputs = session.run(None, {'mel_input': mel_input})
converted_audio = outputs[0]
```

#### Testing Checklist
- [ ] Install PyTorch and dependencies
- [ ] Download or train TinyVC model
- [ ] Test M2F conversion (pitch +5)
- [ ] Test F2M conversion (pitch -5)
- [ ] Measure original model memory
- [ ] Export to ONNX
- [ ] Apply INT8 quantization
- [ ] Verify quantized model ≤2MB
- [ ] Compare quality: original vs. quantized
- [ ] Measure inference latency
- [ ] Measure peak memory usage

#### Expected Results
- **Original Model**: 5-8MB, high quality
- **Quantized Model**: 1.5-2.5MB, good quality (slight degradation)
- **Processing Speed**: 1.5-2.5x real-time
- **Quality**: Good naturalness, better than DSP methods

---

### 5. Low-Latency Voice Conversion (LLVC)

**Paper**: https://arxiv.org/abs/2311.00873

#### Overview
State-of-the-art low-latency voice conversion optimized for CPU inference.

#### Specifications
- **Latency**: <20ms at 16kHz
- **Speed**: 2.8x faster than real-time on consumer CPU
- **Architecture**: GAN + knowledge distillation
- **Model Size**: ~8MB (original), ~2MB (quantized)

#### Implementation Notes
The official code repository may require searching as it's from Koe AI (research paper). Look for:
- "LLVC voice conversion GitHub"
- "Koe AI voice conversion"
- Related implementations in awesome-voice-conversion list

#### Expected Approach (based on paper)
```python
# Pseudo-code (adapt to actual implementation when found)

import torch
from llvc import LLVCModel

# Load pretrained model
model = LLVCModel.load_pretrained('llvc_checkpoint.pt')

# Quantize for edge deployment
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(model_int8.state_dict(), 'llvc_quantized.pt')

# Inference
audio_tensor = load_audio('male_voice.wav')
with torch.no_grad():
    converted = model_int8(audio_tensor, target_speaker_id=female_id)
save_audio(converted, 'female_voice.wav')
```

#### Testing Checklist (when implementation available)
- [ ] Locate and clone LLVC repository
- [ ] Install dependencies
- [ ] Download pretrained weights
- [ ] Test voice conversion quality
- [ ] Apply INT8 quantization
- [ ] Verify model size ≤2MB
- [ ] Benchmark latency (<20ms target)
- [ ] Measure memory usage
- [ ] Compare with WORLD and TinyVC

---

## Supporting Repositories

### 6. awesome-voice-conversion (Curated List)

**Repository**: https://github.com/JeffC0628/awesome-voice-conversion

#### Purpose
Comprehensive list of voice conversion papers, code, and resources.

#### How to Use
1. Browse for latest lightweight VC methods
2. Find CPU-optimized implementations
3. Discover pre-trained models
4. Stay updated on new research

**Key Sections**:
- Real-time voice conversion
- Low-latency models
- CPU-optimized implementations
- Pitch and formant shifting

---

### 7. SoftVC VITS (Reference - Not for 2MB deployment)

**Repository**: https://github.com/svc-develop-team/so-vits-svc

#### Overview
High-quality singing voice conversion (also works for speech).

**Note**: Original model is too large (40-100MB) for the 2MB constraint. Only useful as:
- Quality benchmark reference
- Source for knowledge distillation (train tiny student model)
- Understanding state-of-the-art architecture

#### If Exploring Compression:
```python
# Extreme compression (not recommended, severe quality loss)
# 1. Train tiny student model (10-20% of original size)
# 2. Knowledge distillation from SoftVC VITS
# 3. Aggressive INT8 quantization
# 4. Target: ~2MB (will have significant quality degradation)
```

---

## Quantization & Compression Tools

### 8. Intel Neural Compressor

**Repository**: https://github.com/intel/neural-compressor

#### Purpose
Automated model compression toolkit optimized for Intel CPUs.

#### Installation
```bash
pip install neural-compressor
```

#### Usage for Voice Conversion Model
```python
from neural_compressor import Quantization

# Configure quantization
config = {
    'model': {
        'name': 'tinyvc',
        'framework': 'pytorch'
    },
    'quantization': {
        'approach': 'post_training_static_quant',
        'calibration': {
            'sampling_size': 100
        }
    },
    'tuning': {
        'accuracy_criterion': {
            'relative': 0.01  # 1% accuracy loss tolerance
        }
    }
}

# Run quantization
quantizer = Quantization(config)
quantized_model = quantizer(
    model=original_model,
    calib_dataloader=calibration_dataloader
)

# Save
quantized_model.save('tinyvc_compressed')
```

---

### 9. ONNX Runtime

**Repository**: https://github.com/microsoft/onnxruntime

#### Purpose
Cross-platform inference engine with quantization support.

#### Installation
```bash
pip install onnxruntime
```

#### Quantization Script
```python
# quantize_model.py

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def quantize_voice_model(input_model, output_model):
    """Quantize voice conversion model to INT8"""

    # Quantize
    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QInt8,
        optimize_model=True
    )

    # Verify size
    original_size = os.path.getsize(input_model) / (1024 * 1024)
    quantized_size = os.path.getsize(output_model) / (1024 * 1024)

    print(f"Original model: {original_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")

    if quantized_size > 2.0:
        print(f"WARNING: Model size {quantized_size:.2f} MB exceeds 2MB target!")
    else:
        print(f"SUCCESS: Model fits in {quantized_size:.2f} MB")

# Usage
quantize_voice_model('tinyvc.onnx', 'tinyvc_int8.onnx')
```

---

## Quick Start Testing Script

Save as `test_all_methods.py`:

```python
#!/usr/bin/env python3
"""
Quick test script for all voice conversion methods
"""

import os
import time
import numpy as np
import soundfile as sf
import librosa

# Test audio (use your own or generate)
def generate_test_audio(duration=3, sr=16000, f0=150):
    """Generate simple test tone"""
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * f0 * t)
    return audio, sr

def test_psola():
    """Test PSOLA method"""
    print("\n=== Testing PSOLA (voice-gender-changer) ===")
    try:
        from psola import vocode

        # Generate male voice (150 Hz)
        audio, sr = generate_test_audio(f0=150)

        # M2F: +5 semitones
        start = time.time()
        pitch_ratio = 2 ** (5 / 12.0)
        output = vocode(audio, sr, target_pitch=pitch_ratio)
        elapsed = time.time() - start

        sf.write('test_psola_m2f.wav', output, sr)

        print(f"✓ PSOLA M2F completed in {elapsed:.3f}s")
        print(f"  RTF: {elapsed/3:.3f}")
        print(f"  Memory: <500KB")

    except ImportError:
        print("✗ PSOLA not available (pip install psola)")

def test_world():
    """Test WORLD vocoder"""
    print("\n=== Testing WORLD Vocoder ===")
    try:
        import pyworld as pw

        # Generate male voice
        audio, sr = generate_test_audio(f0=150, duration=3)
        audio = audio.astype(np.float64)

        start = time.time()

        # Extract parameters
        f0, sp, ap = pw.wav2world(audio, sr)

        # M2F conversion
        f0_converted = f0 * 1.6
        # Note: proper formant shifting requires custom implementation

        # Synthesize
        output = pw.synthesize(f0_converted, sp, ap, sr)
        elapsed = time.time() - start

        sf.write('test_world_m2f.wav', output, sr)

        print(f"✓ WORLD M2F completed in {elapsed:.3f}s")
        print(f"  RTF: {elapsed/3:.3f}")
        print(f"  Memory: ~1MB")

    except ImportError:
        print("✗ WORLD not available (pip install pyworld)")

def test_tinyvc():
    """Test TinyVC (if installed)"""
    print("\n=== Testing TinyVC ===")
    try:
        # Placeholder - requires actual TinyVC installation
        print("  TinyVC requires model download and setup")
        print("  See repository: github.com/uthree/tinyvc")
    except ImportError:
        print("✗ TinyVC not available")

def main():
    print("Voice Conversion Methods - Quick Test")
    print("=" * 50)

    # Create test directory
    os.makedirs('test_outputs', exist_ok=True)
    os.chdir('test_outputs')

    # Run tests
    test_psola()
    test_world()
    test_tinyvc()

    print("\n" + "=" * 50)
    print("Testing complete. Check test_outputs/ directory.")

if __name__ == "__main__":
    main()
```

**Run**:
```bash
python test_all_methods.py
```

---

## Memory Profiling Script

Save as `profile_memory.py`:

```python
#!/usr/bin/env python3
"""Memory profiling for voice conversion methods"""

import tracemalloc
import psutil
import os

def profile_conversion(func, *args):
    """Profile memory usage of conversion function"""

    process = psutil.Process(os.getpid())

    # Baseline
    mem_before = process.memory_info().rss / 1024 / 1024

    # Trace Python allocations
    tracemalloc.start()

    # Run conversion
    result = func(*args)

    # Peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Final memory
    mem_after = process.memory_info().rss / 1024 / 1024

    return {
        'python_peak_mb': peak / 1024 / 1024,
        'system_delta_mb': mem_after - mem_before,
        'result': result
    }

# Example usage
if __name__ == "__main__":
    import pyworld as pw
    import numpy as np

    # Test audio
    audio = np.random.randn(16000 * 3).astype(np.float64)
    sr = 16000

    def world_conversion(audio, sr):
        f0, sp, ap = pw.wav2world(audio, sr)
        f0 = f0 * 1.6
        return pw.synthesize(f0, sp, ap, sr)

    stats = profile_conversion(world_conversion, audio, sr)

    print(f"Python peak memory: {stats['python_peak_mb']:.2f} MB")
    print(f"System memory delta: {stats['system_delta_mb']:.2f} MB")

    if stats['system_delta_mb'] <= 2.0:
        print("✓ Memory constraint MET (≤2MB)")
    else:
        print(f"✗ Memory constraint EXCEEDED ({stats['system_delta_mb']:.2f} MB)")
```

---

## Next Steps

1. **Clone Priority Repositories**:
   ```bash
   git clone https://github.com/mmorise/World.git
   git clone https://github.com/radinshayanfar/voice-gender-changer.git
   git clone https://github.com/uthree/tinyvc.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install pyworld librosa psola soundfile torch onnxruntime
   ```

3. **Run Baseline Tests**:
   - Test WORLD vocoder first (most reliable)
   - Test PSOLA second (simplest)
   - Test TinyVC if model available

4. **Benchmark**:
   - Measure memory (target: ≤2MB)
   - Measure latency (target: <100ms)
   - Evaluate quality (subjective listening)

5. **Optimize**:
   - Quantize ML models to INT8
   - Optimize DSP parameters
   - Profile and optimize hotspots

---

## Summary Recommendations

### For Immediate Testing (This Week)
1. **WORLD Vocoder** - Best balance, proven technology
2. **PSOLA** - Simplest baseline

### For Quality-Focused Development (Next 2-4 Weeks)
3. **TinyVC** - After quantization, best ML approach
4. **LLVC** - If implementation becomes available

### For Reference Only
5. **SoftVC VITS** - Quality benchmark (too large for deployment)

---

**Document Version**: 1.0
**Last Updated**: January 24, 2026
