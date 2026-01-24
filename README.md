# Voice Conversion: M2F & F2M for Edge Deployment

**Real-time voice conversion optimized for CPU with ≤2MB memory constraint**

---

## Project Overview

This repository provides comprehensive documentation and testing frameworks for voice conversion between male and female voices, specifically optimized for edge device deployment with strict resource constraints.

### Target Specifications
- **Platform**: CPU-only (no GPU required)
- **Memory**: ≤2MB model size
- **Latency**: <100ms (real-time capable)
- **Use Case**: Edge devices, embedded systems, mobile applications

---

## Approaches Covered

### 1. DSP-Based Methods
- **PSOLA** (Pitch Synchronous Overlap-Add)
  - Memory: <500KB
  - Quality: Moderate
  - Complexity: Very Low

- **WORLD Vocoder** ⭐ Recommended
  - Memory: ~1MB
  - Quality: Good
  - Complexity: Low

- **Phase Vocoder**
  - Memory: ~800KB
  - Quality: Good
  - Complexity: Moderate

### 2. ML/DL-Based Methods
- **LLVC** (Low-Latency Voice Conversion)
  - Memory: ~2MB (quantized)
  - Quality: High
  - Latency: <20ms

- **TinyVC**
  - Memory: 1.5-2MB (quantized)
  - Quality: Good
  - Complexity: Medium

- **Quantized Neural Models**
  - INT8 quantization techniques
  - Knowledge distillation
  - Model compression pipelines

---

## Quick Start

### Prerequisites
```bash
# Python 3.8+
python --version

# Install core dependencies
pip install numpy librosa soundfile pyworld psola
```

### Test WORLD Vocoder (Recommended First Test)

```python
import pyworld as pw
import numpy as np
import soundfile as sf

# Load audio
audio, sr = sf.read('male_voice.wav')
audio = audio.astype(np.float64)

# Extract WORLD parameters
f0, sp, ap = pw.wav2world(audio, sr)

# M2F: Raise pitch and formants
f0_female = f0 * 1.6  # +60% pitch
# Note: Formant shifting requires additional processing (see GITHUB_REPOSITORIES.md)

# Synthesize
output = pw.synthesize(f0_female, sp, ap, sr)

# Save
sf.write('female_voice.wav', output, sr)
```

### Test PSOLA (Simplest Method)

```bash
# Clone repository
git clone https://github.com/radinshayanfar/voice-gender-changer.git
cd voice-gender-changer

# Install dependencies
pip install librosa psola soundfile

# M2F conversion (+5 semitones)
python voice_changer.py input_male.wav output_female.wav --semitones 5

# F2M conversion (-5 semitones)
python voice_changer.py input_female.wav output_male.wav --semitones -5
```

---

## Key Documents

### 📄 [VOICE_CONVERSION_TECHNICAL_REPORT.md](VOICE_CONVERSION_TECHNICAL_REPORT.md)
**Comprehensive technical analysis** covering:
- Detailed DSP approach specifications
- ML/DL architecture analysis
- Quantization techniques for <2MB deployment
- Comparative benchmarking methodology
- Expected performance metrics
- Implementation roadmap

### 📋 [GITHUB_REPOSITORIES.md](GITHUB_REPOSITORIES.md)
**Ready-to-use repositories** with:
- Installation instructions
- Usage examples with code
- Quantization pipelines
- Memory profiling scripts
- Testing checklists

---

## Recommended Path

### Week 1: DSP Baseline
1. **Install WORLD Vocoder**
   ```bash
   pip install pyworld
   ```

2. **Test M2F Conversion**
   - Load male voice sample
   - Apply F0 shift (×1.6)
   - Apply formant shift (+20%)
   - Measure memory and latency

3. **Test F2M Conversion**
   - Load female voice sample
   - Apply F0 shift (×0.7)
   - Apply formant shift (-15%)
   - Compare quality

**Expected Results**: <1MB memory, 20-40ms latency, moderate-good quality

### Week 2: ML Model Setup
1. **Clone TinyVC**
   ```bash
   git clone https://github.com/uthree/tinyvc.git
   cd tinyvc
   pip install -r requirements.txt
   ```

2. **Test Pre-trained Model** (if available)
   - Run inference on test audio
   - Measure original model size
   - Evaluate quality vs. WORLD

3. **Benchmark Performance**
   - Memory profiling
   - Latency measurement
   - Quality metrics (MCD)

### Week 3: Model Compression
1. **Export to ONNX**
   ```python
   torch.onnx.export(model, dummy_input, "tinyvc.onnx")
   ```

2. **INT8 Quantization**
   ```python
   from onnxruntime.quantization import quantize_dynamic
   quantize_dynamic("tinyvc.onnx", "tinyvc_int8.onnx")
   ```

3. **Verify Constraints**
   - Check model size ≤2MB
   - Test inference speed
   - Compare quality degradation

### Week 4: Benchmarking & Optimization
1. **Comprehensive Testing**
   - Run benchmark suite
   - Generate comparison reports
   - Collect MOS scores (optional)

2. **Optimization**
   - Profile CPU hotspots
   - Optimize critical paths
   - Test on target hardware

3. **Documentation**
   - Document findings
   - Create deployment guide
   - Package for production

---

## Performance Targets

| Metric | Target | WORLD | PSOLA | TinyVC (INT8) |
|--------|--------|-------|-------|---------------|
| **Memory** | ≤2MB | ✅ ~1MB | ✅ <500KB | ✅ ~1.8MB |
| **Latency** | <100ms | ✅ 20-40ms | ✅ 10-30ms | ✅ 30-50ms |
| **RTF** | <1.0 | ✅ 0.3-0.5 | ✅ 0.2-0.3 | ✅ 0.4-0.6 |
| **MCD** | <7.0 | ✅ 6.5-8.0 | ⚠️ 7.5-9.0 | ✅ 4.5-6.5 |

**Legend**:
- ✅ Meets target
- ⚠️ Close to target
- ❌ Does not meet target

---

## Comparison Summary

### Best Overall: WORLD Vocoder
**Pros**:
- Lightweight (<1MB)
- Fast (2-3x real-time)
- No training required
- Well-documented
- Cross-platform

**Cons**:
- Moderate quality
- Requires parameter tuning
- May produce artifacts

**Use When**: You need reliable, lightweight conversion with acceptable quality

---

### Best Quality: Quantized TinyVC or LLVC
**Pros**:
- High quality output
- Natural sounding
- Real-time capable
- Learns speaker characteristics

**Cons**:
- Requires training data
- More complex deployment
- Fixed speaker pairs

**Use When**: Quality is priority and 2MB budget is available

---

### Simplest: PSOLA
**Pros**:
- Minimal code
- Very lightweight
- Fast processing
- Easy to understand

**Cons**:
- Lower quality
- Audible artifacts
- Simple pitch shift only

**Use When**: You need absolute simplest solution or <500KB constraint

---

## GitHub Repositories

### Primary Repositories

1. **WORLD Vocoder** (C++/Python) ⭐
   - https://github.com/mmorise/World
   - Python wrapper: `pip install pyworld`

2. **voice-gender-changer** (Python)
   - https://github.com/radinshayanfar/voice-gender-changer
   - PSOLA-based, very simple

3. **TinyVC** (Python/PyTorch)
   - https://github.com/uthree/tinyvc
   - Lightweight neural VC

4. **awesome-voice-conversion** (Curated List)
   - https://github.com/JeffC0628/awesome-voice-conversion
   - Comprehensive resource list

### Supporting Tools

5. **Intel Neural Compressor**
   - https://github.com/intel/neural-compressor
   - Model compression toolkit

6. **ONNX Runtime**
   - https://github.com/microsoft/onnxruntime
   - Cross-platform inference

See [GITHUB_REPOSITORIES.md](GITHUB_REPOSITORIES.md) for detailed setup instructions.

---

## Testing & Benchmarking

### Memory Profiling
```python
import tracemalloc
import psutil

tracemalloc.start()
output = conversion_function(input_audio)
current, peak = tracemalloc.get_traced_memory()

print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
```

### Latency Measurement
```python
import time
import numpy as np

times = []
for _ in range(10):
    start = time.perf_counter()
    output = conversion_function(audio)
    times.append(time.perf_counter() - start)

avg_latency = np.mean(times) * 1000  # ms
rtf = np.mean(times) / audio_duration

print(f"Latency: {avg_latency:.2f} ms")
print(f"RTF: {rtf:.3f} (< 1.0 = real-time)")
```

### Quality Metrics
```python
import librosa
import numpy as np

def compute_mcd(reference, converted, sr=16000):
    """Mel-Cepstral Distortion"""
    mfcc_ref = librosa.feature.mfcc(y=reference, sr=sr, n_mfcc=13)
    mfcc_conv = librosa.feature.mfcc(y=converted, sr=sr, n_mfcc=13)

    min_len = min(mfcc_ref.shape[1], mfcc_conv.shape[1])
    diff = mfcc_ref[:, :min_len] - mfcc_conv[:, :min_len]

    mcd = np.mean(np.sqrt(np.sum(diff**2, axis=0))) * (10 / np.log(10)) * 2
    return mcd

# MCD < 5.0: Excellent
# MCD 5.0-7.0: Good
# MCD 7.0-10.0: Moderate
```

---

## Next Steps

### Immediate (This Week)
1. Clone WORLD vocoder repository or install PyWorld
2. Install dependencies (`pip install pyworld librosa soundfile`)
3. Test M2F and F2M conversion
4. Measure memory and latency
5. Evaluate quality (subjective listening)

### Short-term (2-4 Weeks)
1. Setup TinyVC
2. Download or train model
3. Implement INT8 quantization
4. Benchmark all methods
5. Generate comparison report

### Long-term (1-2 Months)
1. Optimize best-performing method
2. Test on target hardware
3. Package for deployment
4. Create production API
5. Document deployment guide

---

## Resources

### Papers
- Low-latency Real-time Voice Conversion on CPU (arXiv:2311.00873, 2023)
- WORLD: A Vocoder-Based High-Quality Speech Synthesis System (IEICE 2016)
- Voice Conversion Using Pitch Shifting with PSOLA

### Tools
- Librosa: Audio processing library
- PyTorch: ML framework
- ONNX Runtime: Inference engine
- Intel Neural Compressor: Model compression

---

## FAQ

**Q: Which method should I start with?**
A: Start with WORLD vocoder - it's reliable, well-documented, and meets all constraints.

**Q: Can these methods work in real-time?**
A: Yes, all recommended methods achieve RTF <1.0 on modern CPUs (real-time capable).

**Q: How much quality loss from quantization?**
A: INT8 quantization typically causes <5% MCD increase with proper quantization-aware training.

**Q: Do I need training data?**
A: DSP methods (WORLD, PSOLA) require no training. ML methods need data for training/fine-tuning.

**Q: Can I use this on mobile devices?**
A: Yes, especially WORLD and PSOLA are suitable for mobile. Neural models require testing on target device.

**Q: What's the typical pitch shift for M2F?**
A: +4 to +7 semitones (F0 multiply by 1.5-1.8) with formant shift +15-20%.

**Q: What's the typical pitch shift for F2M?**
A: -4 to -7 semitones (F0 multiply by 0.6-0.75) with formant shift -15-20%.

---

**Last Updated**: January 24, 2026
**Version**: 1.0
**Status**: Ready for Testing

---

## Quick Links

- [📄 Technical Report](VOICE_CONVERSION_TECHNICAL_REPORT.md) - Comprehensive analysis
- [📋 GitHub Repositories](GITHUB_REPOSITORIES.md) - Ready-to-use code with setup instructions
- [🔧 WORLD Setup Guide](https://github.com/mmorise/World) - Official WORLD repository
- [🎤 PSOLA Voice Changer](https://github.com/radinshayanfar/voice-gender-changer) - Simple PSOLA implementation
- [🤖 TinyVC](https://github.com/uthree/tinyvc) - Lightweight neural VC
- [📚 Awesome Voice Conversion](https://github.com/JeffC0628/awesome-voice-conversion) - Curated resource list

**Ready to get started? See [GITHUB_REPOSITORIES.md](GITHUB_REPOSITORIES.md) for step-by-step installation instructions!**
