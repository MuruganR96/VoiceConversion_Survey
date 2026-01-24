# Voice Conversion Test Results
**Generated**: 2026-01-24 08:43:20
---

## Test Configuration

- **Platform**: CPU-only
- **Target**: ≤2MB memory, <100ms latency, RTF <1.0
- **Test Audio**: Synthetic male (120Hz) and female (220Hz) voices
- **Duration**: 3 seconds per sample

---

## Results Summary

| Method | Memory (MB) | Latency (ms) | RTF | Status |
|--------|-------------|--------------|-----|--------|
| WORLD Vocoder | See detailed results | See detailed results | See detailed results | ✓ 2 files |
| PSOLA | See detailed results | See detailed results | See detailed results | ✓ 2 files |

---

## Detailed Results

See test output above for detailed metrics including:

- Memory profiling (Python peak and system delta)
- Latency measurements (average ± std dev)
- Real-time factor (RTF)
- Pitch shift accuracy

## Generated Files

### WORLD Vocoder

- `results/world/f2m_output.wav`
- `results/world/m2f_output.wav`

### PSOLA

- `results/psola/f2m_output.wav`
- `results/psola/m2f_output.wav`

---

## Next Steps

1. Listen to output files in `results/` directories
2. Compare quality subjectively
3. Verify memory and latency meet constraints
4. Select best approach for your use case

**Note**: For ML/DL approaches (TinyVC), model training and quantization are required before testing.
