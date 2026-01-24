# Google Colab Testing Guide - GPU Voice Conversion Models

**Complete step-by-step instructions to test all 7 GPU models**

---

## 📋 What You'll Test

1. ✅ **GPT-SoVITS** - Best quality
2. ✅ **RVC** - Real-time capable
3. ✅ **SoftVC VITS** - Singing voice
4. ✅ **Seed-VC** - Lowest latency
5. ✅ **FreeVC** - Zero-shot
6. ✅ **VITS** - Multi-speaker TTS
7. ⚠️ **Kaldi** - Traditional (skip - too complex)

**Time Required**: 3-4 hours
**Cost**: FREE (Google Colab)
**Requirements**: Google account

---

## 🚀 Step-by-Step Instructions

### Step 1: Access Google Colab

1. Open your web browser
2. Go to: https://colab.research.google.com/
3. Sign in with your Google account

**Screenshot needed?** The page should show "Welcome to Colaboratory"

---

### Step 2: Upload the Notebook

**Option A: Upload from GitHub (Recommended)**

1. In Colab, click **File** → **Open notebook**
2. Click the **GitHub** tab
3. Enter: `MuruganR96/VoiceConversion_Survey`
4. Select: `GPU_Models_Colab_Notebook.ipynb`
5. Click **Open**

**Option B: Upload Manually**

1. Download `GPU_Models_Colab_Notebook.ipynb` from the repository
2. In Colab, click **File** → **Upload notebook**
3. Select the downloaded `.ipynb` file
4. Click **Upload**

---

### Step 3: Enable GPU Runtime ⚠️ **CRITICAL**

**This is the most important step!**

1. Click **Runtime** (in top menu)
2. Click **Change runtime type**
3. Under "Hardware accelerator", select **GPU**
4. Under "GPU type", select **T4** (free tier)
5. Click **Save**

**Verify GPU is enabled**:
- You should see "Connected" in top-right with a green checkmark
- It may show "RAM: 12.7 GB | Disk: 78.2 GB"

---

### Step 4: Run the Notebook Cells

**Important**: Run cells **one at a time** in order (top to bottom)

#### Cell 1: Check GPU
```
Click the ▶ (play) button on the left of the cell
```

**Expected Output**:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 11.8
GPU device: Tesla T4
GPU memory: 15.xx GB
```

If you see `CUDA available: False`, **STOP** and go back to Step 3.

---

#### Cell 2: Install Dependencies
```
Click the ▶ button
Wait 1-2 minutes
```

**Expected Output**:
```
✓ Common dependencies installed
```

---

#### Cell 3: Create Directories
```
Click ▶
```

**Expected Output**:
```
✓ Directory structure created
```

---

#### Cell 4: Generate Test Audio
```
Click ▶
Wait 5-10 seconds
```

**Expected Output**:
```
Generated: test_audio/male_voice.wav (F0=120Hz, 3s)
Generated: test_audio/female_voice.wav (F0=220Hz, 3s)
✓ Test audio files generated
```

---

#### Cell 5: Load Benchmarking Utilities
```
Click ▶
```

**Expected Output**:
```
✓ Benchmarking utilities loaded
```

---

### Step 5: Test Individual Models

Now you'll test each model. **Warning**: Some models may take 20-40 minutes each to download and setup.

#### Model 1: GPT-SoVITS

**Cell: Clone Repository**
```
Click ▶
Wait 2-5 minutes (downloading ~500MB)
```

**Expected Output**:
```
✓ GPT-SoVITS repository cloned and dependencies installed
```

**Cell: Download Models**
```
Click ▶
Wait 5-10 minutes (downloading pretrained models ~1GB)
```

⚠️ **Common Issues**:
- If download fails: Re-run the cell
- If "disk space full": You may need to skip some models

**Cell: Test GPT-SoVITS**
```
Click ▶
```

**Note**: This cell may show "Manual testing required" - this is normal. The model has been cloned and you can explore it manually.

---

#### Model 2: RVC

**Cell: Clone Repository**
```
Click ▶
Wait 2-3 minutes
```

**Cell: Download Models**
```
Click ▶
Wait 3-5 minutes
```

**Cell: Test RVC**
```
Click ▶
```

---

#### Model 3: SoftVC VITS

**Cell: Clone Repository**
```
Click ▶
Wait 2-3 minutes
```

**Cell: Download Models**
```
Click ▶
Wait 3-5 minutes
```

**Cell: Test SoftVC VITS**
```
Click ▶
```

---

#### Model 4: Seed-VC

**Cell: Clone Repository**
```
Click ▶
Wait 1-2 minutes
```

**Cell: Download Models**
```
Click ▶
Wait 2-3 minutes (smaller model ~150MB)
```

**Cell: Test Seed-VC**
```
Click ▶
```

**Expected Output** (if successful):
```
Seed-VC Results:
  Latency: 85.32 ± 5.12 ms
  GPU Memory: 2.34 GB
✓ Results saved to results/Seed-VC_results.json
```

---

#### Model 5: FreeVC

**Cell: Clone Repository**
```
Click ▶
Wait 1-2 minutes
```

**Cell: Download Models**
```
Click ▶
Wait 5-8 minutes (WavLM model ~300MB)
```

**Cell: Test FreeVC**
```
Click ▶
```

---

#### Model 6: VITS

**Cell: Clone Repository**
```
Click ▶
Wait 1-2 minutes
```

**Cell: Test VITS**
```
Click ▶
```

**Note**: VITS may require additional setup as it's primarily for TTS.

---

#### Model 7: Kaldi

**Skip this model** - too complex for automated testing.

---

### Step 6: View Results

**Cell: Results Summary**
```
Click ▶
```

**Expected Output**:
```
============================================================
VOICE CONVERSION GPU MODELS - TEST RESULTS SUMMARY
============================================================

Model                Latency (ms)    GPU Mem (GB)    MCD
------------------------------------------------------------
Seed-VC              85.32           2.34            5.67
[Other models if successfully tested]
============================================================
```

**Cell: Save Report**
```
Click ▶
```

This will automatically download `comprehensive_report.json` to your computer.

---

## 📊 Understanding the Results

### Metrics Explained

**Latency (ms)**:
- How long it takes to convert 3 seconds of audio
- Lower is better
- Target: <500ms for server use

**GPU Memory (GB)**:
- Peak GPU memory used during inference
- Indicates deployment requirements
- T4 GPU has 16GB total

**MCD (Mel-Cepstral Distortion)**:
- Quality metric (lower is better)
- <5.0: Excellent
- 5.0-7.0: Good
- 7.0-10.0: Moderate
- >10.0: Poor

---

## ⚠️ Troubleshooting

### Issue 1: "CUDA available: False"

**Solution**:
1. Click **Runtime** → **Disconnect and delete runtime**
2. Click **Runtime** → **Change runtime type**
3. Select **GPU** (T4)
4. Click **Save**
5. Re-run all cells from the beginning

---

### Issue 2: "Disk space full"

**Error**: `No space left on device`

**Solution**:
1. Click the **Files** icon (folder) on the left sidebar
2. Delete large model files you don't need
3. Or skip some models to save space

**Which models to skip if needed**:
- Skip Kaldi (already marked)
- Skip VITS (requires TTS adaptation)
- Keep: Seed-VC, RVC, GPT-SoVITS

---

### Issue 3: "Runtime disconnected"

**Colab free tier limits**:
- Maximum 12 hours per session
- May disconnect after 90 minutes of inactivity

**Solution**:
1. Re-connect: Click **Connect** button
2. Re-run cells from where you left off
3. Tip: Keep the browser tab active

---

### Issue 4: Model download fails

**Error**: `ConnectionError` or `TimeoutError`

**Solution**:
```python
# Re-run the download cell
# Or manually download:
!wget -c <url>  # -c continues interrupted downloads
```

---

### Issue 5: "Manual testing required"

**Why**: Some models have complex APIs that vary between versions.

**What to do**:
1. The model is cloned - you can explore it
2. Check the model's README in the cloned folder
3. Follow the model's official documentation
4. Use the benchmarking utilities provided

---

### Issue 6: Out of Memory Error

**Error**: `CUDA out of memory`

**Solution**:
```python
import torch
torch.cuda.empty_cache()  # Run this in a new cell
```

Then re-run the failing cell.

---

## 💡 Tips for Success

### 1. Run During Off-Peak Hours
- Colab resources are limited during peak times (US daytime)
- Try running at night (US timezone)

### 2. Save Progress Frequently
- Download result files after each model
- Take screenshots of outputs
- Don't rely on Colab session staying alive

### 3. One Model at a Time
- Don't rush - each model needs 20-40 minutes
- Clear GPU memory between models:
  ```python
  import torch
  torch.cuda.empty_cache()
  ```

### 4. Manual Testing May Be Required
- Some models are complex and need human intervention
- Use the cloned repositories for detailed exploration
- Follow each model's official documentation

---

## 📥 Downloading Results

### Option 1: Automatic Download (Last Cell)
```python
from google.colab import files
files.download('results/comprehensive_report.json')
```

This automatically downloads the report.

### Option 2: Manual Download
1. Click the **Files** icon (folder) on left sidebar
2. Navigate to `results/` folder
3. Right-click on `comprehensive_report.json`
4. Click **Download**

### Option 3: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r results /content/drive/MyDrive/voice_conversion_results
```

---

## 📊 Expected Results (Approximate)

Based on similar testing, expect:

| Model | Latency (ms) | GPU Memory (GB) | Status |
|-------|-------------|-----------------|--------|
| **Seed-VC** | 50-150 | 2-4 | ✅ Should work |
| **RVC** | 100-300 | 2-6 | ⚠️ May need manual setup |
| **SoftVC VITS** | 150-400 | 3-5 | ⚠️ May need manual setup |
| **GPT-SoVITS** | 300-800 | 6-12 | ⚠️ Complex setup |
| **FreeVC** | 200-600 | 4-6 | ⚠️ May need manual setup |
| **VITS** | 100-400 | 4-8 | ⚠️ Requires adaptation |
| **Kaldi** | - | - | ❌ Skip |

---

## 🎯 What If Automated Testing Doesn't Work?

**Don't worry!** This is expected for some models.

### What You've Accomplished:
1. ✅ Cloned all repositories
2. ✅ Downloaded pretrained models
3. ✅ Setup complete environment
4. ✅ Have benchmarking utilities ready

### Next Steps for Manual Testing:

**For each model**:
1. Navigate to the cloned folder
2. Read the model's README.md
3. Follow their inference examples
4. Use the `ModelBenchmark` class (provided in notebook) to measure:
   ```python
   benchmark = ModelBenchmark('ModelName')
   metrics = benchmark.measure_inference(your_conversion_function)
   benchmark.results['metrics'] = metrics
   benchmark.save_results()
   ```

---

## 📝 Reporting Your Results

### Create a Results Document

After testing, create a file `MY_GPU_TEST_RESULTS.md` with:

```markdown
# My GPU Voice Conversion Test Results

**Date**: [Date]
**GPU**: [e.g., Tesla T4]
**Colab Tier**: [Free/Pro]

## Models Tested

### Seed-VC
- **Latency**: XX ms
- **GPU Memory**: XX GB
- **MCD**: XX
- **Status**: ✅ Worked / ⚠️ Manual setup / ❌ Failed
- **Notes**: [Any issues or observations]

### RVC
[Same format]

[Continue for all models]

## Challenges Faced
[Describe any issues]

## Conclusions
[Your observations]
```

---

## 🆘 Need Help?

### Resources:
1. **This repository**: Check SERVER_SIDE_GPU_MODELS.md
2. **Model documentation**: Each cloned repo has its own README
3. **GitHub Issues**: Post in https://github.com/MuruganR96/VoiceConversion_Survey/issues

### Common Questions:

**Q: How long will this take?**
A: 3-4 hours total, but you can stop and resume.

**Q: Will I lose progress if disconnected?**
A: Yes, but you can re-run cells. Download results frequently.

**Q: Can I use Colab Pro?**
A: Yes! Pro gives better GPUs (V100/A100) and longer sessions.

**Q: What if a model doesn't work?**
A: That's okay! Document what happened and move to the next model.

**Q: Do I need to test all 7?**
A: No. Focus on: Seed-VC, RVC, SoftVC VITS (most likely to work automatically).

---

## ✅ Success Checklist

Before you finish, make sure you have:

- [ ] GPU enabled (verified CUDA available)
- [ ] Test audio generated
- [ ] At least 1-2 models tested successfully
- [ ] Results downloaded (JSON file)
- [ ] Screenshots of outputs
- [ ] Notes on what worked/didn't work

---

## 🎉 Next Steps After Testing

1. **Compare with documented results** in SERVER_SIDE_GPU_MODELS.md
2. **Share your findings** (optional - create GitHub issue with results)
3. **Choose best model** for your use case based on metrics
4. **Deploy** using SERVER_DEPLOYMENT_GUIDE.md

---

**Good luck with testing! Remember: Even partial results are valuable!** 🚀

**Repository**: https://github.com/MuruganR96/VoiceConversion_Survey
