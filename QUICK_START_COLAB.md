# Google Colab Quick Start - 5 Minute Setup

**Test GPU voice conversion models for FREE**

---

## 🚀 Option 1: Simplified Testing (Recommended)

### Step 1: Open Colab
1. Go to: https://colab.research.google.com/
2. Sign in with Google account

### Step 2: Load Notebook
1. Click **File** → **Open notebook**
2. Click **GitHub** tab
3. Enter: `MuruganR96/VoiceConversion_Survey`
4. Select: **Simple_GPU_Test_Colab.ipynb**
5. Click **Open**

### Step 3: Enable GPU ⚠️ CRITICAL
1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (T4)
3. Click **Save**

### Step 4: Run Cells
Click the ▶ button on each cell, top to bottom.

**First cell must show**:
```
CUDA available: True
GPU: Tesla T4
✅ GPU is ready!
```

If you see `CUDA available: False`, go back to Step 3!

---

## 📊 What You'll Get

After running all cells:

✅ **3 Model Repositories Cloned**:
- Seed-VC (easiest)
- RVC (most popular)
- GPT-SoVITS (best quality)

✅ **Dependencies Installed**

✅ **Test Audio Generated**

✅ **Ready for Manual Testing**

---

## 🎯 Next Steps

Each model needs **manual testing** because APIs vary. The notebook will:
1. ✅ Clone the repo
2. ✅ Install dependencies
3. ✅ Download models (if available)
4. 📝 Show you where to find documentation

Then **you explore** each model's README for inference instructions.

---

## 📖 Full Instructions

For detailed step-by-step guide, see: **COLAB_TESTING_GUIDE.md**

---

## 💡 Key Tips

1. **Keep browser tab active** (or Colab will disconnect)
2. **Download results frequently** (sessions expire after 12 hours)
3. **One cell at a time** - don't rush
4. **Read error messages** - they're usually helpful
5. **Be patient** - downloads take 5-10 minutes per model

---

## ⏱️ Time Estimate

- Setup: 5 minutes
- Repository cloning: 10-15 minutes
- Model downloads: 20-40 minutes
- Manual testing: 1-2 hours (your pace)

**Total**: 2-3 hours

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPU | Runtime → Change runtime type → GPU → Save |
| Disconnected | Re-run cells from where you left off |
| Disk full | Delete unnecessary files in Files panel |
| Download fails | Re-run the cell, or check model's repo |

---

## 🎓 What If Automated Testing Fails?

**This is normal!** These models are complex.

**What you've achieved**:
- ✅ Complete development environment on GPU
- ✅ All repositories cloned
- ✅ Dependencies installed
- ✅ Benchmarking tools ready

**Next**: Follow each model's official documentation for inference.

---

## 📥 Getting Your Notebooks

### Method 1: Direct Link
- **Simple version**: https://github.com/MuruganR96/VoiceConversion_Survey/blob/main/Simple_GPU_Test_Colab.ipynb
- **Full version**: https://github.com/MuruganR96/VoiceConversion_Survey/blob/main/GPU_Models_Colab_Notebook.ipynb

### Method 2: Clone Repo
```bash
git clone https://github.com/MuruganR96/VoiceConversion_Survey.git
cd VoiceConversion_Survey
# Upload .ipynb files to Colab
```

---

## ✅ Success Looks Like

After running the notebook:

```
✅ GPU is ready!
✅ Dependencies installed
✅ Test audio created
✅ Seed-VC cloned
✅ RVC cloned
✅ GPT-SoVITS cloned
```

Then you manually explore each model!

---

**Ready? Go to**: https://colab.research.google.com/

**Choose**: Simple_GPU_Test_Colab.ipynb (from GitHub tab)

**Good luck!** 🚀
