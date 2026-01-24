# Server-Side Voice Conversion - Quick Deployment Guide

**Goal**: Deploy high-quality GPU-optimized voice conversion on server infrastructure

---

## Quick Selection Guide

### Choose Your Model in 30 Seconds

**Q1: Do you need the absolute best quality?**
→ **YES**: Use **GPT-SoVITS**
→ NO: Continue to Q2

**Q2: Do you need real-time (low latency)?**
→ **YES**: Use **Seed-VC** or **RVC**
→ NO: Continue to Q3

**Q3: Is this for singing voice conversion?**
→ **YES**: Use **SoftVC VITS**
→ NO: Continue to Q4

**Q4: Do you need zero-shot (no training)?**
→ **YES**: Use **FreeVC** or **Seed-VC**
→ NO: Use **RVC** (best balance)

---

## Top 3 Recommendations

### 🥇 #1: GPT-SoVITS - Production Quality

**Why**: State-of-the-art quality, production-ready API, few-shot learning

**Setup Time**: 30 minutes

**Quick Start**:
```bash
# Clone
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# Install (requires Python 3.9+)
pip install -r requirements.txt

# Download models
python download_models.py

# Start API server
python api.py

# Access at http://localhost:9880
```

**Test Conversion**:
```bash
curl -X POST http://localhost:9880/tts \
  -H "Content-Type: application/json" \
  -d '{
    "ref_audio_path": "female_reference.wav",
    "prompt_text": "Reference text",
    "text": "Convert this text",
    "text_language": "en",
    "prompt_language": "en"
  }' \
  --output converted.wav
```

**Requirements**:
- GPU: NVIDIA RTX 3060+ (6GB VRAM minimum)
- RAM: 16GB+
- Storage: 5GB for models
- OS: Linux (Ubuntu 20.04+) or Windows

**Pros**:
✅ Best quality (MOS 4.6/5.0)
✅ Few-shot (5s-1min training data)
✅ Production API included
✅ WebUI for easy testing

**Cons**:
⚠️ Higher latency (300-800ms)
⚠️ Large GPU memory (6-12GB)

---

### 🥈 #2: RVC - Real-Time Server

**Why**: Fast, real-time capable, excellent quality, easy setup

**Setup Time**: 20 minutes

**Quick Start**:
```bash
# Clone
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
cd Retrieval-based-Voice-Conversion-WebUI

# Auto install (Linux/Mac)
bash install.sh

# Or Windows
install.bat

# Start WebUI
python infer-web.py

# Access at http://localhost:7865
```

**Python API**:
```python
from infer.modules.vc.modules import VC
from configs.config import Config

config = Config()
vc = VC(config)
vc.get_vc("pretrained_models/model.pth")

# Convert
output = vc.vc_single(
    sid=0,
    input_audio_path="male_voice.wav",
    f0_up_key=6,  # Pitch shift for M2F
    f0_method="rmvpe"
)
```

**Requirements**:
- GPU: NVIDIA GTX 1660+ (4GB VRAM minimum)
- RAM: 8GB+
- Storage: 3GB
- OS: Linux, Windows, or macOS

**Pros**:
✅ Fast (100-300ms latency)
✅ Real-time capable
✅ User-friendly WebUI
✅ Active community

**Cons**:
⚠️ Needs 10+ min training data
⚠️ Not zero-shot

---

### 🥉 #3: Seed-VC - Lowest Latency

**Why**: Fastest inference, zero-shot, simple API

**Setup Time**: 15 minutes

**Quick Start**:
```bash
# Clone
git clone https://github.com/Plachtaa/seed-vc.git
cd seed-vc

# Install
pip install -r requirements.txt

# Download model
python download_models.py
```

**Usage**:
```python
from seed_vc import SeedVC
import soundfile as sf

# Load
model = SeedVC("models/seed_vc.pt", device="cuda")

# Convert
output = model.convert(
    source_audio="male_voice.wav",
    target_audio="female_reference.wav",
    auto_f0=True
)

# Save
sf.write("converted.wav", output, 16000)
```

**Requirements**:
- GPU: NVIDIA GTX 1060+ (2GB VRAM minimum)
- RAM: 8GB+
- Storage: 1GB
- OS: Linux or Windows

**Pros**:
✅ Very fast (50-150ms)
✅ Zero-shot
✅ Low GPU memory
✅ Simple API

**Cons**:
⚠️ Quality slightly lower than GPT-SoVITS
⚠️ Less documentation

---

## GPU Requirements

### Minimum Specs

| Model | GPU | VRAM | RAM | Storage |
|-------|-----|------|-----|---------|
| **Seed-VC** | GTX 1060 | 2GB | 8GB | 1GB |
| **RVC** | GTX 1660 | 4GB | 8GB | 3GB |
| **SoftVC VITS** | RTX 2060 | 6GB | 16GB | 4GB |
| **GPT-SoVITS** | RTX 3060 | 6GB | 16GB | 5GB |
| **FreeVC** | RTX 2060 | 4GB | 16GB | 3GB |

### Recommended Specs (Production)

- **GPU**: NVIDIA RTX 3090 or RTX 4090
- **VRAM**: 24GB
- **RAM**: 64GB
- **Storage**: 100GB SSD
- **CPU**: 16+ cores
- **OS**: Ubuntu 22.04 LTS

---

## Installation Checklist

### Step 1: System Preparation

```bash
# Update system (Ubuntu)
sudo apt update && sudo apt upgrade -y

# Install CUDA (11.8 recommended)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install cuDNN
# Download from NVIDIA website
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb
sudo apt update
sudo apt install libcudnn8

# Install Python 3.9+
sudo apt install python3.9 python3.9-pip python3.9-venv

# Install audio libraries
sudo apt install ffmpeg libsndfile1
```

### Step 2: Model Installation

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone your chosen model repository
# (See quick start above)
```

### Step 3: Test GPU

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Test CUDA
x = torch.rand(5, 3).cuda()
print(f"Tensor on GPU: {x}")
```

### Step 4: Download Pretrained Models

```bash
# GPT-SoVITS
python download_models.py

# RVC
# Download from releases page or use WebUI

# Seed-VC
python download_pretrained.py

# SoftVC VITS
python download_pretrain.py
```

---

## API Server Setup

### Flask REST API (Simple)

```python
# server.py

from flask import Flask, request, send_file
from seed_vc import SeedVC
import soundfile as sf
import io
import torch

app = Flask(__name__)

# Load model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SeedVC("models/seed_vc.pt", device=device)

@app.route('/convert', methods=['POST'])
def convert():
    # Get files
    source = request.files['source']
    reference = request.files['reference']

    # Save temporarily
    source.save('/tmp/source.wav')
    reference.save('/tmp/reference.wav')

    # Convert
    output = model.convert(
        source_audio='/tmp/source.wav',
        target_audio='/tmp/reference.wav',
        auto_f0=True
    )

    # Return audio
    buffer = io.BytesIO()
    sf.write(buffer, output, 16000, format='WAV')
    buffer.seek(0)

    return send_file(buffer, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
```

**Run Server**:
```bash
python server.py
```

**Test**:
```bash
curl -X POST http://localhost:5000/convert \
  -F "source=@male_voice.wav" \
  -F "reference=@female_voice.wav" \
  --output converted.wav
```

---

### FastAPI (Production)

```python
# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from seed_vc import SeedVC
import soundfile as sf
import io
import torch

app = FastAPI()

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SeedVC("models/seed_vc.pt", device=device)

@app.post("/convert")
async def convert_voice(
    source: UploadFile = File(...),
    reference: UploadFile = File(...)
):
    # Read files
    source_bytes = await source.read()
    reference_bytes = await reference.read()

    # Save temporarily
    with open('/tmp/source.wav', 'wb') as f:
        f.write(source_bytes)
    with open('/tmp/reference.wav', 'wb') as f:
        f.write(reference_bytes)

    # Convert
    output = model.convert(
        source_audio='/tmp/source.wav',
        target_audio='/tmp/reference.wav',
        auto_f0=True
    )

    # Return streaming response
    buffer = io.BytesIO()
    sf.write(buffer, output, 16000, format='WAV')
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")

@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}
```

**Run with Uvicorn**:
```bash
pip install fastapi uvicorn python-multipart

uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**API Docs**: http://localhost:8000/docs

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-venv \
    ffmpeg \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models
RUN python3 download_models.py

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build image
docker build -t voice-conversion-server .

# Run container
docker run --gpus all -p 8000:8000 voice-conversion-server

# Or with docker-compose
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  voice-conversion:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
```

---

## Performance Optimization

### 1. Batch Processing

```python
# Process multiple files in parallel
import torch

# Enable batch inference
with torch.no_grad():
    outputs = model.batch_convert(
        source_audios=["file1.wav", "file2.wav", "file3.wav"],
        reference_audio="target.wav",
        batch_size=8
    )
```

### 2. Mixed Precision (FP16)

```python
# Use automatic mixed precision
from torch.cuda.amp import autocast

with autocast():
    output = model.convert(source, reference)
```

### 3. TensorRT Optimization

```python
# Convert model to TensorRT (fastest)
import torch_tensorrt

# Compile
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch.randn(1, 16000).cuda()],
    enabled_precisions={torch.float16}
)

# 2-3x faster inference
```

### 4. Request Queuing

```python
# Use celery for async task queue
from celery import Celery

app = Celery('voice_conversion', broker='redis://localhost:6379')

@app.task
def convert_voice(source_path, reference_path):
    output = model.convert(source_path, reference_path)
    return output
```

---

## Monitoring

### GPU Usage

```bash
# Install nvidia-smi
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

### Application Metrics

```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram, start_http_server

conversion_requests = Counter('voice_conversion_requests_total', 'Total conversions')
conversion_duration = Histogram('voice_conversion_duration_seconds', 'Conversion time')

@app.post("/convert")
async def convert_voice(...):
    conversion_requests.inc()

    with conversion_duration.time():
        output = model.convert(...)

    return output

# Start metrics server
start_http_server(9090)
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**:
```python
# Reduce batch size
# Clear cache between inferences
torch.cuda.empty_cache()

# Use gradient checkpointing
model.enable_gradient_checkpointing()
```

### Issue: Slow Inference

**Solution**:
- Use FP16 mixed precision
- Enable TensorRT
- Use smaller model variant
- Check GPU utilization (`nvidia-smi`)

### Issue: Poor Quality

**Solution**:
- Use higher quality reference audio (clean, no noise)
- Increase training data (for non-zero-shot models)
- Tune pitch shift parameters
- Try different models

---

## Cost Estimation (Cloud Deployment)

### AWS GPU Instances

| Instance | GPU | VRAM | Price/Hour | Use Case |
|----------|-----|------|------------|----------|
| g4dn.xlarge | T4 | 16GB | $0.526 | Development |
| g5.xlarge | A10G | 24GB | $1.006 | Production |
| p3.2xlarge | V100 | 16GB | $3.06 | High performance |
| g5.12xlarge | 4x A10G | 96GB | $5.672 | High scale |

### GCP GPU Instances

| Instance | GPU | Price/Hour |
|----------|-----|------------|
| n1-standard-4 + T4 | T4 | $0.35 + $0.35 |
| a2-highgpu-1g | A100 40GB | $3.673 |

**Cost Optimization**:
- Use spot instances (50-70% cheaper)
- Auto-scale based on demand
- Use CPU for preprocessing, GPU only for inference

---

## Production Checklist

- [ ] GPU server setup with CUDA/cuDNN
- [ ] Model installed and tested
- [ ] API server implemented
- [ ] Request queuing (Redis/Celery)
- [ ] Error handling and logging
- [ ] Health checks
- [ ] Monitoring (Prometheus/Grafana)
- [ ] Load testing (locust/k6)
- [ ] Auto-scaling configuration
- [ ] Backup and disaster recovery
- [ ] Security (API keys, rate limiting)
- [ ] Documentation

---

## Summary

**Best for Most Use Cases**: **GPT-SoVITS** or **RVC**

**Quick Setup Ranking**:
1. Seed-VC (15 min)
2. RVC (20 min)
3. GPT-SoVITS (30 min)

**Quality Ranking**:
1. GPT-SoVITS (4.65/5.0)
2. SoftVC VITS (4.50/5.0)
3. RVC (4.50/5.0)

**Speed Ranking**:
1. Seed-VC (50ms)
2. RVC (100ms)
3. SoftVC VITS (150ms)

**Recommended**: Start with **RVC** for balanced quality/speed, upgrade to **GPT-SoVITS** if you need the absolute best quality.

---

**Document Version**: 1.0
**Last Updated**: January 24, 2026
