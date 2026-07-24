# GPU setup (optional — for fast large-v3 transcription)

`faster-whisper` runs `large-v3` on an NVIDIA GPU via CTranslate2. On the target
machine (RTX 3060 Laptop 6GB, Windows 11) transcription runs at roughly RTF 0.45
(~80 min for a 3-hour meeting; faster on real meetings once silence is skipped).

Without a GPU, the pipeline still works — it falls back to CPU automatically
(`device="cpu"`, `compute_type="int8"`), just slower.

## One-time install (Windows, CUDA GPU)

CUDA-enabled PyTorch (used for GPU detection and pyannote), matching the driver's
CUDA version (13.0 on the target machine):

```powershell
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0+cu130" "torchaudio==2.11.0+cu130"
```

CUDA 12 cuBLAS + cuDNN 9 that CTranslate2 loads at runtime (these ship the
`cublas64_12.dll` / `cudnn_ops64_9.dll` that ctranslate2 needs; PyTorch's own
CUDA 13 libraries are not compatible with ctranslate2's CUDA 12 build):

```powershell
.\.venv\Scripts\python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

`src/transcribe.py` registers these libraries' `bin` directories as DLL search
paths automatically at model-load time, so no PATH changes are needed.

## Verify

```powershell
.\.venv\Scripts\python -c "import torch; print('cuda:', torch.cuda.is_available())"
```
Should print `cuda: True`. If it prints `False`, transcription falls back to CPU.
