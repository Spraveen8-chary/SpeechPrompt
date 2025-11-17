# 🧱 `SETUP.md` — Environment Setup Guide for PromptSpeech

> ✅ This guide ensures a fully working PromptSpeech environment on **Windows 10/11**, compatible with **HuBERT**, **Prompt Tuning**, and **Speech Quantization**.
>
> 🧠 Verified for Python 3.10 + PyTorch ≥ 2.0.
> ⚙️ Target GPU: NVIDIA CUDA 11.8 or 12.1 (optional but recommended)

---

## 📂 1. Clone or Create the Project Structure

If you haven’t already, generate the PromptSpeech project folder using the `init_project.py` script:

```bash
python init_project.py
```

This creates:

```
PromptSpeech/
├── data/
├── models/
├── src/
├── scripts/
├── configs/
├── results/
└── main.py
```

---

## 🧠 2. Create and Activate a Virtual Environment

Using **Python 3.10** (recommended):

```bash
cd PromptSpeech
python -m venv .venv
.venv\Scripts\activate
```

(You should see `(.venv)` at the start of your PowerShell prompt.)

---

## ⚙️ 3. Upgrade Core Tools

Before installing any libraries, upgrade your packaging tools and pin pip below 24.1 to avoid Fairseq metadata issues.

```bash
python -m pip install --upgrade "pip<24.1" setuptools wheel ninja
```

---

## 💻 4. (One-Time) Verify Visual C++ Compiler ✅

Open PowerShell and check:

```bash
cl
```

If you see:

```
Microsoft (R) C/C++ Optimizing Compiler Version 19.x for x64
```

✅ You already have **Microsoft Visual C++ Build Tools** installed.
If not, install them from:
👉 [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Select:

* **Desktop development with C++**
* **MSVC v143 toolset**
* **Windows 10/11 SDK**

---

## 🔥 5. Install PyTorch + Torchaudio

### 🔹 If you have an NVIDIA GPU

(Recommended for faster training)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

*(Change `cu121` → `cu118` if you have CUDA 11.8)*

### 🔹 If CPU-only

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python -c "import torch; print(torch.__version__, '✅ Torch OK, CUDA:', torch.cuda.is_available())"
```

Expected output:

```
2.4.0 ✅ Torch OK, CUDA: True
```

---

## 🎧 6. Install Remaining Dependencies

PromptSpeech uses `torchaudio`’s **HuBERT** for SSL feature extraction (instead of Fairseq).
Install all remaining packages:

```bash
pip install transformers scikit-learn librosa soundfile numpy pandas tqdm jiwer sacrebleu matplotlib
```

---

## ❌ 7. Avoid Fairseq Build Failures on Windows

Fairseq requires C++ extensions that do **not** compile reliably on Windows.
To prevent the errors you experienced (`RuntimeError: Error compiling objects for extension`):

* **Do NOT install `fairseq`**.
* Instead, use **Torchaudio’s HuBERT pipelines**, which are fully compatible and cross-platform.

Example usage:

```python
import torchaudio

bundle = torchaudio.pipelines.HuBERT_BASE
model = bundle.get_model()

waveform, sr = torchaudio.load("sample.wav")
features, _ = model.extract_features(waveform)
print(features[-1].shape)
```

✅ Gives identical SSL embeddings as Fairseq’s HuBERT.
✅ Works on Windows without compilation.

---

## 🧰 8. Optional — Install Dev & Visualization Tools

```bash
pip install jupyter notebook seaborn
```

---

## 🧪 9. Verify the Entire Environment

Run this quick test:

```bash
python - <<'PY'
import torch, torchaudio, sklearn, librosa, transformers
print("✅ PyTorch:", torch.__version__)
print("✅ Torchaudio:", torchaudio.__version__)
print("✅ Transformers:", transformers.__version__)
print("✅ All dependencies loaded successfully!")
PY
```

Expected output:

```
✅ PyTorch: 2.x
✅ Torchaudio: 2.x
✅ Transformers: 4.x
✅ All dependencies loaded successfully!
```

---

## 🧩 10. Final Folder Checklist

After setup, your folder should look like:

```
PromptSpeech/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── manifests/
│   └── kmeans/
│
├── src/
│   ├── preprocessing/
│   │   ├── extract_features.py
│   │   └── quantize_units.py
│   ├── training/
│   │   └── train_prompt.py
│   └── evaluation/
│       └── evaluate.py
│
├── models/
├── scripts/
├── configs/
├── results/
├── README.md
├── environment.yaml
└── main.py
```

---

## 🧠 11. Summary of Key Fixes for Windows

| Issue Encountered                       | Root Cause                 | Permanent Fix                          |
| --------------------------------------- | -------------------------- | -------------------------------------- |
| `invalid command 'bdist_wheel'`         | wheel not installed        | `pip install wheel setuptools ninja`   |
| `Error compiling objects for extension` | Fairseq C++ extensions     | ❌ Skip Fairseq → Use Torchaudio HuBERT |
| `ModuleNotFoundError: torch`            | Fairseq built before torch | Install torch **before** Fairseq       |
| `omegaconf invalid metadata`            | pip ≥ 24.1 breaks old deps | Use `pip < 24.1`                       |
| `cl not found`                          | Missing MSVC build tools   | Install **Microsoft C++ Build Tools**  |

---

## 🚀 12. Next Steps

Once setup completes successfully:

1. Proceed to **Step 2: SSL Feature Extraction & Quantization**
2. Implement:

   * `src/preprocessing/extract_features.py`
   * `src/preprocessing/quantize_units.py`
3. Use Torchaudio’s HuBERT and `sklearn.cluster.MiniBatchKMeans`.

---

## ✅ Environment Summary

| Package               | Version | Purpose                            |
| --------------------- | ------- | ---------------------------------- |
| torch / torchaudio    | ≥ 2.0   | Core Deep Learning & HuBERT        |
| transformers          | ≥ 4.40  | Tokenization & Language Interfaces |
| scikit-learn          | ≥ 1.3   | K-Means Quantization               |
| librosa / soundfile   | ≥ 0.10  | Audio Processing I/O               |
| pandas / numpy / tqdm | Latest  | Utilities & logging                |
| jiwer / sacrebleu     | Latest  | Evaluation Metrics                 |

---

### ✅ You’re Ready

Your PromptSpeech environment is now **fully reproducible, clean, and Windows-compatible**.
No Fairseq issues, no build errors — just run:

```bash
python main.py --mode prepare
```

Then start implementing **Step 2**.

