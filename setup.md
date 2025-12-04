Below is a **clean, professional, end-to-end `SETUP.md`** containing:

* Environment setup
* Dependencies
* Downloading LibriSpeech
* Extracting features
* Running K-Means
* Generating unit files
* Preparing ASR transcripts
* Training Speech Classification
* Training ASR
* Folder structure
* Verification steps

No emojis.
Fully technical.
Ready for GitHub.
Formatted in Markdown.

---

# SETUP.md

PromptSpeech: Full Environment Setup and Data Preparation Guide
Windows / Linux Compatible

---

## 1. Project Structure

Create a clean project directory:

```
PromptSpeech/
    configs/
    data/
        raw/
        units/
        processed/
    results/
        checkpoints/
    scripts/
    src/
```

You may clone an existing repository or create the structure manually.

---

## 2. Python Environment Setup

Use Python 3.10.

```
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux
```

Upgrade tools:

```
python -m pip install --upgrade "pip<24.1" setuptools wheel ninja
```

---

## 3. Install PyTorch and Torchaudio

### GPU (CUDA 12.1)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CPU-only

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify installation:

```
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

---

## 4. Install Required Python Packages

```
pip install transformers scikit-learn librosa soundfile numpy pandas tqdm jiwer sacrebleu matplotlib
```

Optional for development:

```
pip install jupyter notebook
```

---

## 5. Download LibriSpeech

Download the subset you need (for example, dev-clean):

```
mkdir -p data/raw/librispeech
cd data/raw/librispeech
```

Download from:

[https://www.openslr.org/12/](https://www.openslr.org/12/)

Extract downloaded archives inside:

```
data/raw/librispeech/LibriSpeech/dev-clean/
```

The folder will contain:

```
*.flac audio files
*.trans.txt transcription files
```

---

## 6. Extract HuBERT SSL Features

Create script: `src/preprocessing/extract_features.py`

Example command to generate frame-level features:

```
python src/preprocessing/extract_features.py \
    --input_dir data/raw/librispeech/LibriSpeech/dev-clean \
    --output_dir data/processed/librispeech/features \
    --model hubert_base
```

---

## 7. Train K-Means on SSL Features

```
python src/preprocessing/train_kmeans.py \
    --feature_dir data/processed/librispeech/features \
    --k 100 \
    --save_path data/kmeans/kmeans_100.pkl
```

---

## 8. Quantize SSL Features into Discrete Units

```
python src/preprocessing/quantize_units.py \
    --feature_dir data/processed/librispeech/features \
    --kmeans_path data/kmeans/kmeans_100.pkl \
    --output_dir data/units/libri/dev-clean
```

This produces files like:

```
8842-304647-0009_units.npy
8842-304647-0010_units.npy
...
```

---

## 9. Prepare ASR Transcripts (Required for ASR Training)

LibriSpeech stores transcription text in grouped files:

```
xxxx-xxxx.trans.txt
```

Each file contains multiple lines such as:

```
8842-304647-0009 the transcript text here
8842-304647-0010 another sentence here
```

However, ASR training requires per-utterance files:

```
8842-304647-0009_units.npy
8842-304647-0009.trans.txt
```

Use this script (`scripts/prepare_librispeech_transcripts.py`) to generate per-file transcriptions:

```
import os
from pathlib import Path

UNITS_DIR = Path("data/units/libri/dev-clean")
TRANSCRIPT_ROOT = Path("data/raw/librispeech/LibriSpeech/dev-clean")

print("Scanning LibriSpeech transcripts...")

utt2text = {}
for trans_file in TRANSCRIPT_ROOT.rglob("*.trans.txt"):
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            utt_id, text = parts[0], parts[1]
            utt2text[utt_id] = text

print(f"Loaded {len(utt2text)} transcriptions")

counter = 0

for units_file in UNITS_DIR.rglob("*.npy"):
    stem = units_file.stem
    utt_id = stem.replace("_units", "").replace(".units", "")

    if utt_id not in utt2text:
        print(f"No transcript found for {utt_id}")
        continue

    out_path = units_file.with_suffix(".trans.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(utt2text[utt_id])

    counter += 1

print(f"\nSuccessfully created {counter} .trans.txt files in {UNITS_DIR}")
```

Run the script:

```
python scripts/prepare_librispeech_transcripts.py
```

You should now have:

```
8842-304647-0009_units.npy
8842-304647-0009.trans.txt
...
```

---

## 10. Training: Speech Classification (Google Speech Commands)

Configuration: `configs/train_classification.yaml`

```
task: speech_classification
gslm_ckpt: pretrained/hubert100_lm/checkpoint_best_fixed.pt
vocab_size: 100
prompt_len: 20
hidden_dim: 1024
lr: 5e-3
batch_size: 16
epochs: 1
patience: 3
device: auto
prompt_type: input
fast_debug: true
samples_per_class: 30
datasets:
  - name: speech_commands
    path: data/processed/speech_commands/units
    num_classes: 36
save_dir: results/checkpoints/speech_classification
```

Run:

```
python -m src.training.train_prompt --config configs/train_classification.yaml
```

Checkpoint produced:

```
results/checkpoints/speech_classification/promptspeech_best.pt
```

---

## 11. Training: ASR

Configuration: `configs/train_asr.yaml`

```
task: asr
gslm_ckpt: pretrained/hubert100_lm/checkpoint_best_fixed.pt
vocab_size: 100
hidden_dim: 1024
prompt_len: 50
lr: 5e-3
batch_size: 8
epochs: 1
patience: 3
device: auto
prompt_type: deep
fast_debug: true
samples_per_class: 50
datasets:
  - name: librispeech_dev_clean
    path: data/units/libri/dev-clean
save_dir: results/checkpoints/asr_librispeech
```

Run:

```
python -m src.training.train_prompt --config configs/train_asr.yaml
```

Requirements for success:

1. `*.units.npy` and `*.trans.txt` must exist side-by-side.
2. Dataset must include all required pairs.
3. Script automatically loads transcript pairs using `UnitSeqDataset`.

Checkpoint produced:

```
results/checkpoints/asr_librispeech/promptspeech_best.pt
```

---

## 12. Results Folder Structure

Expected layout:

```
results/
    checkpoints/
        speech_classification/
            promptspeech_best.pt
        asr_librispeech/
            promptspeech_best.pt
```

---

## 13. Verification Steps

### Verify transcripts were created:

```
dir data/units/libri/dev-clean/*.trans.txt
```

### Verify ASR dataset loads correctly:

```
python scripts/prepare_librispeech_transcripts.py
```

Should output non-zero matches.

### Verify training:

```
python -m src.training.train_prompt --config configs/train_asr.yaml
```

Should not crash and should begin ASR training.

---

## 14. Troubleshooting

### Issue: No transcript found

Cause: File naming mismatch.
Fix: Ensure filenames follow LibriSpeech style:

```
ID:      8842-304647-0009
Units:   8842-304647-0009_units.npy
Text:    8842-304647-0009.trans.txt
```

### Issue: 0 matches created

Cause: `glob` path not recursive.
Fix: Use `rglob("*.npy")`.

### Issue: Fairseq extension compilation errors

Cause: Windows cannot compile C++ ops.
Fix: Use torchaudio HuBERT; do not install fairseq from pip.

---

## 15. Summary

This setup guide includes:

1. Environment creation
2. Dependency installation
3. LibriSpeech download
4. Feature extraction
5. K-Means clustering
6. Unit quantization
7. Transcript preparation
8. Classification training
9. ASR training
10. Troubleshooting

This defines the full workflow required to run PromptSpeech end-to-end on Windows or Linux.

---

If you want, I can also generate:

* TRAINING.md
* INFERENCE.md
* DATA_PREPARATION.md
* End-to-end shell scripts

Just ask.
