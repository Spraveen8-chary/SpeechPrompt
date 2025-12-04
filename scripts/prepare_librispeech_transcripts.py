import os
from pathlib import Path

# Path where YOUR units.npy files are
UNITS_DIR = Path("data/units/libri/dev-clean")

# Path to original LibriSpeech transcripts
TRANSCRIPT_ROOT = Path("data/raw/librispeech/LibriSpeech/dev-clean")

print("🔍 Scanning LibriSpeech transcripts...")

# 1. Build mapping: utt_id -> transcript_text
utt2text = {}

# Find all transcript files recursively
for trans_file in TRANSCRIPT_ROOT.rglob("*.trans.txt"):
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split(" ", 1)
            utt_id = parts[0]                  # example: 1272-128104-0000
            text = parts[1]
            utt2text[utt_id] = text

print(f"📄 Loaded {len(utt2text)} transcriptions")

# 2. Match with ALL units recursively
counter = 0

# Match ANY .npy under the directory
for units_file in UNITS_DIR.rglob("*.npy"):
    stem = units_file.stem  # filename without extension
    # Remove optional ".units"
    utt_id = stem.replace("_units", "")

    if utt_id not in utt2text:
        print(f"⚠️ No transcript found for {utt_id}")
        continue

    # Write .trans.txt next to the .npy file
    out_txt = units_file.with_suffix(".trans.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(utt2text[utt_id])

    counter += 1

print(f"\n✅ Successfully created {counter} .trans.txt files")
print(f"   Inside: {UNITS_DIR}")
print("🎉 ASR dataset is now READY for training!")
