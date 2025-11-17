import soundfile as sf
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_hubert_features(input_dir, output_dir, sample_rate=16000, device=None):
    """
    Extract HuBERT embeddings from all WAV files (recursively)
    and save as .npy files, preserving subfolder structure.
    Compatible with Windows (uses soundfile loader).
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Load pretrained HuBERT
    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model().to(device).eval()
    print(f"✅ Loaded HuBERT model ({bundle.sample_rate} Hz)")

    wav_files = list(input_dir.rglob("*.wav"))
    print(f"🔍 Found {len(wav_files)} WAV files under {input_dir}")
    if not wav_files:
        print("⚠️ No WAV files found — check path or folder structure.")
        return

    for wav_path in tqdm(wav_files, desc="Extracting HuBERT features"):
        try:
            # ---- safe loader ----
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            waveform = torch.tensor(audio).unsqueeze(0)
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            waveform = waveform.to(device)

            # ---- feature extraction ----
            with torch.inference_mode():
                feats, _ = model.extract_features(waveform)
                emb = feats[-1].squeeze(0).cpu().numpy()

            # ---- preserve folder structure ----
            rel_path = wav_path.relative_to(input_dir)
            subdir = output_dir / rel_path.parent
            subdir.mkdir(parents=True, exist_ok=True)
            out_path = subdir / f"{wav_path.stem}_hubert.npy"
            np.save(out_path, emb)

            print(f"💾 Saved: {out_path}")

        except Exception as e:
            print(f"❌ Error on {wav_path}: {e}")

    print(f"🎯 Features saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    extract_hubert_features(args.input_dir, args.output_dir, args.sample_rate, args.device)
