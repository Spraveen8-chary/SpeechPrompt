import torchaudio
from pathlib import Path

def convert_flac_to_wav(src_root="data/raw/librispeech/LibriSpeech", dst_root="data/raw/librispeech_wav"):
    src = Path(src_root)
    dst = Path(dst_root)
    dst.mkdir(parents=True, exist_ok=True)

    flac_files = list(src.rglob("*.flac"))
    print(f"Found {len(flac_files)} FLAC files to convert.")

    for flac_path in flac_files:
        waveform, sr = torchaudio.load(flac_path)
        out_path = dst / flac_path.relative_to(src)
        out_path = out_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(out_path, waveform, sr)

    print(f"✅ Converted all FLAC → WAV files to: {dst}")

if __name__ == "__main__":
    convert_flac_to_wav()
