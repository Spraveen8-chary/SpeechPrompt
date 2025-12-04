import soundfile as sf
import torchaudio
from pathlib import Path
import torch

def convert_flac_to_wav(src_root="data/raw/librispeech/LibriSpeech/dev-clean", dst_root="data/raw/librispeech_wav/dev-clean"):
    src = Path(src_root)
    dst = Path(dst_root)
    dst.mkdir(parents=True, exist_ok=True)

    flac_files = list(src.rglob("*.flac"))
    print(f"Found {len(flac_files)} FLAC files to convert.")

    for flac_path in flac_files:
        audio, sr = sf.read(flac_path, dtype="float32")

        # convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # ensure 16k sample rate
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # save WAV using soundfile (NO torchaudio)
        out_path = dst / flac_path.relative_to(src)
        out_path = out_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, sr)

    print(f"✅ Converted all FLAC → WAV to: {dst}")


if __name__ == "__main__":
    convert_flac_to_wav()