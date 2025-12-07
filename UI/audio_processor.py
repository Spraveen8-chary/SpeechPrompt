import subprocess
import os
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from speechbrain.pretrained import EncoderClassifier


# -------------------------------------------------------
# 1) Convert ANY audio file into SpeechBrain-compatible WAV
# -------------------------------------------------------
def convert_to_speechbrain_wav(input_path):
    base, _ = os.path.splitext(input_path)
    output_path = base + "_sb.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz
        "-sample_fmt", "s16",# PCM 16-bit
        output_path
    ]

    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{process.stderr.decode()}")

    return output_path


# -------------------------------------------------------
# 2) ASR Model (LibriSpeech – Wav2Vec2 Base 960h)
# -------------------------------------------------------
print("Loading ASR model (Wav2Vec2 960h)...")
asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


def run_asr(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = asr_processor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = asr_model(inputs.input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    text = asr_processor.batch_decode(pred_ids)[0]

    return text


# -------------------------------------------------------
# 3) Classification Model (Google Speech Commands)
# -------------------------------------------------------
print("Loading Speech Commands model...")
gsc_model = EncoderClassifier.from_hparams(
    source="speechbrain/google_speech_command_xvector",
    savedir="pretrained_gsc"
)

def run_classification(audio_path):
    out_prob, score, index = gsc_model.classify_file(audio_path)
    label = gsc_model.hparams.labels[index]
    return label, float(score)


# -------------------------------------------------------
# 4) MAIN FUNCTION – Upload, Convert & Process
# -------------------------------------------------------
def process_audio_file(path):
    print(f"\nUploading file: {path}")

    # Convert audio
    converted = convert_to_speechbrain_wav(path)
    print(f"Converted to: {converted}")

    # Classification
    label, conf = run_classification(converted)

    # ASR transcription
    transcription = run_asr(converted)

    print("\n------------------------------")
    print("CLASSIFICATION (GSC MODEL)")
    print(f"Label: {label}")
    print(f"Confidence: {conf:.4f}")

    print("\nASR (LibriSpeech Wav2Vec2)")
    print(f"Transcription: {transcription}")
    print("------------------------------")


# -------------------------------------------------------
# 5) INPUT HANDLER
# -------------------------------------------------------
if __name__ == "__main__":
    file_path = input("Enter the audio file path: ").strip()

    if not os.path.exists(file_path):
        print("File not found!")
        exit()

    process_audio_file(file_path)
