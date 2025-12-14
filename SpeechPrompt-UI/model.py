# import torch
# import torchaudio
# import numpy as np

# from speechbrain.inference import EncoderDecoderASR, Tacotron2, HIFIGAN
# from transformers import (
#     AutoModelForAudioClassification,
#     AutoFeatureExtractor,
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )

# # -------------------------------------------------------------------
# # DEVICE SETUP
# # -------------------------------------------------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("Running on:", DEVICE)

# # ===============================================================
# # 1. LOAD ALL MODELS
# # ===============================================================

# print("\nLoading ASR model (SpeechBrain)...")
# asr_model = EncoderDecoderASR.from_hparams(
#     source="speechbrain/asr-transformer-transformerlm-librispeech",
#     savedir="pretrained_asr"
# )

# # ---------- EMOTION ----------
# print("Loading Emotion classifier...")
# emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(
#     "superb/wav2vec2-base-superb-er"
# )
# emotion_model = AutoModelForAudioClassification.from_pretrained(
#     "superb/wav2vec2-base-superb-er"
# ).to(DEVICE)

# # ---------- INTENT ----------
# print("Loading Intent classifier...")
# intent_feature_extractor = AutoFeatureExtractor.from_pretrained(
#     "superb/hubert-base-superb-ic"
# )
# intent_model = AutoModelForAudioClassification.from_pretrained(
#     "superb/hubert-base-superb-ic"
# ).to(DEVICE)

# # ---------- SPEECH CATEGORY ----------
# print("Loading Speech Category classifier...")
# category_feature_extractor = AutoFeatureExtractor.from_pretrained(
#     "superb/wav2vec2-base-superb-ks"
# )
# category_model = AutoModelForAudioClassification.from_pretrained(
#     "superb/wav2vec2-base-superb-ks"
# ).to(DEVICE)

# # ---------- TTS ----------
# print("Loading TTS (Tacotron2 + HiFiGAN)...")
# tts = Tacotron2.from_hparams(
#     source="speechbrain/tts-tacotron2-ljspeech",
#     savedir="pretrained_tts"
# )
# vocoder = HIFIGAN.from_hparams(
#     source="speechbrain/tts-hifigan-ljspeech",
#     savedir="pretrained_vocoder"
# )

# # ---------- GPT-2 LLM ----------
# print("Loading GPT-2 for text continuation...")
# lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# lm_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)


# # ===============================================================
# # UNIVERSAL AUDIO LOADER
# # ===============================================================
# def load_audio_for_transformers(path, target_sr=16000):
#     wav, sr = torchaudio.load(path)

#     if wav.size(0) > 1:
#         wav = wav.mean(dim=0)

#     wav = wav.squeeze(0)

#     if sr != target_sr:
#         wav = torchaudio.functional.resample(wav, sr, target_sr)

#     return wav.numpy(), target_sr


# # ===============================================================
# # EMOTION / INTENT / CATEGORY FUNCTIONS
# # ===============================================================
# def classify_emotion(audio_path):
#     wav, sr = load_audio_for_transformers(audio_path)
#     inputs = emotion_feature_extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
#     with torch.no_grad():
#         logits = emotion_model(**inputs).logits
#     pred = torch.argmax(logits, dim=-1).item()
#     return emotion_model.config.id2label[pred]


# def classify_intent(audio_path):
#     wav, sr = load_audio_for_transformers(audio_path)
#     inputs = intent_feature_extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
#     with torch.no_grad():
#         logits = intent_model(**inputs).logits
#     pred = torch.argmax(logits, dim=-1).item()
#     return intent_model.config.id2label[pred]


# def classify_category(audio_path):
#     wav, sr = load_audio_for_transformers(audio_path)
#     inputs = category_feature_extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
#     with torch.no_grad():
#         logits = category_model(**inputs).logits
#     pred = torch.argmax(logits, dim=-1).item()
#     return category_model.config.id2label[pred]


# # ===============================================================
# # GPT-2 TEXT CONTINUATION
# # ===============================================================
# def generate_continuation(text, max_new_tokens=60):
#     inputs = lm_tokenizer(text, return_tensors="pt").to(DEVICE)

#     output = lm_model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         top_p=0.92,
#         top_k=40,
#         temperature=0.8,
#         eos_token_id=lm_tokenizer.eos_token_id
#     )

#     return lm_tokenizer.decode(output[0], skip_special_tokens=True)


# # ===============================================================
# # TTS – SPEECH GENERATION
# # ===============================================================
# def generate_speech(text, output):
#     mel_outputs = tts.encode_text(text)
#     mel = mel_outputs[0] if isinstance(mel_outputs, tuple) else mel_outputs
#     wave = vocoder.decode_batch(mel).squeeze(0)
#     torchaudio.save(output, wave.cpu(), 22050)
#     return output


# # ===============================================================
# # MAIN PIPELINE
# # ===============================================================
# def run_all(audio_path):
#     print("\n=============== PROCESSING ===============")

#     # ---------- ASR ----------
#     asr_text = asr_model.transcribe_file(audio_path)
#     print("ASR:", asr_text)

#     # ---------- Emotion / Intent / Category ----------
#     emotion = classify_emotion(audio_path)
#     intent = classify_intent(audio_path)
#     category = classify_category(audio_path)

#     # ---------- GPT-2 continuation ----------
#     continuation_text = generate_continuation(asr_text, max_new_tokens=80)
#     print("GPT-2 continuation:", continuation_text)

#     # ---------- SAVE ASR TEXT AS SPEECH ----------
#     asr_audio = generate_speech(asr_text, "asr_original.wav")

#     # ---------- SAVE ONLY GPT-2 GENERATED PART ----------
#     continuation_audio = generate_speech(continuation_text, "gpt2_continuation.wav")

#     # ---------- RETURN EVERYTHING ----------
#     return {
#         "asr_text": asr_text,
#         "continuation_text": continuation_text,
#         "emotion": emotion,
#         "intent": intent,
#         "category": category,
#         "asr_audio_file": asr_audio,
#         "continuation_audio_file": continuation_audio,
#     }


# # ===============================================================
# # RUN PIPELINE
# # ===============================================================
# if __name__ == "__main__":
#     result = run_all("./audios/test15.wav")
#     print("\nFINAL RESULT:", result)

import os
import torch
import torchaudio
import logging
from datetime import datetime

from speechbrain.inference import EncoderDecoderASR, Tacotron2, HIFIGAN
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# ================================================================
# DEVICE SETUP
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# PATHS
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_AUDIO_DIR = os.path.join(BASE_DIR, "uploads", "audio")
OUTPUT_SB_AUDIO_DIR = os.path.join(BASE_DIR, "outputs", "audio", "speechbrain")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(UPLOAD_AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_SB_AUDIO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ================================================================
# LOGGING (FILE ONLY)
# ================================================================
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s [SB] %(levelname)s: %(message)s"
)
log = logging.getLogger("speechbrain")


# ================================================================
# LOAD SPEECHBRAIN MODELS
# ================================================================
log.info("Loading SpeechBrain ASR model...")
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-transformer-transformerlm-librispeech",
    savedir="pretrained_asr"
)

log.info("Loading emotion, intent, category models...")
emotion_extractor = AutoFeatureExtractor.from_pretrained(
    "superb/wav2vec2-base-superb-er"
)
emotion_model = AutoModelForAudioClassification.from_pretrained(
    "superb/wav2vec2-base-superb-er"
).to(DEVICE)

intent_extractor = AutoFeatureExtractor.from_pretrained(
    "superb/hubert-base-superb-ic"
)
intent_model = AutoModelForAudioClassification.from_pretrained(
    "superb/hubert-base-superb-ic"
).to(DEVICE)

category_extractor = AutoFeatureExtractor.from_pretrained(
    "superb/wav2vec2-base-superb-ks"
)
category_model = AutoModelForAudioClassification.from_pretrained(
    "superb/wav2vec2-base-superb-ks"
).to(DEVICE)

log.info("Loading SpeechBrain TTS models...")
tts = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech",
    savedir="pretrained_tts"
)
vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_vocoder"
)

# ================================================================
# HELPERS
# ================================================================
def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0)

    wav = wav.squeeze(0)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav.numpy(), target_sr


def classify(model, extractor, audio_path):
    wav, sr = load_audio(audio_path)
    inputs = extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    pred = torch.argmax(logits, dim=-1).item()
    return model.config.id2label[pred]


def generate_sb_tts(text):
    """Generates Tacotron2 + HiFiGAN audio and saves it."""
    mel_outputs = tts.encode_text(text)
    mel = mel_outputs[0] if isinstance(mel_outputs, tuple) else mel_outputs
    audio = vocoder.decode_batch(mel).squeeze(0)

    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_sb.wav"
    out_path = os.path.join(OUTPUT_SB_AUDIO_DIR, filename)

    torchaudio.save(out_path, audio.cpu(), 22050)

    return filename


# ================================================================
# MAIN API FUNCTION — Called from app.py
# ================================================================
def run_speechbrain(audio_path):
    """
    Runs SpeechBrain processing and returns:
    - ASR
    - Emotion
    - Intent
    - Category
    - TTS audio file generated
    """

    log.info(f"[SB] Processing audio: {audio_path}")

    # ---------- ASR ----------
    sb_asr_text = asr_model.transcribe_file(audio_path)
    log.info(f"[SB] ASR: {sb_asr_text}")

    # ---------- CLASSIFICATIONS ----------
    sb_emotion = classify(emotion_model, emotion_extractor, audio_path)
    sb_intent = classify(intent_model, intent_extractor, audio_path)
    sb_category = classify(category_model, category_extractor, audio_path)

    log.info(f"[SB] Emotion={sb_emotion}, Intent={sb_intent}, Category={sb_category}")

    # ---------- TTS ----------
    sb_audio_file = generate_sb_tts(sb_asr_text)

    return {
        "sb_asr_text": sb_asr_text,
        "sb_emotion": sb_emotion,
        "sb_intent": sb_intent,
        "sb_category": sb_category,
        "sb_audio_file": sb_audio_file,
    }
