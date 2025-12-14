# import torch
# import torchaudio
# import numpy as np

# from speechbrain.inference import EncoderDecoderASR, Tacotron2, HIFIGAN
# from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# # -------------------------------------------------------------------
# # DEVICE SETUP
# # -------------------------------------------------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("Running on:", DEVICE)


# # -------------------------------------------------------------------
# # 1. LOAD MODELS
# # -------------------------------------------------------------------

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


# # -------------------------------------------------------------------
# # UNIVERSAL AUDIO LOADER
# # -------------------------------------------------------------------
# def load_audio_for_transformers(path, target_sr=16000):
#     wav, sr = torchaudio.load(path)

#     # Mix stereo → mono
#     if wav.size(0) > 1:
#         wav = wav.mean(dim=0)

#     wav = wav.squeeze(0)

#     # Resample to 16kHz
#     if sr != target_sr:
#         wav = torchaudio.functional.resample(wav, sr, target_sr)

#     return wav.numpy(), target_sr


# # -------------------------------------------------------------------
# # 2. EMOTION CLASSIFICATION
# # -------------------------------------------------------------------
# def classify_emotion(audio_path):
#     wav, sr = load_audio_for_transformers(audio_path)
#     inputs = emotion_feature_extractor(
#         wav, sampling_rate=sr, return_tensors="pt", padding=True
#     )
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

#     with torch.no_grad():
#         logits = emotion_model(**inputs).logits

#     pred = torch.argmax(logits, dim=-1).item()
#     label = emotion_model.config.id2label[pred]
#     return label


# # -------------------------------------------------------------------
# # 3. INTENT CLASSIFICATION
# # -------------------------------------------------------------------
# def classify_intent(audio_path):
#     wav, sr = load_audio_for_transformers(audio_path)
#     inputs = intent_feature_extractor(
#         wav, sampling_rate=sr, return_tensors="pt", padding=True
#     )
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

#     with torch.no_grad():
#         logits = intent_model(**inputs).logits

#     pred = torch.argmax(logits, dim=-1).item()
#     label = intent_model.config.id2label[pred]
#     return label


# # -------------------------------------------------------------------
# # 4. SPEECH CATEGORY CLASSIFICATION
# # -------------------------------------------------------------------
# def classify_category(audio_path):
#     wav, sr = load_audio_for_transformers(audio_path)
#     inputs = category_feature_extractor(
#         wav, sampling_rate=sr, return_tensors="pt", padding=True
#     )
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

#     with torch.no_grad():
#         logits = category_model(**inputs).logits

#     pred = torch.argmax(logits, dim=-1).item()
#     label = category_model.config.id2label[pred]
#     return label


# # -------------------------------------------------------------------
# # 5. SPEECH GENERATION (TTS)
# # -------------------------------------------------------------------
# def generate_speech(text, output="generated.wav"):
#     mel_outputs = tts.encode_text(text)

#     # Tacotron2 returns (mel, alignment)
#     if isinstance(mel_outputs, tuple):
#         mel = mel_outputs[0]
#     else:
#         mel = mel_outputs

#     wave = vocoder.decode_batch(mel)

#     # Remove batch dimension → must be [C, T]
#     wave = wave.squeeze(0)

#     torchaudio.save(output, wave.cpu(), 22050)
#     return output


# # -------------------------------------------------------------------
# # MAIN PIPELINE
# # -------------------------------------------------------------------
# def run_all(audio_path):
#     print("\n====================================")
#     print("Processing:", audio_path)
#     print("====================================")

#     # 1. ASR
#     print("\n⚡ Running ASR...")
#     asr_text = asr_model.transcribe_file(audio_path)
#     print("ASR Text:", asr_text)

#     # 2. Emotion
#     print("\n⚡ Detecting Emotion...")
#     emotion = classify_emotion(audio_path)
#     print("Emotion:", emotion)

#     # 3. Intent
#     print("\n⚡ Detecting Intent...")
#     intent = classify_intent(audio_path)
#     print("Intent:", intent)

#     # 4. Category
#     print("\n⚡ Detecting Speech Category...")
#     category = classify_category(audio_path)
#     print("Speech Category:", category)

#     # 5. TTS
#     print("\n⚡ Generating speech from ASR output text...")
#     tts_file = generate_speech(asr_text)
#     print("Generated Speech File:", tts_file)

#     print("\n=== FINAL RESULT ===")
#     result = {
#         "asr_text": asr_text,
#         "emotion": emotion,
#         "intent": intent,
#         "category": category,
#         "tts_file": tts_file
#     }

#     print(result)
#     return result


# # -------------------------------------------------------------------
# # RUN
# # -------------------------------------------------------------------
# if __name__ == "__main__":
#     run_all("./audios/test6.wav")

import torch
import torchaudio
import numpy as np

from speechbrain.inference import EncoderDecoderASR, Tacotron2, HIFIGAN
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# -------------------------------------------------------------------
# DEVICE SETUP
# -------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE)

# ===============================================================
# 1. LOAD ALL MODELS
# ===============================================================

print("\nLoading ASR model (SpeechBrain)...")
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-transformer-transformerlm-librispeech",
    savedir="pretrained_asr"
)

# ---------- EMOTION ----------
print("Loading Emotion classifier...")
emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(
    "superb/wav2vec2-base-superb-er"
)
emotion_model = AutoModelForAudioClassification.from_pretrained(
    "superb/wav2vec2-base-superb-er"
).to(DEVICE)

# ---------- INTENT ----------
print("Loading Intent classifier...")
intent_feature_extractor = AutoFeatureExtractor.from_pretrained(
    "superb/hubert-base-superb-ic"
)
intent_model = AutoModelForAudioClassification.from_pretrained(
    "superb/hubert-base-superb-ic"
).to(DEVICE)

# ---------- SPEECH CATEGORY ----------
print("Loading Speech Category classifier...")
category_feature_extractor = AutoFeatureExtractor.from_pretrained(
    "superb/wav2vec2-base-superb-ks"
)
category_model = AutoModelForAudioClassification.from_pretrained(
    "superb/wav2vec2-base-superb-ks"
).to(DEVICE)

# ---------- TTS ----------
print("Loading TTS (Tacotron2 + HiFiGAN)...")
tts = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech",
    savedir="pretrained_tts"
)
vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_vocoder"
)

# ---------- LLM: GPT-2 FOR TEXT CONTINUATION ----------
print("Loading GPT-2 for text continuation...")
lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
lm_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)


# ===============================================================
# UNIVERSAL AUDIO LOADER
# ===============================================================
def load_audio_for_transformers(path, target_sr=16000):
    wav, sr = torchaudio.load(path)

    # Mix stereo → mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0)

    wav = wav.squeeze(0)

    # Resample to 16 kHz
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav.numpy(), target_sr


# ===============================================================
# 2. EMOTION CLASSIFICATION
# ===============================================================
def classify_emotion(audio_path):
    wav, sr = load_audio_for_transformers(audio_path)
    inputs = emotion_feature_extractor(
        wav, sampling_rate=sr, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = emotion_model(**inputs).logits

    pred = torch.argmax(logits, dim=-1).item()
    label = emotion_model.config.id2label[pred]
    return label


# ===============================================================
# 3. INTENT CLASSIFICATION
# ===============================================================
def classify_intent(audio_path):
    wav, sr = load_audio_for_transformers(audio_path)
    inputs = intent_feature_extractor(
        wav, sampling_rate=sr, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = intent_model(**inputs).logits

    pred = torch.argmax(logits, dim=-1).item()
    label = intent_model.config.id2label[pred]
    return label


# ===============================================================
# 4. SPEECH CATEGORY CLASSIFICATION
# ===============================================================
def classify_category(audio_path):
    wav, sr = load_audio_for_transformers(audio_path)
    inputs = category_feature_extractor(
        wav, sampling_rate=sr, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = category_model(**inputs).logits

    pred = torch.argmax(logits, dim=-1).item()
    label = category_model.config.id2label[pred]
    return label


# ===============================================================
# 5. TEXT CONTINUATION USING GPT-2
# ===============================================================
def generate_continuation(text, max_new_tokens=60):
    print(f"\nLLM Input: {text}")

    inputs = lm_tokenizer(text, return_tensors="pt").to(DEVICE)

    output = lm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.92,
        top_k=40,
        temperature=0.8,
        eos_token_id=lm_tokenizer.eos_token_id
    )

    generated = lm_tokenizer.decode(output[0], skip_special_tokens=True)

    print("\nLLM Output:", generated)
    return generated


# ===============================================================
# 6. SPEECH GENERATION (TTS)
# ===============================================================
def generate_speech(text, output="generated.wav"):
    mel_outputs = tts.encode_text(text)

    if isinstance(mel_outputs, tuple):
        mel = mel_outputs[0]
    else:
        mel = mel_outputs

    wave = vocoder.decode_batch(mel)
    wave = wave.squeeze(0)

    torchaudio.save(output, wave.cpu(), 22050)
    return output


# ===============================================================
# MAIN PIPELINE
# ===============================================================
def run_all(audio_path):
    print("\n====================================")
    print("Processing:", audio_path)
    print("====================================")

    # ----- 1. ASR -----
    print("\n⚡ Running ASR...")
    asr_text = asr_model.transcribe_file(audio_path)
    print("ASR Text:", asr_text)

    # ----- 2. Emotion -----
    print("\n⚡ Detecting Emotion...")
    emotion = classify_emotion(audio_path)
    print("Emotion:", emotion)

    # ----- 3. Intent -----
    print("\n⚡ Detecting Intent...")
    intent = classify_intent(audio_path)
    print("Intent:", intent)

    # ----- 4. Category -----
    print("\n⚡ Detecting Speech Category...")
    category = classify_category(audio_path)
    print("Speech Category:", category)

    # ----- 5. LLM Continuation -----
    print("\n⚡ Generating continuation text via GPT-2...")
    extended_text = generate_continuation(asr_text, max_new_tokens=80)
    print("Extended Text:", extended_text)

    # ----- 6. TTS -----
    print("\n⚡ Generating speech from extended text...")
    tts_file = generate_speech(extended_text)
    print("Generated Speech File:", tts_file)

    final = {
        "asr_text": asr_text,
        "extended_text": extended_text,
        "emotion": emotion,
        "intent": intent,
        "category": category,
        "tts_file": tts_file
    }

    print("\n=== FINAL RESULT ===")
    print(final)
    return final


# ===============================================================
# RUN PIPELINE
# ===============================================================
if __name__ == "__main__":
    run_all("./audios/test6.wav")
