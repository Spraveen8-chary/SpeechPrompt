import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, session

import whisper
from gtts import gTTS
from model import run_speechbrain
from query_data import query_rag  # <-- your RAG engine
import ffmpeg

from get_embedding_function import get_embedding_function

# Load ONCE at app startup
EMBEDDINGS = get_embedding_function()

# ==================================================
# DIRECTORY STRUCTURE
# ==================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__)).replace("\\", "/")


UPLOAD_AUDIO_DIR = os.path.join(BASE_DIR, "uploads", "audio")
UPLOAD_DOC_DIR = os.path.join(BASE_DIR, "data", "docs")
OUTPUT_WHISPER_DIR = os.path.join(BASE_DIR, "outputs", "audio", "whisper")
CONVERTED_AUDIO_DIR = os.path.join(BASE_DIR, "uploads", "audio", "converted")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories
os.makedirs(UPLOAD_AUDIO_DIR, exist_ok=True)
os.makedirs(UPLOAD_DOC_DIR, exist_ok=True)
os.makedirs(OUTPUT_WHISPER_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CONVERTED_AUDIO_DIR, exist_ok=True)


# ==================================================
# LOGGING TO FILE ONLY
# ==================================================
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s [APP] %(levelname)s: %(message)s",
)
log = logging.getLogger("APP")


# ==================================================
# LOAD WHISPER
# ==================================================
log.info("Loading Whisper model: small")
whisper_model = whisper.load_model("small")

def convert_to_wav(original_path):
    """Converts any format to 16kHz WAV."""
    base = os.path.splitext(os.path.basename(original_path))[0]
    out_path = os.path.join(CONVERTED_AUDIO_DIR, base + ".wav")

    try:
        (
            ffmpeg
            .input(original_path)
            .output(out_path, ac=1, ar=16000, format="wav")
            .overwrite_output()
            .run(quiet=True)
        )
        return out_path
    except Exception as e:
        print("Conversion error:", e)
        return None

# ==================================================
# FLASK APP SETUP
# ==================================================
app = Flask(__name__)
app.secret_key = "speechprompt-key"

ALLOWED_DOC_EXT = {".pdf", ".doc", ".docx", ".txt", ".md"}
MAX_DOCS = 5


# ==================================================
# SAVE FILES
# ==================================================
def _save_audio_file(file):
    """Save uploaded audio to uploads/audio/"""
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    _, ext = os.path.splitext(file.filename)
    ext = ext or ".wav"
    filename = f"{ts}{ext}"

    save_path = os.path.join(UPLOAD_AUDIO_DIR, filename)
    file.save(save_path)

    log.info(f"[UPLOAD] Audio saved: {filename}")
    return filename


def _save_doc_file(file):
    """Save uploaded docs to data/docs/"""
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    _, ext = os.path.splitext(file.filename)
    filename = f"{ts}{ext}"

    save_path = os.path.join(UPLOAD_DOC_DIR, filename)
    file.save(save_path)

    log.info(f"[UPLOAD] Document saved: {filename}")
    return filename


# ==================================================
# WHISPER TRANSCRIPTION
# ==================================================
def transcribe_whisper(path):
    result = whisper_model.transcribe(path)
    text = result.get("text", "").strip()
    lang = result.get("language", "unknown")
    return lang, text


# ==================================================
# TTS (for Whisper + Mistral output)
# ==================================================
def whisper_tts_save(text):
    """Converts text to audio using gTTS and saves to outputs/audio/whisper/"""
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_whisper.mp3"
    out_path = os.path.join(OUTPUT_WHISPER_DIR, filename)

    try:
        gTTS(text=text, lang="en").save(out_path)
        log.info(f"[TTS] Whisper audio saved: {filename}")
        return filename
    except Exception as e:
        log.error(f"[TTS ERROR] {e}")
        return None


# ==================================================
# ROUTES
# ==================================================
@app.route("/", methods=["GET"])
def index():
    if "docs" not in session:
        session["docs"] = []

    return render_template(
        "home.html",
        sb_result=None,
        whisper_result=None,
        doc_filenames=session["docs"],
    )


@app.route("/", methods=["POST"])
def run_pipeline():
    log.info("======== NEW RUN PIPELINE ========")

    # --------------------------------------
    # COLLECT INPUTS
    # --------------------------------------
    output_type = request.form.get("output_type")
    user_prompt = (request.form.get("prompt_text") or "").strip()

    log.info(f"[MODE] {output_type}")
    log.info(f"[USER PROMPT] {user_prompt}")

    # -------------------------------
    # AUDIO HANDLING
    # -------------------------------
    audio_filename = None
    uploaded = request.files.get("audio_file")

    if uploaded and uploaded.filename:
        audio_filename = _save_audio_file(uploaded)

    if not audio_filename:
        return render_template(
            "home.html",
            sb_result={"error": "No audio uploaded."},
            whisper_result=None,
            doc_filenames=session.get("docs", []),
        )

    original_path = os.path.join(UPLOAD_AUDIO_DIR, audio_filename)
    audio_path = convert_to_wav(original_path)

    if not audio_path:
        return "Audio conversion failed", 500


    # ==========================================================
    # STAGE 1 — SPEECHBRAIN (Card 1)
    # ==========================================================
    sb_result = run_speechbrain(audio_path)

    # ==========================================================
    # DOCUMENT UPLOADS
    # ==========================================================
    new_docs = request.files.getlist("session_docs")
    if "docs" not in session:
        session["docs"] = []

    for d in new_docs:
        ext = os.path.splitext(d.filename)[1].lower()
        if ext in ALLOWED_DOC_EXT:
            saved = _save_doc_file(d)
            session["docs"].append(saved)

    session["docs"] = session["docs"][:MAX_DOCS]
    session.modified = True

    # ==========================================================
    # STAGE 2 — WHISPER + RAG + MISTRAL (Card 2)
    # ==========================================================

    whisper_lang, whisper_asr = transcribe_whisper(audio_path)
    log.info(f"[WHISPER ASR] {whisper_asr}")


    # ==========================================================
    # ASR MODE → ONLY RETURN WHISPER ASR + AUDIO
    # ==========================================================
    if output_type == "asr":
        whisper_output_text = whisper_asr.strip()

        whisper_audio_file = whisper_tts_save(whisper_output_text)

        whisper_result = {
            "whisper_asr": whisper_asr.strip(),
            "whisper_output": whisper_output_text,
            "whisper_audio_file": whisper_audio_file,
        }

        return render_template(
            "home.html",
            sb_result=sb_result,
            whisper_result=whisper_result,
            doc_filenames=session["docs"],
        )

    # ==========================================================
    # CLASSIFICATION MODE → USE RAG + CLEAN BULLET OUTPUT
    # ==========================================================
    elif output_type == "classification":

        classification_instruction = (
            "Return ONLY bullet points (•).\n"
            "No explanations. No stories. No apologies.\n\n"
            "Required format:\n"
            "• Emotion: ...\n"
            "• Intent: ...\n"
            "• Category: ..."
        )

        query_text = f"Transcription:\n{whisper_asr}"

        if user_prompt:
            query_text += f"\n\nUser Instruction:\n{user_prompt}"

        query_text += f"\n\n{classification_instruction}"

        raw_output = query_rag(
            query_text,
            task_type="classification",
            selected_docs=session.get("docs")
        )

        whisper_output_text = "\n".join(
            line.strip()
            for line in raw_output.splitlines()
            if line.strip()
        )

        whisper_audio_file = whisper_tts_save(whisper_output_text)

        whisper_result = {
            "whisper_asr": whisper_asr.strip(),
            "whisper_output": whisper_output_text,
            "whisper_audio_file": whisper_audio_file,
        }



    # ==========================================================
    # GENERATION MODE → CLEAN GENERATIVE OUTPUT
    # ==========================================================
    elif output_type == "generation":

        clean_instruction = (
            "Generate the output directly.\n"
            "No apologies. No meta text. No role labels.\n"
            "Only the final answer."
        )

        query_text = f"Transcription:\n{whisper_asr}"

        if user_prompt:
            query_text += f"\n\nUser Instruction:\n{user_prompt}"

        query_text += f"\n\n{clean_instruction}"

        raw_output = query_rag(
            query_text,
            task_type="generation",
            selected_docs=session.get("docs")
        )

        whisper_output_text = raw_output.strip()

        whisper_audio_file = whisper_tts_save(whisper_output_text)

        whisper_result = {
            "whisper_asr": whisper_asr.strip(),
            "whisper_output": whisper_output_text,
            "whisper_audio_file": whisper_audio_file,
        }


    # ==========================================================
    # RENDER PAGE WITH BOTH RESULT CARDS
    # ==========================================================
    return render_template(
        "home.html",
        sb_result=sb_result,
        whisper_result=whisper_result,
        doc_filenames=session["docs"],
    )



# ==================================================
# Serve audio files
# ==================================================
@app.route("/media/sb/<filename>")
def media_sb(filename):
    return send_from_directory(OUTPUT_WHISPER_DIR.replace("whisper", "speechbrain"), filename)


@app.route("/media/whisper/<filename>")
def media_whisper(filename):
    return send_from_directory(OUTPUT_WHISPER_DIR, filename)


@app.route("/media/uploaded/<filename>")
def media_uploaded(filename):
    return send_from_directory(UPLOAD_AUDIO_DIR, filename)

@app.route("/remove_doc", methods=["POST"])
def remove_doc():
    idx = int(request.form.get("index", -1))

    if "docs" not in session:
        session["docs"] = []

    if 0 <= idx < len(session["docs"]):
        removed = session["docs"].pop(idx)
        session.modified = True
        return jsonify({"ok": True})

    return jsonify({"ok": False}), 400


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    log.info("🚀 Server Running on http://127.0.0.1:5000")
    app.run(debug=True,use_reloader=False, host="0.0.0.0", port=5000)
