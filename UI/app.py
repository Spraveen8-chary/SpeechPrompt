# app.py (updated)
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from datetime import datetime
import os
import whisper
import ollama

# ---------------------------
# Whisper model (choose medium/large-v3 for better multilingual)
# ---------------------------
whisper_model = whisper.load_model("medium")  # change to "large-v3" if you can

app = Flask(__name__)
app.secret_key = "super-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_DOC_EXT = {".pdf", ".doc", ".docx", ".txt", ".md"}
MAX_DOCS = 5


def _save_file(file_storage):
    """Save uploaded file to /uploads and return filename."""
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    _, ext = os.path.splitext(file_storage.filename)
    if not ext:
        ext = ".bin"
    filename = f"{ts}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(save_path)
    return filename


def transcribe_audio(path, forced_language: str | None = None):
    """
    Transcribe audio and auto-detect language using Whisper.
    If forced_language (ISO code, e.g. 'te'/'hi') is provided, Whisper will try to use it.
    Returns (language_code, transcription) or ("error", error_text)
    """
    try:
        if forced_language:
            result = whisper_model.transcribe(path, language=forced_language)
        else:
            result = whisper_model.transcribe(path)
        detected_lang = result.get("language", "unknown")
        transcription = result.get("text", "")
        return detected_lang, transcription
    except Exception as e:
        return "error", f"[Error transcribing audio]: {str(e)}"


def ask_mistral(prompt_messages: list[dict], model_name: str = "mistral"):
    """
    Call Ollama (Mistral). prompt_messages is a list of dicts like:
      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    Returns the assistant content string or an error string.
    """
    try:
        resp = ollama.chat(model=model_name, messages=prompt_messages)
        # Ollama returns structure like {"message": {"role": "assistant", "content": "..."}}
        return resp.get("message", {}).get("content", "")
    except Exception as e:
        return f"[Error calling LLM]: {str(e)}"


@app.route("/", methods=["GET"])
def index():
    """Initial UI load."""
    if "docs" not in session:
        session["docs"] = []
    return render_template(
        "home.html",
        result_text=None,
        audio_filename=None,
        doc_filenames=session["docs"]
    )


@app.route("/", methods=["POST"])
def run_pipeline():
    """Main pipeline execution route (single POST handler)."""
    if "docs" not in session:
        session["docs"] = []

    # initialize
    audio_filename = None
    detected_lang = None
    transcription = None
    llm_response = None

    # -----------------------------
    # read form fields
    # -----------------------------
    file = request.files.get("audio_file")
    live_file = request.form.get("live_filename")
    user_prompt = (request.form.get("prompt_text") or "").strip()
    output_type = (request.form.get("output_type") or "asr").lower().strip()

    # -----------------------------
    # save audio if provided
    # -----------------------------
    if file and file.filename:
        audio_filename = _save_file(file)
    elif live_file:
        audio_filename = live_file

    # -----------------------------
    # session docs handling
    # -----------------------------
    new_docs = request.files.getlist("session_docs")
    for f in new_docs:
        if f and f.filename:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext in ALLOWED_DOC_EXT:
                saved_name = _save_file(f)
                session["docs"].append(saved_name)
    session["docs"] = session["docs"][:MAX_DOCS]
    session.modified = True

    # -----------------------------
    # TRANSCRIBE if audio exists
    # -----------------------------
    if audio_filename:
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        # No forced language; Whisper auto-detects. Optionally you can force based on UI.
        detected_lang, transcription = transcribe_audio(audio_path)

    # -----------------------------
    # Build the LLM instruction according to output_type
    # -----------------------------
    # Default system instruction to guide model behavior
    system_msg = (
        "You are a helpful assistant. Follow the user's request strictly and format "
        "the response according to the requested output type. "
        "If user requests 'classification', return a JSON object with 'label' and optional 'confidence' and a short 'explanation'. "
        "If user requests 'asr', return a corrected transcription and, if requested, a transliteration into the original script. "
        "If user requests 'generation', produce a helpful continuation/answer based on the user's prompt and the transcript. "
        "Do not include extraneous commentary."
    )

    # Compose the content to send: include both user's prompt_text and the transcript (if present)
    # The LLM will be able to choose which to use based on the output_type and user's prompt.
    user_parts = []
    if user_prompt:
        user_parts.append(f"User prompt/instruction:\n{user_prompt}")
    if transcription:
        user_parts.append(f"Transcript (detected language: {detected_lang}):\n{transcription}")
    else:
        user_parts.append("No audio transcript was provided.")

    # Add a short explicit instruction about the output_type so model follows it precisely
    user_parts.append(f"Output type requested: {output_type}")

    # Examples: When classification requested, prefer short JSON; when ASR requested, correct minor errors.
    user_parts.append(
        "Respond concisely. Examples:\n"
        "- classification -> JSON: {\"label\":\"<label>\", \"confidence\":0.86, \"explanation\":\"...\"}\n"
        "- asr -> Corrected transcription, keep original language script if possible.\n"
        "- generation -> Natural language answer following user's instruction.\n"
    )

    user_message = "\n\n".join(user_parts)

    # Prepare messages for Ollama
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_message}
    ]

    # -----------------------------
    # Call LLM if there's either a prompt or a transcription (or both)
    # -----------------------------
    if user_prompt or transcription:
        llm_response = ask_mistral(messages, model_name="mistral")
    else:
        llm_response = None

    # -----------------------------
    # Prepare result_text (HTML-safe) for your existing UI
    # -----------------------------
    if transcription or llm_response:
        parts = []
        if detected_lang:
            parts.append(f"<b>Detected Language:</b> {detected_lang.upper()}")
        if transcription:
            parts.append(f"<b>Transcription:</b><br>{transcription}")
        if user_prompt:
            parts.append(f"<b>User Prompt:</b><br>{user_prompt}")
        if llm_response:
            parts.append(f"<b>AI Response (Mistral):</b><br>{llm_response}")
        result_text = "<br><br>".join(parts)
    else:
        result_text = "No input provided (no audio and no prompt). Please upload audio or type a prompt."

    return render_template(
        "home.html",
        result_text=result_text,
        audio_filename=audio_filename,
        doc_filenames=session["docs"]
    )


@app.route("/remove_doc", methods=["POST"])
def remove_doc():
    """Remove a stored document by index."""
    idx = int(request.form.get("index", -1))
    if 0 <= idx < len(session.get("docs", [])):
        session["docs"].pop(idx)
        session.modified = True
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 400


@app.route("/api/upload", methods=["POST"])
def upload_blob():
    """Handles microphone audio uploads (webm blobs)."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    saved = _save_file(f)
    return jsonify({"filename": saved}), 201


@app.route("/media/<path:filename>", endpoint="media")
def serve_media(filename):
    """Serve saved audio and documents."""
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
