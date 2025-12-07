import os
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from flask import Flask, render_template, request, jsonify, send_from_directory, session

# try faster whisper
USE_FASTER_WHISPER = False
try:
    from faster_whisper import WhisperModel as FWWhisperModel
    USE_FASTER_WHISPER = True
except Exception:
    USE_FASTER_WHISPER = False

# fallback whisper
try:
    import whisper as openai_whisper
    HAVE_OPENAI_WHISPER = True
except Exception:
    HAVE_OPENAI_WHISPER = False

# TTS (edge-tts preferred)
USE_EDGE_TTS = False
try:
    import edge_tts
    USE_EDGE_TTS = True
except Exception:
    USE_EDGE_TTS = False

# fallback gTTS
try:
    from gtts import gTTS
    HAVE_GTTS = True
except Exception:
    HAVE_GTTS = False

# LLM
import ollama

# RAG
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# populate script for auto-update
from populate_database import load_documents, split_documents, add_to_chroma

# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

ALLOWED_DOC_EXT = {".pdf", ".doc", ".docx", ".txt", ".md"}
MAX_DOCS = 5

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

app = Flask(__name__)
app.secret_key = "SpeechPrompt-Dev"

EXECUTOR = ThreadPoolExecutor(max_workers=4)

WHISPER_ENGINE = None
EMBEDDING_FN = None
CHROMA_DB = None


# ---------------------------------------------------------
# Save audio into uploads/
# ---------------------------------------------------------
def _save_audio(file_storage):
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    _, ext = os.path.splitext(file_storage.filename)
    ext = ext or ".bin"
    filename = f"{ts}{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(path)
    return filename


# ---------------------------------------------------------
# Save documents into data/
# ---------------------------------------------------------
def save_document_to_data(file_storage):
    """
    Save uploaded PDF/DOC/DOCX/TXT/MD into data/ folder
    and return filename.
    """
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    _, ext = os.path.splitext(file_storage.filename)
    ext = ext.lower()

    filename = f"{ts}{ext}"
    dst_path = os.path.join(DATA_FOLDER, filename)
    file_storage.save(dst_path)
    return filename


# ---------------------------------------------------------
# load whisper
# ---------------------------------------------------------
def load_whisper_model():
    global WHISPER_ENGINE

    if USE_FASTER_WHISPER:
        try:
            model = FWWhisperModel("small", device="cpu", compute_type="int8")
            WHISPER_ENGINE = ("faster", model)
            app.logger.info("Loaded faster-whisper")
            return
        except Exception as e:
            app.logger.warning("faster-whisper failed: %s", e)

    if HAVE_OPENAI_WHISPER:
        model = openai_whisper.load_model("small")
        WHISPER_ENGINE = ("whisper", model)
        app.logger.info("Loaded whisper-small")
        return

    raise RuntimeError("No whisper backend available")


# ---------------------------------------------------------
# transcribe
# ---------------------------------------------------------
def transcribe_blocking(path):
    engine_type, model = WHISPER_ENGINE
    if engine_type == "faster":
        segments, info = model.transcribe(path, beam_size=5)
        text = "".join([s.text for s in segments])
        return info.language or "unknown", text
    else:
        res = model.transcribe(path)
        return res.get("language", "unknown"), res.get("text", "")


# ---------------------------------------------------------
# init RAG (load embeddings + db)
# ---------------------------------------------------------
def init_rag():
    global EMBEDDING_FN, CHROMA_DB
    EMBEDDING_FN = get_embedding_function()
    CHROMA_DB = Chroma(persist_directory=CHROMA_PATH,
                       embedding_function=EMBEDDING_FN)
    app.logger.info("Chroma DB loaded.")


# ---------------------------------------------------------
# Retrieve context from DB
# ---------------------------------------------------------
def retrieve_context_blocking(txt, k=5):
    results = CHROMA_DB.similarity_search_with_score(txt, k=k)
    context = "\n\n---\n\n".join([d.page_content for d, _ in results])
    sources = [d.metadata.get("id") for d, _ in results]
    return context, sources


# ---------------------------------------------------------
# Auto-update RAG DB when new docs uploaded
# ---------------------------------------------------------
def update_rag_database():
    """
    Loads new docs from data/ and updates Chroma incrementally.
    """
    docs = load_documents()
    chunks = split_documents(docs)
    add_to_chroma(chunks)


# ---------------------------------------------------------
# LLM call
# ---------------------------------------------------------
def call_mistral_blocking(prompt):
    try:
        resp = ollama.chat(
            model="mistral:latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp["message"]["content"]
    except Exception as e:
        return f"[LLM Error: {e}]"


# ---------------------------------------------------------
# TTS generation
# ---------------------------------------------------------
def tts_save_blocking(text, lang="en"):
    fname = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_tts.mp3"
    out_path = os.path.join(UPLOAD_FOLDER, fname)

    if USE_EDGE_TTS:
        async def _save():
            speak = edge_tts.Communicate(text, "en-US-AriaNeural")
            await speak.save(out_path)
        asyncio.run(_save())
        return fname

    if HAVE_GTTS:
        gTTS(text=text, lang=lang).save(out_path)
        return fname

    raise RuntimeError("No TTS backend available")


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------
def startup_init():
    load_whisper_model()
    init_rag()
    # warm mistral
    EXECUTOR.submit(lambda: call_mistral_blocking("Warm up."))


with app.app_context():
    startup_init()


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    if "docs" not in session:
        session["docs"] = []
    return render_template("home.html", result_text=None,
                           audio_filename=None,
                           doc_filenames=session["docs"])


@app.route("/", methods=["POST"])
def run_pipeline():
    if "docs" not in session:
        session["docs"] = []

    output_type = request.form.get("output_type")
    user_prompt = (request.form.get("prompt_text") or "").strip()

    # ------------------------------------------
    # Audio processing
    # ------------------------------------------
    audio_filename = None
    file_audio = request.files.get("audio_file")
    live_file = request.form.get("live_filename")

    if file_audio and file_audio.filename:
        audio_filename = _save_audio(file_audio)
    elif live_file:
        audio_filename = live_file

    # ------------------------------------------
    # Document upload → save to data/ → update RAG
    # ------------------------------------------
    new_docs = request.files.getlist("session_docs") or []
    new_docs_saved = []

    for d in new_docs:
        ext = os.path.splitext(d.filename)[1].lower()
        if ext in ALLOWED_DOC_EXT:
            fname = save_document_to_data(d)
            new_docs_saved.append(fname)

    if new_docs_saved:
        session["docs"].extend(new_docs_saved)
        session["docs"] = session["docs"][:MAX_DOCS]
        session.modified = True

        # 🔥 auto-update RAG with new documents
        EXECUTOR.submit(update_rag_database)

    # ------------------------------------------
    # TRANSCRIBE
    # ------------------------------------------
    lang, transcription = None, None
    if audio_filename:
        fut = EXECUTOR.submit(transcribe_blocking,
                              os.path.join(UPLOAD_FOLDER, audio_filename))
        lang, transcription = fut.result(timeout=60)

    # ASR mode
    if output_type == "asr":
        return render_template(
            "home.html",
            result_text=f"<b>Language:</b> {lang}<br><br><b>Transcription:</b><br>{transcription}",
            audio_filename=audio_filename,
            doc_filenames=session["docs"],
        )

    # ------------------------------------------
    # CLASSIFICATION / GENERATION WITH RAG
    # ------------------------------------------
    query = transcription or user_prompt or ""
    context, sources = "", []

    if query:
        fut = EXECUTOR.submit(retrieve_context_blocking, query, 5)
        context, sources = fut.result(timeout=15)

    # build full prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    final_prompt = prompt_template.format(
        context=context or "No context available",
        question=user_prompt or transcription or "",
    )

    # call llm
    fut = EXECUTOR.submit(call_mistral_blocking, final_prompt)
    response_text = fut.result(timeout=60)

    # ------------------------------------------
    # CLASSIFICATION OUTPUT
    # ------------------------------------------
    if output_type == "classification":
        tts_file = EXECUTOR.submit(tts_save_blocking, response_text).result()

        return render_template(
            "home.html",
            result_text=(
                f"<b>Language:</b> {lang}<br><br>"
                f"<b>Transcription:</b><br>{transcription}<br><br>"
                f"<b>Classification:</b><br>{response_text}<br><br>"
                f"<b>Sources:</b> {sources}"
            ),
            audio_filename=tts_file,
            doc_filenames=session["docs"]
        )

    # ------------------------------------------
    # GENERATION OUTPUT
    # ------------------------------------------
    if output_type == "generation":
        tts_file = EXECUTOR.submit(tts_save_blocking, response_text).result()

        return render_template(
            "home.html",
            result_text=(
                f"<b>Language:</b> {lang}<br><br>"
                f"<b>Transcription:</b><br>{transcription}<br><br>"
                f"<b>Generated:</b><br>{response_text}<br><br>"
                f"<b>Sources:</b> {sources}"
            ),
            audio_filename=tts_file,
            doc_filenames=session["docs"]
        )

    return "Invalid output type", 400


# ---------------------------------------------------------
# API: remove doc
# ---------------------------------------------------------
@app.route("/remove_doc", methods=["POST"])
def remove_doc():
    idx = int(request.form.get("index", -1))
    if 0 <= idx < len(session["docs"]):
        fname = session["docs"].pop(idx)
        session.modified = True

        # delete from data folder
        doc_path = os.path.join(DATA_FOLDER, fname)
        if os.path.exists(doc_path):
            os.remove(doc_path)

        EXECUTOR.submit(update_rag_database)
        return jsonify({"ok": True})

    return jsonify({"ok": False}), 400


# ---------------------------------------------------------
# microphone upload
# ---------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload_blob():
    f = request.files.get("file")
    saved = _save_audio(f)
    return jsonify({"filename": saved})


# ---------------------------------------------------------
# serve media
# ---------------------------------------------------------
@app.route("/media/<path:filename>", endpoint="media")
def serve_media(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
