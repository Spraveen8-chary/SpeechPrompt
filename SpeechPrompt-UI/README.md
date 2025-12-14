# 🎙️ SpeechPrompt

**SpeechPrompt** is a unified, prompt-driven speech processing system that supports **ASR, Speech Classification, and Speech Generation** using a single pipeline.
It integrates **SpeechBrain**, **Whisper**, **Mistral (LLM)**, and **optional RAG (Retrieval-Augmented Generation)** in a clean, modular architecture.

---

## 🚀 Features

* 🎧 Audio-to-text using **SpeechBrain** and **Whisper**
* 🧠 Prompt-based task control (no task-specific models)
* 📊 Speech Classification (emotion, intent, category)
* ✍️ Speech-based text generation
* 📚 Optional **RAG** using uploaded documents
* 🔊 Text-to-Speech output
* 🌐 Interactive Flask UI
* 🧩 Modular & extensible design

---

## 🧠 Core Idea

Instead of training separate models for each speech task, **SpeechPrompt** uses:

* **ASR** → convert speech to text
* **Prompts** → control the task (ASR / classification / generation)
* **LLM (Mistral)** → reasoning & generation
* **Optional RAG** → domain-specific context from user documents

---

## 🏗️ System Architecture

```
Audio Input
   ↓
SpeechBrain ASR
   ↓
Whisper ASR (parallel validation)
   ↓
Prompt + Task Selection
   ↓
(Optional) RAG over selected documents
   ↓
Mistral LLM
   ↓
Text Output
   ↓
TTS (Audio Output)
```

---

## 📁 Project Structure

```
PromptSpeech/
│
├── Frontend/
│   ├── app.py                  # Flask app (main entry)
│   ├── model.py                # SpeechBrain pipeline
│   ├── query_data.py           # RAG logic (doc-scoped)
│   ├── get_embedding_function.py
│   │
│   ├── templates/
│   │   ├── base.html
│   │   └── home.html
│   │
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   │
│   ├── uploads/
│   │   └── audio/
│   │
│   ├── outputs/
│   │   └── audio/
│   │
│   ├── data/
│   │   └── docs/               # Uploaded documents for RAG
│   │
│   └── logs/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Component  | Technology            |
| ---------- | --------------------- |
| ASR        | SpeechBrain, Whisper  |
| LLM        | Mistral (Ollama)      |
| RAG        | LangChain + Chroma    |
| Embeddings | Sentence-Transformers |
| Backend    | Flask                 |
| UI         | HTML, CSS, JS         |
| TTS        | gTTS / SpeechBrain    |
| Audio      | FFmpeg                |

---

## 🛠️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/SpeechPrompt.git
cd SpeechPrompt
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Install FFmpeg

```bash
# Windows
winget install ffmpeg
```

---

## 🧪 Execution Steps

### Start Ollama (Mistral)

```bash
ollama run mistral
```

### Run Flask App

```bash
python Frontend/app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 🎛️ Usage Modes

### 1️⃣ ASR Mode

* Upload audio
* Returns:

  * SpeechBrain ASR
  * Whisper ASR
  * Audio output

---

### 2️⃣ Classification Mode

* Upload audio
* Optional prompt
* Optional documents
* Output:

  * Emotion
  * Intent
  * Category
* Format:

```
• Emotion: ...
• Intent: ...
• Category: ...
```

---

### 3️⃣ Generation Mode

* Upload audio
* Optional prompt
* Optional documents
* Produces clean generated output
* No meta text / no hallucinated apologies

---

## 📚 RAG Behavior (IMPORTANT)

| Scenario              | Behavior                     |
| --------------------- | ---------------------------- |
| No docs selected      | Pure LLM (NO RAG)            |
| Docs selected         | RAG enabled                  |
| Docs removed via UI ❌ | RAG disabled                 |
| Large PDFs            | Text is truncated for safety |

RAG is **document-scoped per request** (no stale context).

---

## 🧠 Prompt Handling

* Prompt is **optional**
* If prompt is empty → task still runs
* Prompt influences:

  * Classification style
  * Generation behavior
* Task type always takes priority over prompt

---

## 🔐 Logging

* Logs are stored in:

```
Frontend/logs/app.log
```

* No console spam
* Useful for debugging ASR, RAG, and model issues

---

## ❗ Common Issues & Fixes

### Chroma embedding error

```
TypeError: got multiple values for embedding_function
```

✅ Fixed by:

* Avoiding `Chroma.from_documents()`
* Using manual `add_documents()`

---

### Repeated crashes

✅ Run Flask with:

```python
debug=False
use_reloader=False
```

---

## 📌 Future Improvements

* Streaming responses
* Per-document citations
* Confidence scoring
* GPU acceleration
* Multi-language support

---

## 🤝 Contributing

Pull requests are welcome.
For major changes, please open an issue first.

---

