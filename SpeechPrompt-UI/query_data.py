import os
import docx2txt
from typing import List, Optional

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function


# ==================================================
# CONFIG
# ==================================================
DOC_DIR = "data/docs"
MAX_DOC_CHARS = 20_000  # safety limit for CPU


# ==================================================
# LOAD ONLY SELECTED DOCUMENTS
# ==================================================
def load_selected_docs(selected_docs: List[str]) -> List[Document]:
    documents: List[Document] = []

    for fname in selected_docs:
        path = os.path.join(DOC_DIR, fname)
        if not os.path.exists(path):
            continue

        ext = os.path.splitext(fname)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(path)
            pages = loader.load()
            for p in pages:
                p.page_content = p.page_content[:MAX_DOC_CHARS]
                documents.append(p)

        elif ext in {".doc", ".docx"}:
            text = docx2txt.process(path)[:MAX_DOC_CHARS]
            documents.append(
                Document(page_content=text, metadata={"source": fname})
            )

        elif ext in {".txt", ".md"}:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()[:MAX_DOC_CHARS]
            documents.append(
                Document(page_content=text, metadata={"source": fname})
            )

    return documents


# ==================================================
# MAIN RAG FUNCTION (BULLETPROOF)
# ==================================================
def query_rag(
    query_text: str,
    task_type: str = "default",
    selected_docs: Optional[List[str]] = None,
) -> str:

    # --------------------------------------------------
    # NO DOCS → PURE LLM
    # --------------------------------------------------
    if not selected_docs:
        model = Ollama(model="mistral")
        return model.invoke(query_text).strip()

    docs = load_selected_docs(selected_docs)

    if not docs:
        model = Ollama(model="mistral")
        return model.invoke(query_text).strip()

    # --------------------------------------------------
    # BUILD CHROMA SAFELY (NO from_documents)
    # --------------------------------------------------
    embeddings = get_embedding_function()

    db = Chroma(
        embedding_function=embeddings
    )

    db.add_documents(docs)

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------
    results = db.similarity_search(query_text, k=5)

    if not results:
        model = Ollama(model="mistral")
        return model.invoke(query_text).strip()

    context_text = "\n\n---\n\n".join(d.page_content for d in results)

    # --------------------------------------------------
    # PROMPT SELECTION
    # --------------------------------------------------
    if task_type == "classification":
        prompt = f"""
Return ONLY bullet points (•).
No explanations.

Context:
{context_text}

---
Input:
{query_text}

Format:
• Emotion: ...
• Intent: ...
• Category: ...
"""

    elif task_type == "generation":
        prompt = f"""
Generate output strictly from the input.

Rules:
- No apologies
- No meta text

Context:
{context_text}

---
Input:
{query_text}
"""

    else:
        prompt = f"""
Answer using ONLY the context.

Context:
{context_text}

---
Question:
{query_text}
"""

    model = Ollama(model="mistral")
    raw_output = model.invoke(prompt)

    return "\n".join(
        line.strip()
        for line in raw_output.splitlines()
        if line.strip()
    )
