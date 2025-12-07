import argparse
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

import docx2txt

CHROMA_PATH = "chroma"
DATA_PATH = "data"

ALLOWED_DOC_EXT = {".pdf", ".doc", ".docx", ".txt", ".md"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Resetting vector DB…")
        clear_database()

    print("📚 Loading documents…")
    documents = load_documents()

    print(f"📄 Found {len(documents)} documents. Splitting into chunks…")
    chunks = split_documents(documents)

    print("🧠 Updating Chroma DB…")
    add_to_chroma(chunks)


def load_documents():
    """
    Load ALL allowed document types:
    PDF, DOC, DOCX, TXT, MD
    """
    docs = []

    for root, _, files in os.walk(DATA_PATH):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ALLOWED_DOC_EXT:
                continue

            path = os.path.join(root, fname)

            if ext == ".pdf":
                loader = PyPDFLoader(path)
                docs.extend(loader.load())

            elif ext in {".doc", ".docx"}:
                text = docx2txt.process(path)
                docs.append(Document(page_content=text, metadata={"source": path, "page": 0}))

            elif ext in {".txt", ".md"}:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                docs.append(Document(page_content=text, metadata={"source": path, "page": 0}))

    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len
    )
    return splitter.split_documents(documents)


def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing = db.get(include=[])
    existing_ids = set(existing["ids"])

    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"🟢 Adding {len(new_chunks)} new chunks to database…")
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
        db.persist()
    else:
        print("✓ No new chunks to add (already processed).")


def calculate_chunk_ids(chunks):
    """
    IDs look like:
    data/filename.ext:page:chunk#
    """
    last_page_id = None
    chunk_idx = 0

    for chunk in chunks:
        src = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 0)
        page_id = f"{src}:{page}"

        if page_id == last_page_id:
            chunk_idx += 1
        else:
            chunk_idx = 0

        chunk.metadata["id"] = f"{page_id}:{chunk_idx}"
        last_page_id = page_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
