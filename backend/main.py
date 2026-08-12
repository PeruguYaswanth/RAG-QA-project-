import os
import uuid
import time
import re
from collections import defaultdict
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from PyPDF2 import PdfReader

import chromadb
from chromadb.utils import embedding_functions

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import pipeline


# =========================================================
# Load environment variables
# =========================================================

load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
VECTOR_DIR = os.getenv("VECTOR_DIR", "./data/vectorstores")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_PAGES = 100

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


# =========================================================
# FastAPI app
# =========================================================

app = FastAPI(title="RAG PDF QA Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Models
# =========================================================

# Reranker
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    reranker = None

# Local LLM
try:
    llm = pipeline(
        task="text-generation",
        model="Qwen/Qwen2-0.5B-Instruct",
        max_new_tokens=500,
        do_sample=False,
        temperature=0,
        return_full_text=False,
)
except Exception:
    llm = None

# =========================================================
# In-memory session store
# =========================================================

sessions = {}


# =========================================================
# API Models
# =========================================================

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    size: int
    pages: int
    status: str


class AskRequest(BaseModel):
    session_id: str
    question: str


# =========================================================
# Chunking
# =========================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]


def normalize_answer(text: str) -> str:
    if not text:
        return text

    text = text.strip()
    text = re.sub(r'(?i)^answer:\s*', '', text).strip()

    split_point = re.search(r'(?mi)^(Explanation|Note|Analysis|Reason)\b', text)
    if split_point:
        text = text[: split_point.start()].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'^(?:\d+[\.\)]|[-*•]|\(\d+\))\s*', '', line)
        line = re.sub(r'^(?:number\s*\d+:?)\s*', '', line, flags=re.IGNORECASE)
        if line:
            cleaned_lines.append(line)

    if len(cleaned_lines) > 1:
        return '; '.join(cleaned_lines[:5])
    elif len(cleaned_lines) == 1:
        text = cleaned_lines[0]

    if re.fullmatch(r'[\d\.\s-]+', text):
        return ''

    text = re.sub(r'(?i)^(the answer is|answer is|it is|it was|it are)\s*', '', text).strip()

    if ':' in text and not text.lower().startswith('i couldn\'t find'):
        tail = text.split(':', 1)[1].strip()
        parts = re.split(r',\s*|;\s*|\sand\s|\sor\s', tail)
        parts = [p.strip() for p in parts if p and not re.fullmatch(r'\d+[\.\)]?', p)]
        if len(parts) > 1:
            return '; '.join(parts[:5])

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences:
        first_sentence = sentences[0].strip()
        if first_sentence:
            return first_sentence

    return text


def extract_list_items(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    items = []
    for line in lines:
        m = re.match(r'^(?:\d+[\.\)]|[-*•])\s*(.+)$', line)
        if m:
            candidate = m.group(1).strip()
            if candidate and not re.fullmatch(r'\d+[\.\)]?', candidate):
                items.append(candidate)

    if items:
        return '; '.join(items[:10])

    colon_match = re.search(r'^[^:\n]+:\s*(.+)$', text, flags=re.MULTILINE)
    if colon_match:
        tail = colon_match.group(1).strip()
        parts = re.split(r',\s*|;\s*|\sand\s|\sor\s', tail)
        parts = [p.strip() for p in parts if p and not re.fullmatch(r'\d+[\.\)]?', p)]
        if len(parts) > 1:
            return '; '.join(parts[:10])
    return ''


def extract_answer_from_context(text: str, question: str) -> str:
    parsed = extract_list_items(text)
    if parsed:
        return parsed

    if ':' in text:
        colon_match = re.search(r'^[^:\n]+:\s*(.+)$', text, flags=re.MULTILINE)
        if colon_match:
            answer = normalize_answer(colon_match.group(1).strip())
            if answer:
                return answer

    if any(keyword in question.lower() for keyword in ['college', 'colleges', 'branch', 'department', 'institute', 'university']):
        candidate = normalize_answer(text)
        if candidate and len(candidate.split()) <= 20:
            return candidate

    return ''


def chunk_text(text: str) -> List[str]:
    text = text.strip()

    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)

        if end == text_length:
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            break

        split_at = -1
        split_len = 0

        for sep in CHUNK_SEPARATORS:
            idx = text.rfind(sep, start, end)

            if idx > split_at:
                split_at = idx
                split_len = len(sep)

        if split_at <= start:
            split_at = end
        else:
            split_at += split_len

        chunk = text[start:split_at].strip()

        if chunk:
            chunks.append(chunk)

        start = max(split_at - CHUNK_OVERLAP, split_at)

    return chunks


# =========================================================
# Chroma client
# =========================================================

def get_chroma_client():
    return chromadb.PersistentClient(path=VECTOR_DIR)


# =========================================================
# Upload endpoint
# =========================================================

@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
        )

    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum upload size of {MAX_UPLOAD_MB} MB."
        )

    session_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")

    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        reader = PdfReader(save_path)
        num_pages = len(reader.pages)
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)

        raise HTTPException(
            status_code=400,
            detail="Failed to read PDF file."
        )

    if num_pages == 0:
        os.remove(save_path)

        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
        )

    if num_pages > MAX_PAGES:
        os.remove(save_path)

        raise HTTPException(
            status_code=400,
            detail="File exceeds the maximum limit of 100 pages."
        )

    docs = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        for chunk in chunk_text(text):
            docs.append({
                "page": i + 1,
                "content": chunk,
            })

    if not docs:
        os.remove(save_path)

        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from the PDF."
        )

    collection_name = f"collection_{session_id}"

    client = get_chroma_client()

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"session_id": session_id},
    )

    collection.add(
        documents=[d["content"] for d in docs],
        metadatas=[{"page": d["page"]} for d in docs],
        ids=[f"{session_id}-{i}" for i in range(len(docs))],
    )

    tokenized_docs = [d["content"].lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    sessions[session_id] = {
        "filename": file.filename,
        "size": len(contents),
        "pages": num_pages,
        "uploaded_at": int(time.time()),
        "collection_name": collection_name,
        "bm25": bm25,
        "docs": docs,
    }

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        size=len(contents),
        pages=num_pages,
        status="ready",
    )


# =========================================================
# Ask endpoint
# =========================================================

@app.post("/api/ask")
async def ask(req: AskRequest):
    session = sessions.get(req.session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired."
        )

    client = get_chroma_client()

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_collection(
        name=session["collection_name"],
        embedding_function=embedding_fn,
    )

    # Vector retrieval
    vector_results = collection.query(
        query_texts=[req.question],
        n_results=10,
        include=["documents", "metadatas"],
    )

    vector_docs = (
        vector_results["documents"][0]
        if vector_results["documents"]
        else []
    )

    vector_metas = (
        vector_results["metadatas"][0]
        if vector_results["metadatas"]
        else []
    )

    # BM25 retrieval
    bm25 = session["bm25"]
    docs = session["docs"]

    query_tokens = req.question.lower().split()
    scores = bm25.get_scores(query_tokens)

    top_bm25_idx = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:10]

    bm25_docs = [docs[i] for i in top_bm25_idx]

    # Merge candidates and preserve page data
    candidate_docs = []
    seen_contents = set()

    for doc, meta in zip(vector_docs, vector_metas):
        if not doc or doc in seen_contents:
            continue
        seen_contents.add(doc)
        pages = []
        if isinstance(meta, dict):
            page = meta.get("page")
            if page is not None:
                pages.append(page)
        candidate_docs.append({"content": doc, "pages": pages})

    for doc_item in bm25_docs:
        doc = doc_item["content"]
        if not doc or doc in seen_contents:
            continue
        seen_contents.add(doc)
        candidate_docs.append({"content": doc, "pages": [doc_item["page"]]})

    if not candidate_docs:
        return JSONResponse({
            "answer": "I couldn't find that information in the uploaded PDF.",
          
        })

    if reranker is not None:
        pairs = [(req.question, doc["content"]) for doc in candidate_docs]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(
            zip(rerank_scores, candidate_docs),
            key=lambda x: x[0],
            reverse=True,
        )
        top_docs_meta = [doc for _, doc in ranked[:5]]
    else:
        top_docs_meta = candidate_docs[:5]

    top_docs = [doc["content"] for doc in top_docs_meta]
    context = "\n\n".join(top_docs)

    generated = None

    if llm is not None:
        prompt = f"""You are a document question-answering assistant.

Answer the question using ONLY the information in the CONTEXT below.

STRICT RULES:
- Base your answer exclusively on the CONTEXT. Do not use outside knowledge, 
  do not infer, and do not guess.
- Reproduce facts, names, numbers, and details EXACTLY as written in the 
  CONTEXT — do not paraphrase or summarize when precision matters.
- Identify precisely WHAT TYPE OF THING the question is asking for (e.g. a 
  name, a title, a date, a quantity, a category). Return ONLY items that 
  genuinely match that type — do not substitute a related but different 
  type of item (e.g. do not return a category label when a specific name 
  was requested, or a description when a title was requested).
- If the question asks about MULTIPLE items, include EVERY matching item 
  found in the CONTEXT, not just one or two.
- If an item has MULTIPLE FIELDS or attached details (e.g. a name, a date, 
  a number, a location), keep those fields TOGETHER on one line for that 
  item — do not scatter or flatten them into a single unlabeled list.
- Format each distinct item on its own line, in a clear and readable way.
- Answer ONLY the specific topic asked about in the QUESTION, and STOP once 
  that answer is complete. Do NOT continue into other topics or sections, 
  even if they appear later in the CONTEXT.
- If you are unsure whether something belongs in the answer, LEAVE IT OUT 
  rather than guessing.
- Do not add explanations, commentary, or anything beyond the direct answer.
- If the CONTEXT does not contain the answer, respond EXACTLY with:
  I couldn't find that information in the uploaded PDF.
Context:
{context}

Question:
{req.question}

Answer:"""
    try:
        result = llm(prompt)
        raw = result[0].get("generated_text", "").strip()
        raw = re.sub(r'(?i)^answer:\s*', '', raw).strip()
        generated = raw
    except Exception:
        generated = None

    if not generated:
        generated = "I couldn't find that information in the uploaded PDF."

    return JSONResponse({
    "answer": generated,
})
# =========================================================
# Clear session
# =========================================================

@app.post("/api/clear")
async def clear(session_id: str = Form(...)):
    session = sessions.pop(session_id, None)

    if session:
        try:
            client = get_chroma_client()

            client.delete_collection(name=session["collection_name"])

            uploaded = os.path.join(
                UPLOAD_DIR,
                f"{session_id}.pdf"
            )

            if os.path.exists(uploaded):
                os.remove(uploaded)

        except Exception:
            pass

    return {"status": "cleared"}


# =========================================================
# Status
# =========================================================

@app.get("/api/status")
async def status(session_id: str):
    s = sessions.get(session_id)

    if not s:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    return s