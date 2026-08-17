import os
import uuid
import time
import re
import tempfile
import logging
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
# from sentence_transformers import CrossEncoder
from groq import Groq

# =========================================================
# Setup Logging
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_backend")

# =========================================================
# Load environment variables
# =========================================================

base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(base_dir, ".env"))
load_dotenv(os.path.join(base_dir, "..", ".env"))
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

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
        "https://rag-qa-project.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Models
# =========================================================

# Reranker
# =========================================================
# Lazy-loaded models (avoid loading at startup to prevent OOM)
# =========================================================

# _reranker = None
# def get_reranker():
#     global _reranker
#     if _reranker is None:
#         try:
#             _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#         except Exception:
#             _reranker = False  # mark as failed, avoid retrying every call
#     return _reranker or None

_embedding_fn = None

def get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_fn

sessions = {}

# =========================================================
# In-memory session store
# =========================================================


# =========================================================
# API Models
# =========================================================

class UploadedDocumentInfo(BaseModel):
    document_id: str
    filename: str
    size: int
    pages: int
    status: str


class UploadResponse(BaseModel):
    session_id: str
    documents: List[UploadedDocumentInfo]


class AskRequest(BaseModel):
    session_id: str
    question: str
    document_id: str


# =========================================================
# Chunking
# =========================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]


# =========================================================
# Answer Type Detection & Post-Processing Helpers
# =========================================================

FALLBACK_RESPONSE = "I couldn't find that information in the uploaded PDF."


def detect_answer_type(question: str) -> Dict[str, str]:
    """
    Analyzes the question to determine the target answer type (name, date, number, 
    list, single value, summary, yes/no, or general explanation) and returns 
    formatting directives to enforce in the LLM prompt.
    """
    q_lower = question.lower().strip()

    # 1. Specific Single Value (Email, Phone, Capital, Address, Title, Website)
    if re.search(r'\b(email|emails|phone|contact|capital|address|website|url|title)\b', q_lower):
        return {
            "type": "single_value",
            "instruction": "The user is asking for a SPECIFIC SINGLE VALUE (e.g. email, phone, capital, title). Return ONLY that exact value from CONTEXT without additional narrative."
        }

    # 2. Name / Person / Author
    if re.search(r'\b(who|author|authors|author\'s|whose|person|name|names)\b', q_lower):
        return {
            "type": "name",
            "instruction": "The user is asking for a NAME. Output ONLY the exact name(s) found in CONTEXT without full narrative sentences."
        }

    # 3. Date / Time (using word boundaries to avoid false positives like 'candidate')
    if re.search(r'\b(when|date|dates|year|years|month|months|day|days|time|deadline)\b', q_lower):
        return {
            "type": "date",
            "instruction": "The user is asking for a DATE or TIME. Output ONLY the exact date or time value found in CONTEXT."
        }

    # 4. Number / Amount / Price / Quantity
    if any(k in q_lower for k in ["how many", "how much", "number of"]) or re.search(r'\b(count|price|cost|salary|percentage|rate)\b', q_lower):
        return {
            "type": "number",
            "instruction": "The user is asking for a NUMBER or QUANTITY. Output ONLY the numerical value or exact amount specified in CONTEXT."
        }

    # 5. List / Multiple items
    if any(k in q_lower for k in ["what are all", "all the", "name all", "which ones"]) or re.search(r'\b(list|enumerate)\b', q_lower):
        return {
            "type": "list",
            "instruction": "The user is asking for a LIST of items. Extract ONLY the matching items found in CONTEXT, formatted one item per line."
        }

    # 6. Yes / No questions
    if re.match(r'^(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b', q_lower):
        return {
            "type": "yes_no",
            "instruction": "The question is a Yes/No question. Start your response with 'Yes' or 'No', followed by a single factual sentence from CONTEXT if needed."
        }

    # 7. Summary
    if re.search(r'\b(summarize|summary)\b', q_lower):
        match_sent = re.search(r'(\d+|\w+)\s+sentences?', q_lower)
        num_str = match_sent.group(1) if match_sent else "two"
        return {
            "type": "summary",
            "instruction": f"Provide a concise summary ({num_str} sentences) using ONLY facts explicitly present in CONTEXT."
        }

    # 8. General / Explanation
    return {
        "type": "general",
        "instruction": "Provide a concise, direct answer based strictly on the CONTEXT. Stop as soon as the question is answered."
    }


def _normalize(s: str) -> str:
    """Collapse whitespace and lowercase, so minor formatting differences
    between the LLM's output and the raw context don't cause false negatives
    in the grounding check."""
    return re.sub(r'\s+', ' ', s.lower()).strip()


def _fuzzy_grounded(answer: str, context: str, min_ratio: float = 0.6) -> bool:
    """Looser grounding check: true if most significant words (len > 2) in
    the answer also appear somewhere in the context. Used as a fallback when
    the exact normalized substring match fails, so slight LLM rephrasing
    (e.g. added title, reordered words) doesn't trigger a false fallback."""
    norm_context = _normalize(context)
    words = [w.strip(".,;:()") for w in answer.lower().split() if len(w.strip(".,;:()")) > 2]
    if not words:
        return True
    matched = sum(1 for w in words if w in norm_context)
    return (matched / len(words)) >= min_ratio


def clean_and_post_process_answer(raw_text: str, question: str, type_info: Dict[str, str], context: str) -> str:
    """
    Post-processes the LLM output:
    - Removes duplicated text and headers
    - Strips 'Answer:' prefixes
    - Verifies that answer is grounded in context
    - Returns FALLBACK_RESPONSE if information was missing or not found.
    """
    if not raw_text or not raw_text.strip():
        return FALLBACK_RESPONSE

    text = raw_text.strip()
    text = re.sub(r'(?i)^(answer|final answer|response):\s*', '', text).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return FALLBACK_RESPONSE

    first_line = lines[0]

    # Explicit fallback indicators when LLM indicates data is missing
    fallback_indicators = [
        "i couldn't find that information",
        "could not find that information",
        "i could not find any information",
        "i couldn't find any information",
        "not found in the uploaded",
        "not mentioned in the uploaded",
        "not provided in the uploaded",
    ]
    if any(ind in text.lower() for ind in fallback_indicators):
        return FALLBACK_RESPONSE

    # Grounding check: verify that key words in answer appear in context
    norm_context = _normalize(context)
    if not _fuzzy_grounded(text, norm_context, min_ratio=0.25):
        # If words in answer have zero overlap with context, it may be a complete hallucination
        if not any(w in norm_context for w in [w.strip(".,;:()") for w in text.lower().split() if len(w) > 3]):
            return FALLBACK_RESPONSE

    return text



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

        start = max(start + 1, split_at - CHUNK_OVERLAP)

    return chunks


# =========================================================
# Chroma client
# =========================================================

chroma_client = chromadb.PersistentClient(path=VECTOR_DIR)


# =========================================================
# Upload endpoint
# =========================================================

@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(
    files: List[UploadFile] = File(None),
    session_id: str = Form(None),
):
    uploads: List[UploadFile] = files or []

    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="No PDF files were uploaded."
        )

    existing_session = sessions.get(session_id) if session_id else None
    session_id = existing_session["session_id"] if existing_session else str(uuid.uuid4())
    collection_name = existing_session["collection_name"] if existing_session else f"collection_{session_id}"
    client = chroma_client


    if existing_session:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=get_embedding_fn()
        )
        document_entries = existing_session["documents"]
    else:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_fn(),
            metadata={"session_id": session_id},
        )
        document_entries = []
    batch_docs: List[str] = []
    batch_metadatas: List[Dict] = []
    batch_ids: List[str] = []

    for upload in uploads:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )

        contents = await upload.read()

        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded PDF {upload.filename} is empty."
            )


        document_id = str(uuid.uuid4())

        # Temporary file creation and immediate cleanup to reduce Render free-tier storage usage
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            reader = PdfReader(tmp_path)
            num_pages = len(reader.pages)

            if num_pages == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded PDF {upload.filename} is empty."
                )

            if num_pages > MAX_PAGES:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {upload.filename} exceeds the maximum limit of {MAX_PAGES} pages."
                )

            docs = []
            tokenized_docs = []

            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                for chunk in chunk_text(text):
                    docs.append({
                        "page": i + 1,
                        "content": chunk,
                        "document_id": document_id,
                        "filename": upload.filename,
                    })
                    tokenized_docs.append(chunk.lower().split())

            if not docs:
                raise HTTPException(
                    status_code=400,
                    detail=f"No text could be extracted from PDF {upload.filename}."
                )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read PDF file {upload.filename}."
            )
        finally:
            # Clean up temporary PDF file immediately to reduce Render free-tier storage usage
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        document_entries.append({
            "document_id": document_id,
            "filename": upload.filename,
            "size": len(contents),
            "pages": num_pages,
            "status": "ready",
            "bm25": BM25Okapi(tokenized_docs),
            "docs": docs,
        })

        for idx, doc in enumerate(docs):
            batch_docs.append(doc["content"])
            batch_metadatas.append({
                "page": doc["page"],
                "document_id": document_id,
                "filename": upload.filename,
            })
            batch_ids.append(f"{session_id}-{document_id}-{idx}")

    collection.add(
        documents=batch_docs,
        metadatas=batch_metadatas,
        ids=batch_ids,
    )

    sessions[session_id] = {
        "session_id": session_id,
        "collection_name": collection_name,
        "documents": document_entries,
        "uploaded_at": int(time.time()),
    }

    return UploadResponse(
        session_id=session_id,
        documents=[
            UploadedDocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                size=doc["size"],
                pages=doc["pages"],
                status=doc["status"],
            )
            for doc in document_entries
        ],
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

    client = chroma_client


    collection = client.get_collection(
        name=session["collection_name"],
        embedding_function=get_embedding_fn(),
    )

    selected_doc = next(
        (doc for doc in session["documents"] if doc["document_id"] == req.document_id),
        None,
    )

    if not selected_doc:
        raise HTTPException(
            status_code=404,
            detail="Selected document not found in session.",
        )

    # Vector retrieval scoped to the selected document only
    vector_results = collection.query(
        query_texts=[req.question],
        n_results=10,
        include=["documents", "metadatas"],
        where={"document_id": req.document_id},
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

    # BM25 retrieval scoped to the selected document only
    bm25 = selected_doc["bm25"]
    docs = selected_doc["docs"]

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

    # Filter out any empty chunks for retrieval robustness
    candidate_docs = [doc for doc in candidate_docs if doc.get("content") and doc["content"].strip()]

    if not candidate_docs:
        return JSONResponse({
            "answer": FALLBACK_RESPONSE,
            "sources": [],
        })

    # Use retrieved Chroma and BM25 candidate documents directly (top 10 chunks)
    top_docs_meta = candidate_docs[:10]

    # Build context from top non-empty relevant chunks
    top_docs = [doc["content"].strip() for doc in top_docs_meta if doc["content"].strip()]
    if not top_docs:
        return JSONResponse({
            "answer": FALLBACK_RESPONSE,
            "sources": [],
        })

    context = "\n\n".join(top_docs)

    # 1. Answer Type Detection
    type_info = detect_answer_type(req.question)

    generated = None

    # 2. Strict Grounded RAG Universal Prompt
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
- If the answer cannot be found in the CONTEXT, respond with: "{FALLBACK_RESPONSE}"
 

CONTEXT:
{context}

QUESTION:
{req.question}

ANSWER:"""

    candidate_models = [GROQ_MODEL] + [m for m in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound-mini"] if m != GROQ_MODEL]

    for model_name in candidate_models:
        try:
            completion = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024,
            )
            raw = completion.choices[0].message.content.strip() if completion.choices else ""
            generated = clean_and_post_process_answer(raw, req.question, type_info, context)
            if generated:
                break
        except Exception as e:
            logger.error(f"Error generating answer with Groq model '{model_name}': {e}")

    if not generated:
        generated = FALLBACK_RESPONSE

    sources = []
    if generated != FALLBACK_RESPONSE:
        for doc in top_docs_meta:
            for p in doc.get("pages", [1]):
                sources.append({
                    "page": p,
                    "text": (doc.get("content", "")[:120] + "...") if len(doc.get("content", "")) > 120 else doc.get("content", "")
                })

    return JSONResponse({
        "answer": generated,
        "sources": sources,
    })
# =========================================================
# Clear session
# =========================================================

@app.post("/api/clear")
async def clear(session_id: str = Form(...)):
    session = sessions.pop(session_id, None)

    if session:
        try:
            client = chroma_client
            client.delete_collection(name=session["collection_name"])

            for doc in session.get("documents", []):
                filepath = doc.get("filepath")
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)

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