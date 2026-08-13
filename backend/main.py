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
from groq import Groq


# =========================================================
# Load environment variables
# =========================================================

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://rag-qa-project.vercel.app",
    ],
    allow_origin_regex=r"https://.*-peruguyaswanths-projects\.vercel\.app",
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

# =========================================================
# In-memory session store
# =========================================================


embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
    )

sessions = {}


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


def clean_and_post_process_answer(raw_text: str, question: str, type_info: Dict[str, str], context: str) -> str:
    """
    Post-processes the LLM output:
    - Removes duplicated text and lines
    - Removes trailing unrelated content or conversational notes
    - Removes unrequested section headers
    - Verifies precision and strictly checks that extracted values appear in context
    - Trims whitespace and defaults to fallback message if answer is missing or ungrounded.
    """
    if not raw_text or not raw_text.strip():
        return FALLBACK_RESPONSE

    text = raw_text.strip()
    text = re.sub(r'(?i)^(answer|final answer|response):\s*', '', text).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return FALLBACK_RESPONSE

    first_line = lines[0]

    # Check if first line indicates fallback
    fallback_indicators = [
        "i couldn't find", "could not find", "not mentioned", "not provided",
        "no information", "cannot find", "not specified", "does not contain",
        "not_found"
    ]
    if any(ind in first_line.lower() for ind in fallback_indicators):
        return FALLBACK_RESPONSE

    ans_type = type_info["type"]

    # Single value / Name / Date / Number processing
    if ans_type in ["name", "date", "number", "single_value"]:
        if ":" in first_line and not first_line.lower().startswith("http"):
            parts = first_line.split(":", 1)
            if parts[1].strip():
                first_line = parts[1].strip()
        first_line = re.sub(r'(?i)^(the\s+[^=:\n]+?\s+(is|was|are|were))\s*', '', first_line).strip()

        # Strict grounding verification: single values must be grounded in retrieved context
        if first_line.lower() not in context.lower():
            return FALLBACK_RESPONSE

        return first_line

    # List items processing
    if ans_type == "list":
        formatted_items = []
        for line in lines:
            if any(ind in line.lower() for ind in ["explanation:", "note:", "context:"]):
                break
            item = re.sub(r'^(?:\d+[\.\)]|[-*•])\s*', '', line).strip()
            if item and item.lower() not in [i.lower() for i in formatted_items]:
                # Grounding check for list item
                if item.lower() in context.lower() or any(w.lower() in context.lower() for w in item.split() if len(w) > 3):
                    formatted_items.append(item)
        if formatted_items:
            return "\n".join(formatted_items)
        return FALLBACK_RESPONSE

    # Yes / No processing
    if ans_type == "yes_no":
        if not re.match(r'(?i)^(yes|no)\b', first_line):
            if "yes" in first_line.lower() and "no" not in first_line.lower():
                first_line = "Yes. " + first_line
            elif "no" in first_line.lower() and "yes" not in first_line.lower():
                first_line = "No. " + first_line
        return first_line

    # General / Summary processing
    valid_lines = []
    for line in lines:
        if any(ind in line.lower() for ind in ["explanation:", "note:", "context:", "question:"]):
            break
        valid_lines.append(line)

    res = "\n".join(valid_lines) if valid_lines else FALLBACK_RESPONSE

    # Grounding check for general/summary answers
    words = [w for w in res.split() if len(w) > 3 and w.isalnum()]
    if words and not any(w.lower() in context.lower() for w in words):
        return FALLBACK_RESPONSE

    return res



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
            embedding_function=embedding_fn,
        )
        document_entries = existing_session["documents"]
    else:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
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

        size_mb = len(contents) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File {upload.filename} exceeds maximum upload size of {MAX_UPLOAD_MB} MB."
            )

        document_id = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

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
                detail=f"Failed to read PDF file {upload.filename}."
            )

        if num_pages == 0:
            os.remove(save_path)
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded PDF {upload.filename} is empty."
            )

        if num_pages > MAX_PAGES:
            os.remove(save_path)
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
            os.remove(save_path)
            raise HTTPException(
                status_code=400,
                detail=f"No text could be extracted from PDF {upload.filename}."
            )

        document_entries.append({
            "document_id": document_id,
            "filename": upload.filename,
            "size": len(contents),
            "pages": num_pages,
            "status": "ready",
            "filepath": save_path,
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
        embedding_function=embedding_fn,
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

    # Build context from top non-empty relevant chunks
    top_docs = [doc["content"].strip() for doc in top_docs_meta if doc["content"].strip()]
    if not top_docs:
        return JSONResponse({
            "answer": FALLBACK_RESPONSE,
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
- If the CONTEXT does not contain the answer, respond EXACTLY with:
  I couldn't find that information in the uploaded PDF.

CONTEXT:
{context}

QUESTION:
{req.question}

ANSWER:"""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        raw = completion.choices[0].message.content.strip() if completion.choices else ""
        generated = clean_and_post_process_answer(raw, req.question, type_info, context)
    except Exception:
        generated = None

    if not generated:
        generated = FALLBACK_RESPONSE

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