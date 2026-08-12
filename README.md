# RAG PDF QA — FastAPI + React + ChromaDB

A Retrieval-Augmented Generation (RAG) web application that allows users to upload PDF documents and ask questions grounded in the uploaded content. The system extracts text from PDFs, stores embeddings in a vector database, retrieves the most relevant chunks, reranks them, and generates answers using a local language model.

---

## Features

* Upload PDF documents (up to **100 pages**)
* Automatic text extraction and chunking
* Semantic search using **SentenceTransformers embeddings**
* Vector storage with **ChromaDB**
* Hybrid retrieval with **BM25 + vector search**
* Reranking with **CrossEncoder**
* Local LLM-based question answering
* React + TypeScript + Tailwind CSS frontend
* FastAPI backend with REST APIs
* Session-based document handling

---

## Tech Stack

### Backend

* FastAPI
* ChromaDB
* SentenceTransformers
* BM25 (rank-bm25)
* CrossEncoder reranker
* PyPDF2
* Transformers

### Frontend

* React
* TypeScript
* Vite
* Tailwind CSS

---

## Project Structure

```text
RAG_PROJECT/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── data/
│   │   ├── uploads/
│   │   └── vectorstores/
│   └── .env.example
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── .gitignore
└── README.md
```

---

## How It Works

1. User uploads a PDF document.
2. Backend validates file size and page count.
3. Text is extracted from each page.
4. Text is split into overlapping chunks.
5. Chunks are converted into embeddings using SentenceTransformers.
6. Embeddings are stored in ChromaDB.
7. User asks a question.
8. Hybrid retrieval fetches relevant chunks using vector similarity and BM25.
9. CrossEncoder reranks the retrieved chunks.
10. The top-ranked context is sent to the LLM.
11. The model generates an answer strictly from the provided context.

---

## API Endpoints

| Method | Endpoint      | Description                               |
| ------ | ------------- | ----------------------------------------- |
| POST   | `/api/upload` | Upload a PDF document                     |
| POST   | `/api/ask`    | Ask a question about the uploaded PDF     |
| POST   | `/api/clear`  | Clear a session and delete stored vectors |
| GET    | `/api/status` | Get session status                        |

---

## Environment Variables

Create a `.env` file based on `.env.example`.

```env
UPLOAD_DIR=./data/uploads
VECTOR_DIR=./data/vectorstores
MAX_UPLOAD_MB=25
```

---

## Backend Setup

### Create virtual environment

```bash
python -m venv .venv
```

### Activate environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r backend/requirements.txt
```

### Run backend

```bash
uvicorn backend.main:app --reload --port 8000
```
## Frontend Setup

### Install dependencies

```bash
cd frontend
npm install
```
## Author

**Yaswanth Muniswar Perugu**



