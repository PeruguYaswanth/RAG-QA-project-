RAG PDF QA — LangChain + FastAPI + React

Overview:
A demo RAG web app allowing PDF upload and question answering grounded in the uploaded document.

Structure:
- backend/ — FastAPI backend with RAG pipeline using LangChain + Chroma + SentenceTransformers
- frontend/ — React + TypeScript + Tailwind UI

Quick start (backend)

1. Create and activate a Python venv

python -m venv .venv
.\.venv\Scripts\activate

2. Install dependencies

pip install -r backend/requirements.txt

3. Copy .env.example to .env and set `OPENAI_API_KEY` if you want OpenAI LLM

4. Run backend

uvicorn backend.main:app --reload --port 8000

Quick start (frontend)

1. cd frontend
2. npm install
3. npm run dev

Open http://localhost:5173

Notes
- Uploaded PDFs are limited to 100 pages (server-side check) and 25 MB size (client + server limit).
- Vector DB is persisted under `backend/data/vectorstores/<session_id>`.
- To swap LLM provider, update backend code to instantiate your LLM and provide credentials.

RAG Flow (short)
1. User uploads a PDF -> backend validates pages and size.
2. Backend extracts text per page and splits into overlapping chunks.
3. Chunks are embedded with SentenceTransformers and stored in ChromaDB.
4. On question, backend retrieves top-k relevant chunks, constructs a prompt including only those chunks, and calls LLM.
5. The model is instructed to answer only using provided context; if information isn't present it returns the exact message: "I could not find this information in the uploaded document.".


