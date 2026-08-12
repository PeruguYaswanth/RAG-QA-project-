RAG Backend (FastAPI)

- API endpoints:
  - POST /api/upload -> upload PDF, validate, extract, create embeddings, store vector DB
  - POST /api/ask -> question -> retrieve -> prompt -> answer
  - POST /api/clear -> clear session data

Run:

pip install -r requirements.txt
uvicorn main:app --reload --port 8000

Environment: copy .env.example to .env and fill keys.
