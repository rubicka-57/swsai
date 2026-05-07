# Company RAG Chatbot API

Backend-only Retrieval Augmented Generation API built with FastAPI, LangChain, ChromaDB, sentence-transformers embeddings, and Gemini.

The chatbot answers only from PDFs in the `documents/` folder. If the answer is not present in retrieved context, it returns:

```text
I don't have that information in the company documents.
```

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and set your Gemini key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Add company PDFs to:

```text
backend/documents/
```

## Run

```bash
uvicorn app:app --reload
```

API docs will be available at:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### POST `/ingest`

Processes every PDF in `documents/`, splits content into chunks, embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`, and stores them in ChromaDB.

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

### POST `/chat`

Request:

```json
{
  "question": "What is the leave policy?"
}
```

Response:

```json
{
  "answer": "...",
  "sources": ["Leave Policy.pdf"]
}
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is the leave policy?\"}"
```

## Project Structure

```text
backend/
|-- app.py
|-- ingest.py
|-- requirements.txt
|-- .env
|-- .env.example
|-- README.md
|-- documents/
|-- chroma_db/
`-- utils/
    |-- __init__.py
    |-- config.py
    |-- gemini.py
    `-- vector_store.py
```
