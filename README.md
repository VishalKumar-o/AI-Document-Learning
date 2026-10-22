# AI-Based Document Learning and Question Answering System Using RAG (Retrieval-Augmented Generation)

An AI-powered document learning application that retrieves relevant information from uploaded PDF documents and uses a local language model to answer questions.

## Features

- Upload and process PDF documents.
- Extract text and split it into chunks.
- Generate text embeddings using Sentence Transformers.
- Retrieve relevant passages using FAISS.
- Generate document-based answers using Ollama.
- Return relevant source information from retrieved passages.
- Provide health and search-debugging endpoints.

## Technology Stack

- Python
- FastAPI
- Sentence Transformers
- FAISS
- NumPy
- PyMuPDF
- Ollama
- Qwen 2.5 7B

## Requirements

- Python compatible with the installed dependencies
- Ollama installed and running
- The `qwen2.5:7b` model downloaded through Ollama

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/VishalKumar-o/AI-Document-Learning.git
cd AI-Document-Learning
```

### 2. Create a virtual environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare Ollama

Install Ollama if necessary, start its service, and download the model:

```bash
ollama pull qwen2.5:7b
```

Ensure Ollama is accessible at `http://127.0.0.1:11434`.

### 5. Start the backend

From the `backend` directory, run:

```bash
uvicorn main:app --reload
```

The API should be available at `http://127.0.0.1:8000`.

Interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

- `GET /` — Application information
- `GET /health` — Health check
- `POST /upload-pdf` — Upload and process a PDF
- `POST /ask` — Ask a question
- `POST /query` — Query the document retrieval system
- `POST /debug-search` — Inspect retrieval results

Check the interactive API documentation for the required request fields and response formats.

## Notes

- The language model runs locally through Ollama.
- The embedding model may need to download its files on first use.
- Uploaded documents and generated indexes are local application data and are not included in this repository.
- The frontend is not included yet.

## Project Status

Backend development in progress.
