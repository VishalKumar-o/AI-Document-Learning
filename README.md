
# AI-Based Document Learning and Question Answering System Using RAG

An AI-powered document learning application that allows users to upload PDF documents and ask questions about their contents. The system uses Natural Language Processing (NLP), semantic search, and Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate document-grounded answers with page references.

## Features

- Upload and process PDF documents.
- Extract text from PDF pages.
- Split extracted text into overlapping chunks.
- Generate text embeddings using Sentence Transformers.
- Store and search embeddings using FAISS.
- Retrieve relevant passages using semantic and keyword search.
- Generate answers using the Google Gemini API.
- Automatically fall back to a local Ollama model if Gemini fails.
- Include relevant document page references in answers.
- Provide REST APIs using FastAPI.
- Support local document processing and local LLM inference.

## Technology Stack

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic

### Artificial Intelligence and NLP
- Natural Language Processing (NLP)
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- `all-MiniLM-L6-v2` embedding model
- FAISS vector search
- Google Gemini API
- Ollama
- Qwen 2.5 7B

### Document Processing
- PyMuPDF
- Text chunking
- Semantic retrieval
- Keyword-based retrieval

### Frontend
- React
- JavaScript
- HTML
- CSS
- Vite

## System Architecture

1. The user uploads a PDF document through the application.
2. FastAPI receives and processes the document.
3. PyMuPDF extracts text and page information.
4. The extracted text is divided into overlapping chunks.
5. Sentence Transformers converts the chunks into numerical embeddings.
6. FAISS indexes the embeddings for similarity-based retrieval.
7. Keyword search helps retrieve relevant passages.
8. The user submits a question about the document.
9. The retrieval system selects relevant passages.
10. The LLM service sends the question and passages to Gemini.
11. If Gemini fails, the service attempts to use the local Ollama model.
12. The generated answer and retrieved source information are returned to the user.

## Project Structure

```text
AI-Document-Learning/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── document_routes.py
│   │   │   ├── health_routes.py
│   │   │   └── question_routes.py
│   │   ├── core/
│   │   ├── repositories/
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── chunking_service.py
│   │   │   ├── document_processing_service.py
│   │   │   ├── embedding_service.py
│   │   │   ├── keyword_search_service.py
│   │   │   ├── llm_service.py
│   │   │   ├── pdf_service.py
│   │   │   ├── retrieval_service.py
│   │   │   └── text_processing_service.py
│   │   └── main.py
│   ├── requirements.txt
│   └── .env                  # Local only; do not commit
├── frontend/                 # React application
├── .gitignore
└── README.md
```

## Requirements

- Python compatible with the pinned dependencies
- Git
- Internet access for installing dependencies, downloading the embedding model, and using Gemini
- A valid Gemini API key
- Ollama and the configured local model for fallback inference

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/VishalKumar-o/AI-Document-Learning.git
cd AI-Document-Learning
```

### 2. Create a Python Virtual Environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file inside the `backend/` directory.

Add the following configuration:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_URL=http://127.0.0.1:11434/api/chat
```

Replace `your_gemini_api_key` with your own API key.

**Security:** Never commit `.env` to GitHub or expose your API key in source code, screenshots, or logs.

### 5. Configure Ollama Fallback

Install Ollama if it is not already installed.

Download the configured model:

```bash
ollama pull qwen2.5:7b
```

Verify that Ollama is available:

```bash
ollama list
```

The backend expects the Ollama chat API at:

```text
http://127.0.0.1:11434/api/chat
```

The local model is used when the Gemini request fails and Ollama is available.

### 6. Start the Backend

From the `backend/` directory, with the virtual environment activated:

```bash
uvicorn app.main:app --reload
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Alternative API documentation:

```text
http://127.0.0.1:8000/redoc
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Check basic application information |
| GET | `/health` | Check backend health |
| POST | `/documents/upload` | Upload and process a PDF document |
| POST | `/questions/ask` | Ask a question about an uploaded document |

Use the interactive Swagger documentation at `/docs` to inspect request fields, upload documents, and test questions.

## Example Workflow

1. Start the backend.
2. Open the API documentation at `http://127.0.0.1:8000/docs`.
3. Upload a PDF using `/documents/upload`.
4. Save the returned document ID.
5. Submit a question through `/questions/ask`, providing the document ID and question.
6. Review the generated answer and returned source passages.

The exact request and response formats are defined by the current API schemas and can be inspected in `/docs`.

## LLM Provider Fallback

The application attempts to use Gemini first.

- **Primary provider:** Google Gemini API.
- **Fallback provider:** Local Ollama with Qwen 2.5 7B.

If the Gemini request fails, the service attempts local Ollama inference. Fallback depends on the local Ollama service and model being available.

A Gemini quota or billing error does not automatically mean the quota will reset immediately. Local fallback can help keep the application usable, but it still requires sufficient laptop resources.

## Data and Security

- Keep `.env` out of version control.
- Do not commit uploaded PDFs or private documents.
- Do not commit generated indexes, caches, virtual environments, or local databases.
- Never publish API keys in source code.
- Use appropriate access controls before deploying the application for multiple users.
- The embedding model may download its model files on first use.

## Testing

The `backend/test_*.py` files currently contain manual service demonstration scripts. They are not necessarily automated pytest tests.

For example, from the project root:

```bash
python backend/test_chunking_service.py
```

Run individual scripts only when their required dependencies and configuration are available. Automated test coverage can be added as the project develops.

## Current Project Status

Backend development in progress.

Planned improvements include:

- Improving retrieval relevance and ranking.
- Supporting additional document formats.
- Adding automated unit and integration tests.
- Improving frontend and backend integration.
- Adding authentication and document management.
- Improving answer evaluation and source citations.
