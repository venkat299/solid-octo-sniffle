# Job Role Analyzer

This project implements a reusable job role analysis engine with support for FAISS + BGE embeddings, modular LLM prompts, and an interactive web interface.

## Features

- Normalize job descriptions into concise summaries and extract ranked competencies.
- Persist role analyses to SQLite and reuse them when similar descriptions are encountered.
- Configurable prompt templates, LLM clients, and embedding backends.
- Responsive FastAPI UI for uploading job descriptions and visualizing results.

## Getting Started

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt  # create this file if you manage dependencies centrally
pip install fastapi uvicorn sentence-transformers jinja2 numpy
```

The UI automatically falls back to a lightweight hashing embedder if the `sentence-transformers` package cannot be installed, but FAISS+BGE is recommended for best quality.

### Running Tests

```bash
python -m pytest
```

### Launching the Web UI

```bash
uvicorn webapp.main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser. Submit a job title, description, and target experience to view the generated summary and competencies.

## Configuration

Settings such as the similarity threshold, embedding model, and prompt directory are managed through `config.yaml`. Each major module can specify distinct LLM providers and models via this configuration file, enabling granular control over model selection.
