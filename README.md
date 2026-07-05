# Financial Document Analyzer

A multi-agent AI system built with **CrewAI** and **FastAPI** that analyzes financial PDFs and provides document verification, financial analysis, investment insights, and risk assessment.

---

## Bugs Fixed

### Deterministic Bugs

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `agents.py` | `llm = llm` — undefined variable | Initialized with `LLM(model=..., api_key=...)` |
| 2 | `agents.py` | `tool=[...]` — wrong parameter name | Changed to `tools=[...]` |
| 3 | `agents.py` | `from crewai.agents import Agent` — wrong import | Changed to `from crewai import Agent, LLM` |
| 4 | `tools.py` | `from crewai_tools import tools` — invalid import | Removed, replaced with correct imports |
| 5 | `tools.py` | `Pdf(...)` — never imported, NameError | Replaced with `PyPDFLoader` from `langchain_community` |
| 6 | `tools.py` | All methods were `async` | Removed `async`, CrewAI needs sync functions |
| 7 | `tools.py` | No `@tool` decorator | Added `@tool` with docstrings to all methods |
| 8 | `task.py` | Wrong agents assigned to all tasks | Fixed: `verifier`, `investment_advisor`, `risk_assessor` |
| 9 | `main.py` | Endpoint name conflicted with imported task | Renamed to `analyze_document_endpoint` |
| 10 | `main.py` | `file_path` never passed to agents | Added to `kickoff(inputs={...})` |

### Prompt Bugs

All 4 agents and tasks had broken prompts instructing them to hallucinate, invent fake URLs, ignore compliance, and contradict themselves. All rewritten with professional, data-grounded, compliance-aware instructions.

---

## Setup

### 1. Clone & install
```bash
git clone https://github.com/riyavaid1109/financial-document-analyzer.git
cd financial-document-analyzer
python -m venv venv
source venv/bin/activate
pip install "crewai==0.130.0" "crewai-tools==0.47.1" fastapi uvicorn python-dotenv python-multipart pypdf langchain-community groq --use-deprecated=legacy-resolver
```

### 2. Environment variables
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```
- Free Groq key → [console.groq.com](https://console.groq.com)
- Free Serper key → [serper.dev](https://serper.dev)

### 3. Run
```bash
mkdir data outputs
python main.py
```
API runs at `http://localhost:8000`

---

## API

#### `GET /`
Health check.

#### `POST /analyze`
Analyze a financial PDF.

| Field | Type | Required |
|-------|------|----------|
| `file` | PDF | Yes |
| `query` | string | No |

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@report.pdf" \
  -F "query=Summarize the company's financial health"
```

Interactive docs → `http://localhost:8000/docs`
