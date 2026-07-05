# Research Lab Simulation

A multi-agent simulation of a research lab — agents collaborate on problems,
write papers, and peer-review each other's work. Every agent response is
grounded in their own private knowledge base via RAG.

## What it studies

How collaboration breaks down or succeeds in a research lab, specifically:
- Does knowledge asymmetry (agents with different KBs) improve or harm review quality?
- Does PI bias (fairness parameter) affect who gets credit over time?
- How does motivation evolve when agents are repeatedly ignored or miscredited?

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.ai) running locally.

```bash
# Pull the models
ollama pull llama3
ollama pull nomic-embed-text

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
# Basic run — 3 students, 2 postdocs, 5 problems, fair PI
python run_sim.py --students 3 --postdocs 2 --timesteps 5 --fairness 1.0

# Biased PI run — compare outcomes
python run_sim.py --students 3 --postdocs 2 --timesteps 5 --fairness 0.2 --pi-favorite 0

# Save results
python run_sim.py --timesteps 5 --save-json results.json
```

## Project structure

```
lab_sim/
  agents/
    base.py           # BaseAgent: motivation, reputation, private KB, speak()
    phd_student.py    # Junior agent — fragile motivation, grows with success
    postdoc.py        # Mentor + competitor — writes substantive RAG reviews
    pi.py             # Final authority — fairness parameter controls bias
  rag/
    retriever.py      # ChromaDB + nomic-embed-text — one instance per agent
  simulation/
    knowledge_space.py  # Shared KB + message log + metrics (bus factor, entropy)
    discussion.py       # RAG-grounded multi-agent lab meeting
    peer_review.py      # Full review cycle: assign → review → rebuttal → decide
    lab.py              # Main simulation loop
  problems/           # .md files describing research problems
  utils.py            # query_ollama() — single LLM injection point
  run_sim.py          # Entry point with argparse + matplotlib plots
```

## Key design decisions

**Private KBs per agent** — each agent has their own ChromaDB collection.
When they speak, RAG pulls from *their* reading history. Two agents can
give genuinely different, grounded responses to the same paper because
they've read different literature.

**Single injection point** — all LLM calls go through `query_ollama()` in
`utils.py`. To swap models, add logging, or inject system-wide context,
change one function.

**Fairness as an experimental variable** — set `--fairness 0.0` for a PI
who always picks favorites; `--fairness 1.0` for pure merit. Run both and
compare how reputation and motivation diverge.

## Metrics tracked

- Agent motivation (0–10) over time
- Agent reputation (cumulative)
- Bus factor (how many people account for 80% of discussion)
- Message entropy (diversity of participation)
- Average peer review score per timestep
- Credit dispute rate

## Adding your own problems

Drop `.md` files into `./problems/`. Each file is one research problem.
Format is free — the simulation treats the full text as the problem description.

```markdown
# Problem title

Description of the research challenge, dataset, or question.
The more specific, the more grounded the agent responses will be.
```

## RAG pipeline (v2)

Retrieval is now hybrid with optional reranking:

```
query ─┬─ vector search (nomic-embed-text, cosine)
       └─ BM25 keyword search (rank_bm25)
              │
     reciprocal rank fusion (k=60)
              │
   [optional] cross-encoder reranker (bge-reranker-base)
              │
           top-k chunks
```

- **Hybrid search** — vector search captures semantics; BM25 catches exact
  terms (method names, acronyms). RRF fuses the two rank lists without score
  calibration. Modes: `retrieve(query, k, mode="hybrid"|"vector"|"bm25")`.
- **Chunking** — documents are recursively chunked before indexing
  (markdown headers → paragraphs → sentences), 512 chars with 64-char
  overlap (see `rag/chunking.py` for the rationale).
- **Reranking** — pass a `CrossEncoderReranker` to `RAGRetriever` to add a
  precision stage after fusion. Requires `sentence-transformers`; degrades
  gracefully without it.
- **Evaluation** — `rag/eval.py` measures precision@k, recall@k, MRR, and
  hit@k per retrieval mode against a labeled eval set, plus LLM-judged
  faithfulness / answer-relevancy for generated responses (RAGAS-style,
  no langchain):

```bash
python -m rag.eval --collection shared --evalset evalset.example.json --k 3
```

## LLM layer (v2)

- **Model abstraction** — all calls route through `llm/client.py`:
  one interface over Ollama / OpenAI / Anthropic. Swap backends with
  `--provider openai` or `LAB_SIM_PROVIDER=anthropic`; `query_ollama()`
  is kept as a backward-compatible wrapper.
- **Structured outputs** — peer reviews are JSON-schema-constrained
  (`REVIEW_SCHEMA` in `agents/base.py`) via Ollama's native `format`
  parameter, with the original regex parse as fallback. Use
  `query_llm_json(prompt, schema)` for any new structured call.
- **Tool use** — agents can act through typed tool calls
  (`search_kb`, `cite_source`, `flag_dispute`, `respond`) via
  `agent.speak_with_tools(task)` — a ReAct-style loop in `agents/tools.py`.
  Implemented with schema-constrained JSON (one action per turn) instead of
  provider-native tool APIs, so it works identically on plain llama3.
- **Observability** — every LLM call is traced to SQLite
  (`./llm_traces.db`): prompt, response, latency, token counts, caller.
  A summary prints at the end of each run; export with
  `llm.observability.export_traces()`. Disable with `--trace-db ''`.

## Tests

```bash
python test_upgrades.py   # offline — mocks embeddings and LLM calls
```

## Multi-seed experiments

A single run is noisy (llama3 sampling + random init). To get a defensible
figure, run several seeds per condition and average:

```bash
# 3 seeds each for fair vs biased. Start with --timesteps 2 to validate,
# then scale up (a 5-timestep run is ~25-30 min on an M1).
python run_experiment.py --seeds 3 --timesteps 5

# then plot mean curves with ±SD error bands (--sem for standard error)
python plot_experiment.py
```

`run_experiment.py` runs each (condition, seed) as an isolated subprocess
with its own ChromaDB, writing `experiments/<condition>/seed_<n>.json` plus
a manifest. `plot_experiment.py` reads the manifest and produces:

- `compare_lab_metrics.png` — mean motivation, reputation spread, review
  score, and dispute rate, fair vs biased overlaid with error bands.
- `compare_motivation_by_role.png` — motivation faceted by role, so PI
  morale can be compared against students/postdocs across conditions.

Use `--seed` on `run_sim.py` directly for a single reproducible run. Note
this pins agent init and Python RNG but not Ollama's own sampling, which is
why averaging over seeds matters.
