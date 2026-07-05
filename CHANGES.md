# Labsim — Changes Log

A point-by-point record of everything added or modified, and why.

## 1. RAG pipeline upgrades

### Hybrid retrieval (`rag/retriever.py`)
- Added BM25 keyword search alongside the existing vector similarity search.
- Reason: vector search captures meaning but misses exact terms (method names,
  acronyms); BM25 catches exact terms but has no semantic understanding. Using
  both covers each other's blind spots.
- Combined the two rankings with Reciprocal Rank Fusion (RRF, k=60).
- Reason: RRF merges two ranked lists without needing to calibrate their
  incomparable score scales; documents ranked well by both float to the top.
- The BM25 index is rebuilt in memory from the stored ChromaDB collection on
  startup, so persisted knowledge bases stay consistent across restarts.
- Retrieval mode is selectable: `mode="hybrid"` (default), `"vector"`, `"bm25"`.

### Recursive chunking (`rag/chunking.py`, new file)
- Documents are split along structural boundaries (markdown headers →
  paragraphs → sentences → words) before indexing, with 512-char chunks and
  64-char overlap.
- Reason: indexing whole documents gives coarse, unfocused retrieval; chunking
  keeps each retrieved unit semantically tight. Overlap prevents losing context
  at chunk boundaries. Chunk size chosen to keep embeddings focused for
  nomic-embed-text.

### Cross-encoder reranker (`rag/reranker.py`, new file)
- Optional bge-reranker-base stage that re-scores retrieved candidates by
  reading (query, document) pairs jointly, then keeps the top-k.
- Reason: bi-encoder retrieval (vector search) scores query and document
  independently, which is fast but lossy; a cross-encoder gives sharper
  relevance. Standard pattern: cheap high-recall retrieval, then expensive
  high-precision reranking.
- Lazy-loaded and degrades gracefully (falls back to fusion order) if
  sentence-transformers is not installed.

### Retrieval-time abstention (`rag/retriever.py`)
- Added a `min_similarity` threshold: when the best retrieved chunk's cosine
  similarity is below the cutoff, retrieval returns nothing instead of forcing
  through loosely-related chunks.
- Reason: the retriever previously always returned its top-k, even for
  questions with no real answer in the knowledge base — giving the agent
  irrelevant context to confabulate from. Abstention removes that context so
  the agent structurally cannot invent a grounded-looking answer.
- `_vector_search_scored()` was added to expose ChromaDB cosine distances
  (converted to similarity = 1 - distance) needed for the threshold.

## 2. RAG evaluation

### Retrieval metrics harness (`rag/eval.py`)
- Measures precision@k, recall@k, MRR, and hit@k per retrieval mode against a
  labeled eval set (queries → relevant chunk substrings).
- Reason: to quantify what hybrid search and reranking actually buy over
  vector-only retrieval, rather than assuming they help.
- Substring-based relevance labels: a chunk counts as relevant if it contains
  any labeled substring, so labels survive changes to chunking parameters.

### Standalone retrieval eval runner (`run_rag_eval.py`, `rag_eval_corpus.json`, new files)
- Seeds a fresh knowledge base and scores all three retrieval modes on a
  20-document corpus with 12 queries.
- The corpus deliberately includes distractor documents that share keywords
  with a query but are not the answer (e.g. Vision Transformer / Reformer /
  RoBERTa as distractors for "the original Transformer").
- Reason: an easy corpus scores 100% on every mode and can't show the
  difference between them; distractors are where BM25, vector, and hybrid
  diverge. Result: hybrid reached MRR 1.000 vs 0.958 for vector-only or
  BM25-only — hybrid ranked the correct document first on every query.

### Faithfulness / hallucination judge (`rag/eval.py`)
- LLM-as-judge scoring of (question, context, answer) triples: faithfulness
  (fraction of claims supported by context), answer relevancy, and
  hallucination rate (fraction of claims the context does NOT support).
- Reason: retrieval precision measures whether the right chunks are found;
  faithfulness measures whether the agent's answer actually stays grounded in
  them — a different and equally important failure mode.
- Hallucination rate is derived by having the judge count supported vs
  unsupported claims (schema updated to include `supported_claims`).

### Faithfulness eval runner (`run_faithfulness_eval.py`, new file)
- Builds a real agent, seeds its KB, then for each question: retrieves context,
  has the agent answer, and judges the result.
- Uses abstract-length KB documents and a mix of answerable questions and
  "trap" questions whose answers are NOT in the KB.
- Reason: traps are the sharpest test of grounding — a faithful agent should
  decline them, a hallucinating one invents answers.
- Runs three modes and compares: ungrounded, grounded (prompt instruction), and
  grounded + abstention (similarity threshold).
- Includes a calibration step that prints in-KB vs trap similarities and
  suggests a threshold from the data.

### Findings from the faithfulness eval
- Enriching KB documents from one-line stubs to abstract-length passages cut
  hallucination on answerable questions from 0.70 to ~0.07.
- Prompt-level grounding ("answer only from context") did NOT fix trap
  hallucination — a small (3B) model still confabulated on 100% of traps.
- Retrieval-time abstention reduced overall hallucination from 0.58 to 0.38
  (36% relative), catching 2 of 3 traps.
- Calibration revealed a limitation: trap questions sharing surface vocabulary
  with the corpus score deceptively high on cosine similarity, so a single
  threshold cannot perfectly separate answerable from unanswerable queries.

## 3. LLM layer

### Model-agnostic client (`llm/client.py`, new file)
- One interface over Ollama, OpenAI, and Anthropic. Backend selected by
  `--provider` or the `LAB_SIM_PROVIDER` env var.
- Reason: lets the whole simulation swap LLM backends without code changes —
  e.g. run locally on Ollama, or use a stronger API model as a judge.
- OpenAI/Anthropic SDKs are imported lazily so only Ollama is a hard dependency.

### Structured outputs (`llm/client.py`, `agents/base.py`)
- Peer reviews are generated with JSON-schema-constrained output instead of
  free text parsed by regex.
- Reason: constrained decoding is far more reliable than hoping the model
  formats its output correctly; falls back to the old regex parse if the
  structured call fails.
- `query_llm_json(prompt, schema)` added as the structured-output entry point
  alongside `query_ollama()`.

### Tool calling (`agents/tools.py`, new file)
- Agents can act through typed tool calls: `search_kb`, `cite_source`,
  `flag_dispute`, `respond`, in a ReAct-style loop.
- Reason: structured tool use is what "agentic" means in practice, versus
  free-text responses.
- Implemented with schema-constrained JSON (one action per turn) rather than
  provider-native tool APIs, so it works identically on plain llama3.2 which
  has no native tool support. Exposed via `agent.speak_with_tools()`.

### Observability (`llm/observability.py`, new file)
- Every LLM call is traced to SQLite: prompt, response, latency, token counts,
  and which agent/method made it.
- Reason: to profile the pipeline and see where cost and latency go. A summary
  prints at the end of each run; traces export to JSONL.
- Revealed that structured peer-review generation is the dominant latency cost
  (~22-24 seconds per call).

## 4. Grounding instruction (`agents/base.py`)
- `speak()` gained an optional `grounded` flag that adds a strict-grounding
  instruction ("answer using only the context; say what you don't know").
- When grounded and retrieval abstains (empty context), the agent is instructed
  to decline rather than answer from memory.
- Reason: the standard prompt-level mitigation for hallucination, and needed to
  measure whether it works (it helped on answerable questions but not traps).
- Off by default so the simulation's normal conversational behavior is
  unchanged; only the faithfulness eval turns it on.

## 5. Simulation fix — PI assignment fairness (`agents/pi.py`)
- Rewrote `assign_problem()` so the fairness parameter controls opportunity
  ROTATION, not just a favorite bonus.
- Reason (the bug): the old logic picked the single highest-scoring agent every
  time (deterministic argmax). Since scores barely changed between rounds, one
  agent monopolized leadership in BOTH fair and biased conditions — the fairness
  knob only changed WHICH agent won, never that one agent won everything. This
  made the fair-vs-biased experiment produce nearly identical results.
- The fix: high fairness adds an equity boost for under-used agents and
  softmax-samples the selection, spreading leadership (including to students);
  low fairness stays argmax-greedy and concentrates on the favorite. The PI now
  tracks per-agent lead counts.
- Result: fair spreads leadership across all agents; biased concentrates it on
  one — the conditions now diverge, and reputation spread climbs to ~4.0 under
  bias vs ~1.5 under fairness.

## 6. Experiment infrastructure

### Multi-seed runner (`run_experiment.py`, new file)
- Runs multiple seeds per condition (fair vs biased), each as an isolated
  subprocess with its own ChromaDB, writing per-seed JSON histories.
- Reason: a single run is noisy (LLM sampling + random init); averaging over
  seeds with error bands is needed for a defensible result.
- The biased condition favors a junior student (agent 0) rather than a strong
  postdoc, to maximize contrast with merit-based fair assignment.
- The manifest merges across runs so conditions can be run separately and still
  plotted together.

### Comparison plotter (`plot_experiment.py`, new file)
- Loads all seeds per condition and produces mean curves with ±SD (or ±SEM)
  error bands.
- Two figures: lab-level metrics (motivation, reputation spread, review score,
  dispute rate) and motivation faceted by role (PI vs postdoc vs student).
- Reason: turns individual noisy runs into averaged, error-banded figures; the
  role-faceted view reveals stratification that the lab-wide average hides.

### Reproducibility (`run_sim.py`)
- Added a `--seed` flag that seeds Python and NumPy RNG.
- Added `--provider` and `--trace-db` flags for the model backend and
  observability.
- Note: seeding pins agent init and RNG but not the LLM's own sampling, which
  is why averaging over seeds still matters.

## 7. Tests (`test_upgrades.py`, new file)
- 19 offline unit tests covering chunking, hybrid retrieval, RRF math, reranker
  fallback, eval metrics, observability, structured-output parsing, the tool
  loop, and structured reviews.
- Reason: all mocked (embeddings and LLM calls), so the logic can be verified
  without a running Ollama server.

## 8. Documentation
- `README.md` rewritten as a single coherent document: setup with llama3.2,
  quick start, multi-seed experiments, RAG pipeline, LLM layer, how fairness
  works, project structure, metrics, and known limitations.
- `requirements.txt` updated: added rank-bm25; sentence-transformers, openai,
  and anthropic listed as optional.
- Eval sets added: `rag_eval_corpus.json` (retrieval), `evalset.example.json`.
