"""
eval.py — retrieval and generation evaluation harness.

Two layers, usable independently:

1. Retrieval metrics (no LLM needed): precision@k, recall@k, MRR, hit@k
   against a labeled eval set of (query → relevant chunk substrings).
   Runs the same query through vector / bm25 / hybrid modes so you can
   quantify what hybrid search actually buys you.

2. Faithfulness / answer relevancy (LLM-as-judge, RAGAS-style but with no
   langchain dependency): given (question, retrieved context, answer), a
   judge model scores whether the answer is grounded in the context and
   whether it addresses the question.

Eval set format (JSON):
    [
      {"query": "how do transformers handle long sequences?",
       "relevant": ["attention complexity", "sparse attention"]},
      ...
    ]
A retrieved chunk counts as relevant if it contains any of the `relevant`
substrings (case-insensitive). Substring labels keep annotation cheap —
you label the *facts* that should be retrieved, not exact chunk boundaries
(which shift whenever chunking parameters change).

CLI:
    python -m rag.eval --collection agent_0_riya --evalset evalset.json --k 3
"""

import json
import logging
import argparse

logger = logging.getLogger("lab_sim")

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number", "minimum": 0, "maximum": 1},
        "answer_relevancy": {"type": "number", "minimum": 0, "maximum": 1},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["faithfulness", "answer_relevancy", "unsupported_claims"],
}


# ── Layer 1: retrieval metrics ────────────────────────────────────────────

def _is_relevant(chunk: str, relevant_substrings: list[str]) -> bool:
    chunk_l = chunk.lower()
    return any(s.lower() in chunk_l for s in relevant_substrings)


def score_retrieval(retrieved: list[str], relevant_substrings: list[str],
                    k: int) -> dict:
    """Per-query IR metrics for one ranked retrieval result."""
    topk = retrieved[:k]
    hits = [_is_relevant(c, relevant_substrings) for c in topk]
    n_hits = sum(hits)

    # Recall denominator: how many *distinct labeled facts* were found
    # anywhere in top-k, over the number of labeled facts.
    found_facts = {
        s for s in relevant_substrings
        if any(s.lower() in c.lower() for c in topk)
    }
    recall = len(found_facts) / len(relevant_substrings) if relevant_substrings else 0.0

    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break

    return {
        "precision_at_k": n_hits / k if k else 0.0,
        "recall_at_k": recall,
        "mrr": mrr,
        "hit_at_k": 1.0 if n_hits > 0 else 0.0,
    }


def evaluate_retriever(retriever, evalset: list[dict], k: int = 3,
                       modes: tuple[str, ...] = ("vector", "bm25", "hybrid")) -> dict:
    """
    Run every eval query through each retrieval mode and average the metrics.

    Returns {mode: {metric: mean_value, "n": n_queries}}.
    """
    results: dict[str, dict] = {}
    for mode in modes:
        agg = {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "hit_at_k": 0.0}
        for item in evalset:
            retrieved = retriever.retrieve(item["query"], k=k, mode=mode)
            scores = score_retrieval(retrieved, item["relevant"], k)
            for m in agg:
                agg[m] += scores[m]
        n = max(1, len(evalset))
        results[mode] = {m: round(v / n, 4) for m, v in agg.items()}
        results[mode]["n"] = len(evalset)
    return results


def format_report(results: dict, k: int) -> str:
    lines = [f"Retrieval evaluation (k={k})",
             f"{'mode':<10}{'P@k':>8}{'R@k':>8}{'MRR':>8}{'hit@k':>8}{'n':>6}"]
    for mode, m in results.items():
        lines.append(f"{mode:<10}{m['precision_at_k']:>8.3f}{m['recall_at_k']:>8.3f}"
                     f"{m['mrr']:>8.3f}{m['hit_at_k']:>8.3f}{m['n']:>6}")
    return "\n".join(lines)


# ── Layer 2: LLM-judged faithfulness ──────────────────────────────────────

def judge_faithfulness(question: str, context: str, answer: str,
                       caller: str = "eval.judge") -> dict | None:
    """
    LLM-as-judge scoring of one (question, context, answer) triple.

    faithfulness      — fraction of the answer's claims supported by context
    answer_relevancy  — how directly the answer addresses the question
    unsupported_claims — claims in the answer with no grounding in context
    """
    from utils import query_llm_json  # local import: layer 1 stays LLM-free

    prompt = f"""You are evaluating a RAG system's output. Judge strictly.

QUESTION:
{question}

RETRIEVED CONTEXT (the ONLY permitted evidence):
{context}

ANSWER:
{answer}

Score:
- faithfulness (0-1): fraction of the answer's factual claims that are
  directly supported by the retrieved context. Claims from outside the
  context lower this score even if true.
- answer_relevancy (0-1): how directly the answer addresses the question.
- unsupported_claims: list each claim in the answer that the context does
  not support (empty list if none)."""
    return query_llm_json(prompt, FAITHFULNESS_SCHEMA, caller=caller)


def evaluate_agent_responses(triples: list[dict]) -> dict:
    """
    Batch-judge a list of {"question", "context", "answer"} triples.
    Returns mean faithfulness / relevancy plus per-item results.
    """
    per_item, faith, rel = [], [], []
    for t in triples:
        verdict = judge_faithfulness(t["question"], t["context"], t["answer"])
        per_item.append(verdict)
        if verdict:
            faith.append(float(verdict["faithfulness"]))
            rel.append(float(verdict["answer_relevancy"]))
    n = len(faith)
    return {
        "mean_faithfulness": round(sum(faith) / n, 4) if n else None,
        "mean_answer_relevancy": round(sum(rel) / n, 4) if n else None,
        "n_judged": n,
        "per_item": per_item,
    }


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG collection")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--evalset", required=True, help="JSON eval set path")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--rerank", action="store_true",
                        help="also apply the cross-encoder reranker")
    args = parser.parse_args()

    from rag.retriever import RAGRetriever
    reranker = None
    if args.rerank:
        from rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    retriever = RAGRetriever(args.collection, persist_dir=args.persist_dir,
                             reranker=reranker)
    with open(args.evalset) as f:
        evalset = json.load(f)

    results = evaluate_retriever(retriever, evalset, k=args.k)
    print(format_report(results, args.k))


if __name__ == "__main__":
    main()
