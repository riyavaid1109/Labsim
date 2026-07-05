#!/usr/bin/env python
"""
run_rag_eval.py — evaluate the retrieval pipeline end to end.

Seeds a fresh knowledge base with the same documents a postdoc agent reads
(the richest KB in the simulation), then scores retrieval quality on a
labeled eval set across all three modes: vector-only, BM25-only, and hybrid
(RRF). Optionally adds the cross-encoder reranker.

This is where precision@k / recall@k are genuinely valid: retrieval has
ground truth (which chunks are relevant to a query), unlike the emergent
simulation dynamics.

Requires Ollama running with nomic-embed-text pulled (for embeddings).

    python run_rag_eval.py --k 3
    python run_rag_eval.py --k 3 --rerank
    python run_rag_eval.py --k 5 --evalset rag_evalset.json
"""

import json
import shutil
import argparse

# The documents a postdoc agent is seeded with (shared + postdoc extras),
# mirrored from run_sim.py's seed_agents_with_knowledge().
POSTDOC_KB = [
    "Attention Is All You Need (Vaswani et al., 2017): introduced the Transformer architecture.",
    "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019).",
    "Language Models are Few-Shot Learners (Brown et al., 2020): GPT-3 paper.",
    "A call for reproducibility in machine learning (Pineau et al., 2021).",
    "Measurement invariance and benchmarking pitfalls in NLP evaluation.",
    "Cross-lingual representational alignment: methods and limitations.",
    "Federated optimisation in heterogeneous networks (Li et al., 2020).",
]

EVAL_COLLECTION = "rag_eval_tmp"
EVAL_PERSIST_DIR = "./chroma_rag_eval"


def main():
    p = argparse.ArgumentParser(description="Evaluate the RAG retrieval pipeline")
    p.add_argument("--k", type=int, default=3, help="top-k to score")
    p.add_argument("--corpus", default="rag_eval_corpus.json",
                   help="combined corpus file: {documents:[...], queries:[...]}. "
                        "Falls back to the small postdoc KB + --evalset if absent.")
    p.add_argument("--evalset", default="rag_evalset.json",
                   help="query-only eval set (used only in fallback mode)")
    p.add_argument("--rerank", action="store_true",
                   help="also evaluate with the cross-encoder reranker")
    p.add_argument("--keep", action="store_true",
                   help="keep the temporary chroma store instead of deleting it")
    args = p.parse_args()

    from rag.retriever import RAGRetriever
    from rag.eval import evaluate_retriever, format_report

    # Prefer the richer corpus (documents + distractors + queries) if present;
    # otherwise fall back to the small postdoc KB with a query-only eval set.
    import os
    if os.path.exists(args.corpus):
        with open(args.corpus) as f:
            data = json.load(f)
        documents = data["documents"]
        evalset = data["queries"]
        print(f"Loaded corpus: {len(documents)} documents "
              f"(with distractors), {len(evalset)} queries.")
    else:
        documents = POSTDOC_KB
        with open(args.evalset) as f:
            evalset = json.load(f)
        print(f"Corpus file not found; using small postdoc KB "
              f"({len(documents)} docs), {len(evalset)} queries.")

    # Fresh store every run so results are deterministic w.r.t. content.
    shutil.rmtree(EVAL_PERSIST_DIR, ignore_errors=True)

    reranker = None
    if args.rerank:
        from rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    retriever = RAGRetriever(EVAL_COLLECTION, persist_dir=EVAL_PERSIST_DIR,
                             reranker=reranker)
    retriever.index(documents)
    print(f"Evaluating at k={args.k}...\n")

    results = evaluate_retriever(retriever, evalset, k=args.k)
    print(format_report(results, args.k))

    # Highlight the headline comparison: what does hybrid buy over vector alone?
    if "vector" in results and "hybrid" in results:
        dv = results["hybrid"]["recall_at_k"] - results["vector"]["recall_at_k"]
        dm = results["hybrid"]["mrr"] - results["vector"]["mrr"]
        print(f"\nHybrid vs vector-only:  recall@{args.k} "
              f"{'+' if dv >= 0 else ''}{dv:.3f}   MRR "
              f"{'+' if dm >= 0 else ''}{dm:.3f}")

    if args.rerank:
        print("(Reranker applied on top of the fused candidate set.)")

    if not args.keep:
        shutil.rmtree(EVAL_PERSIST_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()