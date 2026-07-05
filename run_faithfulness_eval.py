#!/usr/bin/env python
"""
run_faithfulness_eval.py — measure how grounded agent responses are, and
whether a grounding instruction improves faithfulness.

Pipeline per question:
  1. an agent retrieves context from its private KB (retrieve_context)
  2. the same agent answers, given that context (speak)
  3. an LLM judge scores the (question, context, answer) triple for
     faithfulness, answer relevancy, and hallucination rate

Runs each question in TWO modes and compares:
  - ungrounded: the default speak() (agent may elaborate from parametric memory)
  - grounded:   speak(grounded=True) adds a "use ONLY the context" instruction

This turns a single faithfulness number into a before/after result: does the
standard RAG grounding mitigation actually reduce hallucination here?

Some questions are "traps" whose answers are NOT in the KB — a faithful agent
should decline rather than invent. Trap hallucination rate is the sharpest
signal of grounding behavior.

Requires Ollama running (llama3.2 for generation + judging, nomic-embed-text
for retrieval).

    python run_faithfulness_eval.py --model llama3.2 --save faith.json
"""

import json
import shutil
import argparse

# Abstract-length KB documents — enough real content for an answer to be
# genuinely grounded in (vs. the earlier one-line stubs, which forced any
# substantive answer to add unsupported detail).
KB_DOCS = [
    "Attention Is All You Need (Vaswani et al., 2017) introduced the Transformer, "
    "a sequence model based entirely on self-attention, dispensing with recurrence "
    "and convolutions. It uses multi-head attention and positional encodings, and "
    "was evaluated on English-to-German and English-to-French machine translation.",

    "BERT (Devlin et al., 2019) pre-trains deep bidirectional Transformers using two "
    "objectives: masked language modeling, which predicts randomly masked tokens, and "
    "next-sentence prediction. It is pre-trained on the BooksCorpus and English "
    "Wikipedia, then fine-tuned for downstream tasks.",

    "Language Models are Few-Shot Learners (Brown et al., 2020) introduced GPT-3, a "
    "175-billion-parameter autoregressive model. Its central finding is that scaling "
    "enables in-context few-shot learning: the model performs new tasks from a few "
    "examples in the prompt, without gradient updates or fine-tuning.",

    "A call for reproducibility in machine learning (Pineau et al., 2021) proposes a "
    "reproducibility checklist covering code release, dataset documentation, reported "
    "hyperparameters, and multiple runs with variance. It argues that reporting "
    "standards, not just released code, are needed for reliable ML science.",

    "Measurement invariance and benchmarking pitfalls in NLP evaluation discusses how "
    "popular benchmarks can reward superficial cues over genuine understanding, and how "
    "score comparisons across models can be invalid when evaluation conditions differ.",

    "Cross-lingual representational alignment surveys methods for aligning embeddings "
    "across languages, including supervised and unsupervised mapping, and notes "
    "limitations when languages are typologically distant or low-resource.",

    "Federated optimisation in heterogeneous networks (Li et al., 2020) introduces "
    "FedProx, which adds a proximal term to the local objective to handle non-IID data "
    "and system heterogeneity across clients, improving stability over FedAvg when "
    "client data and compute differ.",
]

# in_kb=True  → answer is grounded in KB_DOCS
# in_kb=False → trap: answer NOT in the KB (a faithful agent should decline)
QUESTIONS = [
    {"q": "Which paper introduced the Transformer, and what is it based on?", "in_kb": True},
    {"q": "What two objectives does BERT use for pre-training?", "in_kb": True},
    {"q": "What is the central finding of the GPT-3 paper about scaling?", "in_kb": True},
    {"q": "What does FedProx add to handle non-IID data in federated learning?", "in_kb": True},
    {"q": "What does the reproducibility checklist by Pineau et al. cover?", "in_kb": True},
    {"q": "What accuracy did the Transformer achieve on ImageNet classification?", "in_kb": False},
    {"q": "Who won the Turing Award in 2018 according to your notes?", "in_kb": False},
    {"q": "What learning rate does the FedProx paper recommend for hospital data?", "in_kb": False},
]

EVAL_COLLECTION = "faith_eval_tmp"
EVAL_PERSIST_DIR = "./chroma_faith_eval"


def _agg(rows, key):
    vals = [r["verdict"][key] for r in rows if r.get("verdict")]
    return sum(vals) / len(vals) if vals else float("nan")


def _report(rows, label):
    in_kb = [r for r in rows if r["in_kb"]]
    traps = [r for r in rows if not r["in_kb"]]
    print(f"\n{label}")
    print(f"{'':22}{'faithful':>10}{'relevancy':>11}{'halluc.':>10}")
    for name, subset in [("Answerable (in KB)", in_kb),
                         ("Trap (not in KB)", traps),
                         ("Overall", rows)]:
        if subset:
            print(f"{name:22}{_agg(subset,'faithfulness'):>10.3f}"
                  f"{_agg(subset,'answer_relevancy'):>11.3f}"
                  f"{_agg(subset,'hallucination_rate'):>10.3f}")


def run_mode(agent, judge, k, grounded, min_similarity=None):
    rows = []
    for item in QUESTIONS:
        q = item["q"]
        context = agent.retrieve_context(q, k=k, min_similarity=min_similarity)
        abstained = (min_similarity is not None and context == "")
        answer = agent.speak(q, context, grounded=grounded)
        verdict = judge(q, context, answer, caller="faith_eval")
        rows.append({"question": q, "in_kb": item["in_kb"], "grounded": grounded,
                     "abstained": abstained, "context": context,
                     "answer": answer, "verdict": verdict})
    return rows


def calibrate_threshold(agent, k):
    """Print the best-chunk similarity for in-KB vs trap questions so the
    abstention threshold can be chosen from data, not guessed."""
    print("\nSimilarity calibration (best chunk per question):")
    in_sims, trap_sims = [], []
    for item in QUESTIONS:
        _, sims = agent.kb._vector_search_scored(item["q"], k)
        best = max(sims) if sims else 0.0
        (in_sims if item["in_kb"] else trap_sims).append(best)
        tag = "in-KB" if item["in_kb"] else "TRAP "
        print(f"  [{tag}] sim={best:.3f}  {item['q'][:55]}")
    import statistics as st
    if in_sims and trap_sims:
        # A reasonable threshold sits between the trap max and in-KB min.
        suggested = round((max(trap_sims) + min(in_sims)) / 2, 3)
        print(f"  in-KB min={min(in_sims):.3f}  trap max={max(trap_sims):.3f}"
              f"  → suggested threshold ≈ {suggested}")
        return suggested
    return 0.3


def main():
    p = argparse.ArgumentParser(description="Evaluate agent response faithfulness")
    p.add_argument("--model", default=None, help="Ollama model (default: llama3)")
    p.add_argument("--k", type=int, default=3, help="retrieval top-k")
    p.add_argument("--save", default=None, help="write full results to JSON")
    p.add_argument("--threshold", type=float, default=None,
                   help="abstention similarity cutoff (default: auto-calibrated)")
    args = p.parse_args()

    if args.model:
        import os
        os.environ["LAB_SIM_OLLAMA_MODEL"] = args.model

    from agents.base import BaseAgent
    from rag.eval import judge_faithfulness

    shutil.rmtree(EVAL_PERSIST_DIR, ignore_errors=True)
    agent = BaseAgent(agent_id=0, name="Eval", role="postdoc", experience=3,
                      personality="precise and evidence-based",
                      persist_dir=EVAL_PERSIST_DIR)
    agent.seed_knowledge(KB_DOCS)

    # Data-driven abstention threshold (or use --threshold to override).
    threshold = args.threshold if args.threshold is not None \
        else calibrate_threshold(agent, args.k)

    n = len(QUESTIONS)
    print(f"\nJudging {n} responses x 3 modes, k={args.k}...")

    ungrounded = run_mode(agent, judge_faithfulness, args.k, grounded=False)
    grounded = run_mode(agent, judge_faithfulness, args.k, grounded=True)
    abstain = run_mode(agent, judge_faithfulness, args.k, grounded=True,
                       min_similarity=threshold)

    _report(ungrounded, "=== UNGROUNDED (default speak) ===")
    _report(grounded,   "=== GROUNDED (answer only from context) ===")
    _report(abstain,    f"=== GROUNDED + ABSTENTION (threshold={threshold}) ===")

    n_abstained = sum(1 for r in abstain if r["abstained"])
    n_trap_abstained = sum(1 for r in abstain if r["abstained"] and not r["in_kb"])
    print(f"\nAbstention fired on {n_abstained}/{n} questions "
          f"({n_trap_abstained} of them traps).")

    base_h = _agg(ungrounded, "hallucination_rate")
    abst_h = _agg(abstain, "hallucination_rate")
    print(f"Overall hallucination:  ungrounded {base_h:.3f} "
          f"→ grounded {_agg(grounded,'hallucination_rate'):.3f} "
          f"→ +abstention {abst_h:.3f}  "
          f"(net {'+' if abst_h-base_h>=0 else ''}{abst_h-base_h:.3f})")

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"ungrounded": ungrounded, "grounded": grounded,
                       "abstention": abstain, "threshold": threshold},
                      f, indent=2, default=str)
        print(f"\nFull results saved to {args.save}")

    shutil.rmtree(EVAL_PERSIST_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
