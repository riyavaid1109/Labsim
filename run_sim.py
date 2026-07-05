"""
run_sim.py — entry point for the lab simulation.

Usage:
    python run_sim.py --students 3 --postdocs 2 --timesteps 5 --fairness 0.4
    python run_sim.py --students 3 --postdocs 2 --timesteps 5 --fairness 1.0

Vary --fairness between runs to study how PI bias affects outcomes.
"""

import os
import argparse
import logging
import random
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from agents.phd_student import PhDStudent
from agents.postdoc import Postdoc
from agents.pi import PI
from simulation.lab import Lab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("lab_sim.log"),
    ],
)

SAMPLE_PAPERS = [
    "Transformer architectures for low-resource NLP tasks. "
    "Key challenge: limited labeled data in target domain.",
    "Improving reproducibility in deep learning experiments. "
    "Focus: random seed sensitivity and hardware variability.",
    "Cross-lingual transfer learning for biomedical NER. "
    "Dataset: PubMed abstracts in 5 languages.",
    "Evaluation bias in LLM benchmarks. "
    "Hypothesis: popular benchmarks reward memorisation over reasoning.",
    "Federated learning for privacy-preserving clinical NLP. "
    "Challenge: non-IID data across hospital sites.",
]

STUDENT_NAMES = ["Aiko", "Ben", "Camille", "Dev", "Elena"]
POSTDOC_NAMES = ["Farah", "Giulio", "Hana"]


def seed_agents_with_knowledge(students, postdocs, pi):
    """Give each agent a starting knowledge base relevant to their role."""
    # Shared background all agents have read
    shared_texts = [
        "Attention Is All You Need (Vaswani et al., 2017): introduced the Transformer architecture.",
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019).",
        "Language Models are Few-Shot Learners (Brown et al., 2020): GPT-3 paper.",
    ]

    # Students get a smaller, more specific set
    student_extras = [
        "Survey of data augmentation techniques for low-resource NLP.",
        "Practical guide to fine-tuning pre-trained models on domain-specific data.",
    ]

    # Postdocs get the broadest KB
    postdoc_extras = [
        "A call for reproducibility in machine learning (Pineau et al., 2021).",
        "Measurement invariance and benchmarking pitfalls in NLP evaluation.",
        "Cross-lingual representational alignment: methods and limitations.",
        "Federated optimisation in heterogeneous networks (Li et al., 2020).",
    ]

    # PI has strategic / grant-relevant knowledge
    pi_extras = [
        "NSF 2024 priorities: trustworthy AI, reproducibility, and responsible data use.",
        "h-index and citation impact: metrics and their limitations.",
        "Building interdisciplinary research teams: lessons from DARPA programs.",
    ]

    for s in students:
        s.seed_knowledge(shared_texts + student_extras)
    for p in postdocs:
        p.seed_knowledge(shared_texts + postdoc_extras)
    pi.seed_knowledge(shared_texts + pi_extras)


def plot_results(history: list[dict]):
    """Matplotlib figure showing motivation, reputation, and review scores over time."""
    if not history or all(h.get("skipped") for h in history):
        print("No data to plot.")
        return

    timesteps = [h["timestep"] for h in history if not h.get("skipped")]
    decisions = [h["review_decision"] for h in history if not h.get("skipped")]
    scores = [h["avg_review_score"] for h in history if not h.get("skipped")]

    # Collect per-agent time series
    agent_names = [s["name"] for s in history[0]["metrics"]["agent_states"]]
    motivation_series = {n: [] for n in agent_names}
    reputation_series = {n: [] for n in agent_names}

    for h in history:
        if h.get("skipped"):
            continue
        for s in h["metrics"]["agent_states"]:
            motivation_series[s["name"]].append(s["motivation"])
            reputation_series[s["name"]].append(s["reputation"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

    # Motivation
    ax1 = axes[0]
    for name, values in motivation_series.items():
        ax1.plot(timesteps, values, label=name, marker="o", markersize=4)
    ax1.set_title("Agent motivation over time")
    ax1.set_ylabel("Motivation (0–10)")
    ax1.set_ylim(0, 11)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    # Reputation
    ax2 = axes[1]
    for name, values in reputation_series.items():
        ax2.plot(timesteps, values, label=name, marker="s", markersize=4)
    ax2.set_title("Agent reputation over time")
    ax2.set_ylabel("Reputation")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    # Review scores + decisions
    ax3 = axes[2]
    colors = {"ACCEPT": "green", "REVISE": "orange", "REJECT": "red", None: "gray"}
    bar_colors = [colors.get(d, "gray") for d in decisions]
    ax3.bar(timesteps, scores, color=bar_colors, alpha=0.7)
    ax3.set_title("Average review score per timestep (green=accept, orange=revise, red=reject)")
    ax3.set_ylabel("Score (1–5)")
    ax3.set_ylim(0, 5.5)
    ax3.set_xlabel("Timestep")
    ax3.grid(alpha=0.3, axis="y")

    plt.savefig("lab_sim_results.png", dpi=150)
    plt.show()
    print("Plot saved to lab_sim_results.png")


def main():
    parser = argparse.ArgumentParser(description="Research Lab Simulation")
    parser.add_argument("--students", type=int, default=3, help="Number of PhD students")
    parser.add_argument("--postdocs", type=int, default=2, help="Number of postdocs")
    parser.add_argument("--timesteps", type=int, default=5, help="Number of problems to run")
    parser.add_argument("--fairness", type=float, default=0.7,
                        help="PI fairness 0.0 (plays favorites) to 1.0 (pure merit)")
    parser.add_argument("--pi-favorite", type=int, default=None,
                        help="Agent ID the PI secretly favors (optional)")
    parser.add_argument("--problems-dir", type=str, default="./problems")
    parser.add_argument("--persist-dir", type=str, default="./chroma_db")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Path to save simulation history as JSON")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["ollama", "openai", "anthropic"],
                        help="LLM backend (default: ollama, or LAB_SIM_PROVIDER env var)")
    parser.add_argument("--trace-db", type=str, default="./llm_traces.db",
                        help="SQLite file for LLM call traces ('' to disable)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible agent init / RNG")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        try:
            import numpy as np
            np.random.seed(args.seed)
        except ImportError:
            pass

    # Configure LLM backend + observability before any agent is created
    if args.provider:
        os.environ["LAB_SIM_PROVIDER"] = args.provider
    from llm import observability
    observability.configure(db_path=args.trace_db or "./llm_traces.db",
                            enabled=bool(args.trace_db))

    # Build agents
    n_students = min(args.students, len(STUDENT_NAMES))
    n_postdocs = min(args.postdocs, len(POSTDOC_NAMES))

    students = [
        PhDStudent(agent_id=i, name=STUDENT_NAMES[i])
        for i in range(n_students)
    ]
    postdocs = [
        Postdoc(agent_id=n_students + i, name=POSTDOC_NAMES[i])
        for i in range(n_postdocs)
    ]
    pi = PI(
        agent_id=n_students + n_postdocs,
        name="Prof. Chen",
        fairness=args.fairness,
    )

    # Optionally set a PI favorite
    if args.pi_favorite is not None:
        pi.add_favorite(args.pi_favorite)
        print(f"PI secretly favors agent ID {args.pi_favorite}")

    all_agents = students + postdocs + [pi]
    print(f"\nAgents: {[str(a) for a in all_agents]}")
    print(f"PI fairness: {pi.fairness}\n")

    # Seed knowledge bases
    seed_agents_with_knowledge(students, postdocs, pi)

    # If no problem files exist, create sample ones
    import os
    os.makedirs(args.problems_dir, exist_ok=True)
    if not os.listdir(args.problems_dir):
        print("No problem files found — creating sample problems...")
        for i, text in enumerate(SAMPLE_PAPERS[:args.timesteps], 1):
            path = os.path.join(args.problems_dir, f"problem_{i:02d}.md")
            with open(path, "w") as f:
                f.write(f"# Research Problem {i}\n\n{text}\n")
        print(f"Created {min(args.timesteps, len(SAMPLE_PAPERS))} sample problems in {args.problems_dir}/\n")

    # Run simulation
    lab = Lab(
        phd_students=students,
        postdocs=postdocs,
        pi=pi,
        problems_dir=args.problems_dir,
        persist_dir=args.persist_dir,
    )
    history = lab.run(n_timesteps=args.timesteps)

    # Save JSON if requested
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(history, f, indent=2, default=str)
        print(f"History saved to {args.save_json}")

    # Plot
    plot_results(history)

    # LLM observability summary
    if args.trace_db:
        print("\nLLM call summary:")
        print(observability.trace_summary())
        print(f"Full traces in {args.trace_db} "
              "(export: llm.observability.export_traces())")


if __name__ == "__main__":
    main()
