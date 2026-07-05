# #!/usr/bin/env python
# """
# run_experiment.py — run the simulation across multiple seeds per condition.

# A single simulation run is an anecdote: llama3 sampling and random agent
# init make any one trajectory noisy. This runner executes N seeds for each
# experimental condition and writes every run's history to disk, so the
# plotter can average across seeds and draw error bands.

# Each run is a fresh subprocess calling run_sim.py — this guarantees a clean
# ChromaDB and RNG state per seed (the simulation persists embeddings to
# ./chroma_db, so runs must not share a store). Outputs land in:

#     experiments/<condition>/seed_<n>.json

# Usage:
#     # Two default conditions (fair vs biased), 3 seeds each, 3 timesteps:
#     python run_experiment.py --seeds 3 --timesteps 3

#     # Custom: more seeds, more timesteps, a faster model
#     python run_experiment.py --seeds 5 --timesteps 5 --model llama3.2

# Runtime warning: on an M1 with llama3, one 5-timestep run is ~25-30 min,
# so --seeds 3 --timesteps 5 across two conditions is several hours. Start
# small (--timesteps 2) to validate the pipeline, then scale up.
# """

# import os
# import sys
# import json
# import shutil
# import argparse
# import subprocess

# # Each condition is a set of extra CLI flags passed to run_sim.py.
# DEFAULT_CONDITIONS = {
#     "fair":   ["--fairness", "1.0"],
#     "biased": ["--fairness", "0.2", "--pi-favorite", "3"],
# }

# OUT_ROOT = "experiments"


# def run_one(condition: str, flags: list[str], seed: int, timesteps: int,
#             students: int, postdocs: int, model: str | None,
#             python: str) -> str | None:
#     """Run a single simulation as a subprocess. Returns the JSON path on
#     success, None on failure."""
#     out_dir = os.path.join(OUT_ROOT, condition)
#     os.makedirs(out_dir, exist_ok=True)
#     out_json = os.path.join(out_dir, f"seed_{seed}.json")

#     # Isolate the vector store per run so seeds don't contaminate each other.
#     chroma_dir = os.path.join(OUT_ROOT, condition, f"chroma_seed_{seed}")
#     shutil.rmtree(chroma_dir, ignore_errors=True)

#     cmd = [
#         python, "run_sim.py",
#         "--students", str(students),
#         "--postdocs", str(postdocs),
#         "--timesteps", str(timesteps),
#         "--seed", str(seed),
#         "--persist-dir", chroma_dir,
#         "--save-json", out_json,
#         "--trace-db", os.path.join(OUT_ROOT, condition, f"traces_seed_{seed}.db"),
#         *flags,
#     ]
#     env = dict(os.environ)
#     if model:
#         env["LAB_SIM_OLLAMA_MODEL"] = model

#     print(f"\n=== {condition} | seed {seed} ===\n  {' '.join(cmd)}")
#     try:
#         # matplotlib plotting at the end of run_sim needs a non-interactive
#         # backend in a headless subprocess.
#         env.setdefault("MPLBACKEND", "Agg")
#         subprocess.run(cmd, check=True, env=env)
#     except subprocess.CalledProcessError as e:
#         print(f"  !! run failed ({condition} seed {seed}): {e}", file=sys.stderr)
#         return None
#     finally:
#         # The per-run chroma store is large and disposable; drop it once the
#         # history JSON is written.
#         shutil.rmtree(chroma_dir, ignore_errors=True)

#     return out_json if os.path.exists(out_json) else None


# def main():
#     p = argparse.ArgumentParser(description="Multi-seed experiment runner")
#     p.add_argument("--seeds", type=int, default=3, help="Seeds per condition")
#     p.add_argument("--seed-start", type=int, default=0, help="First seed value")
#     p.add_argument("--timesteps", type=int, default=3)
#     p.add_argument("--students", type=int, default=3)
#     p.add_argument("--postdocs", type=int, default=2)
#     p.add_argument("--model", type=str, default=None,
#                    help="Ollama model override (e.g. llama3.2 for speed)")
#     p.add_argument("--conditions", type=str, default=None,
#                    help="Comma-separated subset of conditions to run "
#                         f"(default: {','.join(DEFAULT_CONDITIONS)})")
#     args = p.parse_args()

#     conditions = DEFAULT_CONDITIONS
#     if args.conditions:
#         wanted = {c.strip() for c in args.conditions.split(",")}
#         conditions = {k: v for k, v in DEFAULT_CONDITIONS.items() if k in wanted}
#         if not conditions:
#             sys.exit(f"No known conditions in {args.conditions}")

#     python = sys.executable  # use the same interpreter (respects the venv)
#     seeds = list(range(args.seed_start, args.seed_start + args.seeds))

#     manifest: dict[str, list[str]] = {}
#     for condition, flags in conditions.items():
#         paths = []
#         for seed in seeds:
#             path = run_one(condition, flags, seed, args.timesteps,
#                            args.students, args.postdocs, args.model, python)
#             if path:
#                 paths.append(path)
#         manifest[condition] = paths

#     os.makedirs(OUT_ROOT, exist_ok=True)
#     manifest_path = os.path.join(OUT_ROOT, "manifest.json")
#     with open(manifest_path, "w") as f:
#         json.dump({"conditions": manifest, "seeds": seeds,
#                    "timesteps": args.timesteps}, f, indent=2)

#     print(f"\nDone. {sum(len(v) for v in manifest.values())} runs written.")
#     print(f"Manifest: {manifest_path}")
#     print("Now plot with:  python plot_experiment.py")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python
"""
run_experiment.py — run the simulation across multiple seeds per condition.

A single simulation run is an anecdote: llama3 sampling and random agent
init make any one trajectory noisy. This runner executes N seeds for each
experimental condition and writes every run's history to disk, so the
plotter can average across seeds and draw error bands.

Each run is a fresh subprocess calling run_sim.py — this guarantees a clean
ChromaDB and RNG state per seed (the simulation persists embeddings to
./chroma_db, so runs must not share a store). Outputs land in:

    experiments/<condition>/seed_<n>.json

Usage:
    # Two default conditions (fair vs biased), 3 seeds each, 3 timesteps:
    python run_experiment.py --seeds 3 --timesteps 3

    # Custom: more seeds, more timesteps, a faster model
    python run_experiment.py --seeds 5 --timesteps 5 --model llama3.2

Runtime warning: on an M1 with llama3, one 5-timestep run is ~25-30 min,
so --seeds 3 --timesteps 5 across two conditions is several hours. Start
small (--timesteps 2) to validate the pipeline, then scale up.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess

# Each condition is a set of extra CLI flags passed to run_sim.py.
DEFAULT_CONDITIONS = {
    "fair":   ["--fairness", "1.0"],
    # Favor agent 0 (a low-experience PhD student) — favoring a weak agent
    # maximizes the contrast with merit-based fair assignment.
    "biased": ["--fairness", "0.2", "--pi-favorite", "0"],
}

OUT_ROOT = "experiments"


def run_one(condition: str, flags: list[str], seed: int, timesteps: int,
            students: int, postdocs: int, model: str | None,
            python: str) -> str | None:
    """Run a single simulation as a subprocess. Returns the JSON path on
    success, None on failure."""
    out_dir = os.path.join(OUT_ROOT, condition)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"seed_{seed}.json")

    # Isolate the vector store per run so seeds don't contaminate each other.
    chroma_dir = os.path.join(OUT_ROOT, condition, f"chroma_seed_{seed}")
    shutil.rmtree(chroma_dir, ignore_errors=True)

    cmd = [
        python, "run_sim.py",
        "--students", str(students),
        "--postdocs", str(postdocs),
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--persist-dir", chroma_dir,
        "--save-json", out_json,
        "--trace-db", os.path.join(OUT_ROOT, condition, f"traces_seed_{seed}.db"),
        *flags,
    ]
    env = dict(os.environ)
    if model:
        env["LAB_SIM_OLLAMA_MODEL"] = model

    print(f"\n=== {condition} | seed {seed} ===\n  {' '.join(cmd)}")
    try:
        # matplotlib plotting at the end of run_sim needs a non-interactive
        # backend in a headless subprocess.
        env.setdefault("MPLBACKEND", "Agg")
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"  !! run failed ({condition} seed {seed}): {e}", file=sys.stderr)
        return None
    finally:
        # The per-run chroma store is large and disposable; drop it once the
        # history JSON is written.
        shutil.rmtree(chroma_dir, ignore_errors=True)

    return out_json if os.path.exists(out_json) else None


def main():
    p = argparse.ArgumentParser(description="Multi-seed experiment runner")
    p.add_argument("--seeds", type=int, default=3, help="Seeds per condition")
    p.add_argument("--seed-start", type=int, default=0, help="First seed value")
    p.add_argument("--timesteps", type=int, default=3)
    p.add_argument("--students", type=int, default=3)
    p.add_argument("--postdocs", type=int, default=2)
    p.add_argument("--model", type=str, default=None,
                   help="Ollama model override (e.g. llama3.2 for speed)")
    p.add_argument("--conditions", type=str, default=None,
                   help="Comma-separated subset of conditions to run "
                        f"(default: {','.join(DEFAULT_CONDITIONS)})")
    args = p.parse_args()

    conditions = DEFAULT_CONDITIONS
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = {k: v for k, v in DEFAULT_CONDITIONS.items() if k in wanted}
        if not conditions:
            sys.exit(f"No known conditions in {args.conditions}")

    python = sys.executable  # use the same interpreter (respects the venv)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    manifest: dict[str, list[str]] = {}
    for condition, flags in conditions.items():
        paths = []
        for seed in seeds:
            path = run_one(condition, flags, seed, args.timesteps,
                           args.students, args.postdocs, args.model, python)
            if path:
                paths.append(path)
        manifest[condition] = paths

    os.makedirs(OUT_ROOT, exist_ok=True)
    manifest_path = os.path.join(OUT_ROOT, "manifest.json")
    # Merge with any existing manifest so running one condition at a time
    # accumulates rather than clobbering the other condition's entry.
    existing = {"conditions": {}, "seeds": seeds, "timesteps": args.timesteps}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                existing = json.load(f)
            existing.setdefault("conditions", {})
        except (json.JSONDecodeError, OSError):
            existing = {"conditions": {}, "seeds": seeds, "timesteps": args.timesteps}
    existing["conditions"].update(manifest)
    existing["seeds"] = seeds
    existing["timesteps"] = args.timesteps
    with open(manifest_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nDone. {sum(len(v) for v in manifest.values())} runs written.")
    print(f"Manifest: {manifest_path}")
    print("Now plot with:  python plot_experiment.py")


if __name__ == "__main__":
    main()