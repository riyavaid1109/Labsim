#!/usr/bin/env python
"""
plot_experiment.py — averaged, error-banded comparison of conditions.

Loads every seed's history JSON per condition and produces figures where
each curve is the mean across seeds and the shaded band is ±1 standard
deviation (or SEM with --sem). This is the difference between "here's one
run" and "here's a robust effect across n seeds."

Two figure types:

1. Lab-level metrics over time (one axis each), fair vs biased overlaid:
     - mean agent motivation
     - reputation spread (max - min: how concentrated is credit?)
     - average review score
     - credit dispute rate

2. Per-role motivation over time, faceted by role, conditions overlaid —
   shows e.g. whether PI morale diverges from students under bias.

Usage:
    python plot_experiment.py                      # uses experiments/manifest.json
    python plot_experiment.py --sem                # SEM bands instead of SD
    python plot_experiment.py --fair a.json b.json --biased c.json d.json
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND_COLORS = {"fair": "#2a9d8f", "biased": "#e76f51"}
DEFAULT_COLORS = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a"]


# ── Loading ────────────────────────────────────────────────────────────────

def load_history(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_condition(paths: list[str]) -> list[list[dict]]:
    """Return a list of histories (one per seed) for a condition."""
    runs = []
    for p in paths:
        if os.path.exists(p):
            runs.append(load_history(p))
        else:
            print(f"  (missing: {p})")
    return runs


# ── Metric extraction ──────────────────────────────────────────────────────

def _agent_states(step: dict) -> list[dict]:
    return step.get("metrics", {}).get("agent_states", [])


def series_mean_motivation(history: list[dict]) -> list[float]:
    return [float(np.mean([a["motivation"] for a in _agent_states(s)]))
            for s in history]


def series_reputation_spread(history: list[dict]) -> list[float]:
    out = []
    for s in history:
        reps = [a["reputation"] for a in _agent_states(s)]
        out.append(max(reps) - min(reps) if reps else 0.0)
    return out


def series_avg_score(history: list[dict]) -> list[float]:
    return [float(s.get("avg_review_score",
                        s.get("metrics", {}).get("avg_review_score", 0.0)))
            for s in history]


def series_dispute_rate(history: list[dict]) -> list[float]:
    return [float(s.get("metrics", {}).get("credit_dispute_rate", 0.0))
            for s in history]


def series_motivation_by_role(history: list[dict]) -> dict[str, list[float]]:
    """Mean motivation per role at each timestep."""
    roles = defaultdict(list)
    for s in history:
        by_role = defaultdict(list)
        for a in _agent_states(s):
            by_role[a["role"]].append(a["motivation"])
        for role, vals in by_role.items():
            roles[role].append(float(np.mean(vals)))
    return roles


# ── Aggregation across seeds ───────────────────────────────────────────────

def stack_runs(runs: list[list[dict]], extractor) -> np.ndarray:
    """Apply a per-history extractor to each seed, align to the shortest
    length, and stack into a (n_seeds, n_timesteps) array."""
    series = [extractor(r) for r in runs if r]
    series = [s for s in series if s]
    if not series:
        return np.empty((0, 0))
    min_len = min(len(s) for s in series)
    return np.array([s[:min_len] for s in series], dtype=float)


def mean_band(arr: np.ndarray, use_sem: bool):
    """Return (x, mean, lo, hi) for a (n_seeds, T) array."""
    if arr.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    err = sd / np.sqrt(arr.shape[0]) if use_sem else sd
    x = np.arange(1, len(mean) + 1)
    return x, mean, mean - err, mean + err


def _plot_band(ax, arr, label, color, use_sem):
    x, mean, lo, hi = mean_band(arr, use_sem)
    if len(x) == 0:
        return
    ax.plot(x, mean, color=color, label=label, linewidth=2, marker="o", markersize=4)
    if arr.shape[0] > 1:
        ax.fill_between(x, lo, hi, color=color, alpha=0.2)


# ── Figures ────────────────────────────────────────────────────────────────

def figure_lab_metrics(cond_runs: dict[str, list], use_sem: bool, out_path: str):
    band = "SEM" if use_sem else "SD"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    n = {c: len(r) for c, r in cond_runs.items()}
    fig.suptitle(f"Lab metrics: fair vs biased  (mean ± {band}, "
                 f"n={n})", fontsize=13)

    panels = [
        (axes[0, 0], series_mean_motivation, "Mean agent motivation", "motivation (0–10)"),
        (axes[0, 1], series_reputation_spread, "Reputation spread (max − min)", "spread"),
        (axes[1, 0], series_avg_score, "Average review score", "score (1–5)"),
        (axes[1, 1], series_dispute_rate, "Credit dispute rate", "rate"),
    ]
    for ax, extractor, title, ylabel in panels:
        for cond, runs in cond_runs.items():
            arr = stack_runs(runs, extractor)
            _plot_band(ax, arr, cond, COND_COLORS.get(cond), use_sem)
        ax.set_title(title)
        ax.set_xlabel("timestep")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}")


def figure_motivation_by_role(cond_runs: dict[str, list], use_sem: bool,
                              out_path: str):
    # Discover the union of roles present.
    roles = set()
    for runs in cond_runs.values():
        for r in runs:
            for s in r:
                for a in _agent_states(s):
                    roles.add(a["role"])
    roles = sorted(roles)
    if not roles:
        print("  (no roles found — skipping role figure)")
        return

    fig, axes = plt.subplots(1, len(roles), figsize=(5 * len(roles), 4.5),
                             squeeze=False)
    band = "SEM" if use_sem else "SD"
    fig.suptitle(f"Motivation by role  (mean ± {band})", fontsize=13)

    for ax, role in zip(axes[0], roles):
        for cond, runs in cond_runs.items():
            arr = stack_runs(runs, lambda h, rl=role:
                             series_motivation_by_role(h).get(rl, []))
            _plot_band(ax, arr, cond, COND_COLORS.get(cond), use_sem)
        ax.set_title(role)
        ax.set_xlabel("timestep")
        ax.set_ylabel("motivation (0–10)")
        ax.set_ylim(0, 10)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    print(f"  wrote {out_path}")


# ── Entry ──────────────────────────────────────────────────────────────────

def resolve_conditions(args) -> dict[str, list[str]]:
    # Explicit paths take precedence over the manifest.
    explicit = {}
    if args.fair:
        explicit["fair"] = args.fair
    if args.biased:
        explicit["biased"] = args.biased
    if explicit:
        return explicit

    if not os.path.exists(args.manifest):
        raise SystemExit(
            f"No manifest at {args.manifest} and no explicit --fair/--biased "
            "paths given. Run run_experiment.py first, or pass paths."
        )
    with open(args.manifest) as f:
        return json.load(f)["conditions"]


def main():
    p = argparse.ArgumentParser(description="Plot averaged experiment results")
    p.add_argument("--manifest", default=os.path.join("experiments", "manifest.json"))
    p.add_argument("--fair", nargs="*", help="Explicit fair-condition JSON paths")
    p.add_argument("--biased", nargs="*", help="Explicit biased-condition JSON paths")
    p.add_argument("--sem", action="store_true",
                   help="Use standard error bands instead of standard deviation")
    p.add_argument("--outdir", default="experiments")
    args = p.parse_args()

    cond_paths = resolve_conditions(args)
    cond_runs = {c: load_condition(paths) for c, paths in cond_paths.items()}
    cond_runs = {c: r for c, r in cond_runs.items() if r}  # drop empty
    if not cond_runs:
        raise SystemExit("No runs loaded — check your JSON paths.")

    for c, r in cond_runs.items():
        print(f"{c}: {len(r)} seed(s) loaded")

    os.makedirs(args.outdir, exist_ok=True)
    figure_lab_metrics(cond_runs, args.sem,
                       os.path.join(args.outdir, "compare_lab_metrics.png"))
    figure_motivation_by_role(cond_runs, args.sem,
                              os.path.join(args.outdir, "compare_motivation_by_role.png"))
    print("Done.")


if __name__ == "__main__":
    main()
