"""Plot learning curves: the mean over seeds, with a band of ± one standard deviation.

Examples:
    python plot_results.py --group "shared PPO" "results/ppo_N3_lr0.5_seed*" \
        --random results/random_N3_lr0.5.json --out results/learning_curve.png

    # Compare two settings on one figure:
    python plot_results.py \
        --group "local_ratio 0.5" "results/ppo_N3_lr0.5_seed*" \
        --group "local_ratio 0.0" "results/ppo_N3_lr0.0_seed*" \
        --metric collisions --out results/local_ratio_collisions.png
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files; no window needed
import matplotlib.pyplot as plt
import numpy as np

from marl.metrics import METRICS


def load_runs(pattern, metric):
    """Return (timesteps, values) where values has one row per run (seed)."""
    run_dirs = sorted(glob.glob(pattern))
    curves = []
    for run_dir in run_dirs:
        with open(Path(run_dir) / "progress.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        steps = [int(row["timesteps"]) for row in rows]
        values = [float(row[f"{metric}_mean"]) for row in rows]
        curves.append((steps, values))
    if not curves:
        raise SystemExit(f"No runs match {pattern!r}")
    length = min(len(values) for _, values in curves)  # in case a run stopped early
    steps = np.array(curves[0][0][:length])
    values = np.array([values[:length] for _, values in curves])
    return steps, values


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--group",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATTERN"),
        required=True,
        help="a label and a pattern matching run folders (one per seed)",
    )
    parser.add_argument("--metric", choices=list(METRICS), default="team_return")
    parser.add_argument("--random", help="random-policy JSON written by explore.py")
    parser.add_argument("--out", default="results/learning_curve.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, pattern in args.group:
        steps, values = load_runs(pattern, args.metric)
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(steps, mean, label=f"{label} ({len(values)} seeds)")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    if args.random:
        random_value = json.loads(Path(args.random).read_text())[f"{args.metric}_mean"]
        ax.axhline(random_value, color="gray", linestyle="--", label="random policy")

    ax.set_xlabel("Timesteps (agent steps)")
    ax.set_ylabel(METRICS[args.metric])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
