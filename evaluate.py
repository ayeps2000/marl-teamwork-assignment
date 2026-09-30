"""Compare a trained policy with the random policy over 100 episodes.

Examples:
    python evaluate.py results/ppo_N3_lr0.5_seed0
    python evaluate.py results/ppo_N3_lr0.5_seed0 --render            # watch it
    python evaluate.py results/ppo_N3_lr0.5_seed0 --gif results/ppo.gif
"""

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from explore import random_policy
from marl.metrics import evaluate, format_table, render_episode
from train import make_policy_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="a folder written by train.py")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="watch one episode")
    parser.add_argument("--gif", help="save one episode to this GIF file")
    parser.add_argument("--seed", type=int, default=0, help="episode to render")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    env_kwargs = json.loads((run_dir / "config.json").read_text())["env_kwargs"]
    model = PPO.load(run_dir / "model.zip")
    policy_fn = make_policy_fn(model)

    summaries = {
        "random": evaluate(random_policy, env_kwargs, n_episodes=args.episodes),
        run_dir.name: evaluate(policy_fn, env_kwargs, n_episodes=args.episodes),
    }
    print(f"Environment: {env_kwargs}, {args.episodes} episodes (mean ± std)\n")
    print(format_table(summaries))

    if args.render or args.gif:
        render_episode(policy_fn, env_kwargs, seed=args.seed, gif_path=args.gif)


if __name__ == "__main__":
    main()
