"""A Stable-Baselines3 callback that evaluates the policy during training.

It writes one row to progress.csv every `eval_every` timesteps. The learning
curve plotted by plot_results.py comes from this file.
"""

import csv
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from marl.metrics import METRICS, evaluate


class EvalLogger(BaseCallback):
    def __init__(self, make_policy_fn, env_kwargs, csv_path, eval_every, n_episodes=20):
        super().__init__()
        self.make_policy_fn = make_policy_fn
        self.env_kwargs = env_kwargs
        self.csv_path = Path(csv_path)
        self.eval_every = eval_every
        self.n_episodes = n_episodes
        self.last_eval = None
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["timesteps", "wall_time"]
        for name in METRICS:
            columns += [f"{name}_mean", f"{name}_std"]
        with self.csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(columns)
        self._evaluate()

    def _on_step(self):
        if self.num_timesteps >= self.last_eval + self.eval_every:
            self._evaluate()
        return True

    def _on_training_end(self):
        # Always log the final policy.
        if self.num_timesteps > self.last_eval:
            self._evaluate()

    def _evaluate(self):
        policy_fn = self.make_policy_fn(self.model)
        summary = evaluate(policy_fn, self.env_kwargs, n_episodes=self.n_episodes)
        row = [self.num_timesteps, round(time.time() - self.start_time, 1)]
        for name in METRICS:
            row += [summary[f"{name}_mean"], summary[f"{name}_std"]]
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        print(
            f"  timesteps {self.num_timesteps:>9,d} | "
            f"team return {summary['team_return_mean']:7.2f} | "
            f"collisions {summary['collisions_mean']:.2f}",
            flush=True,
        )
        self.last_eval = self.num_timesteps
