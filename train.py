"""Task 2: train one PPO policy that is shared by all agents.

Examples:
    python train.py --seeds 0 1 2                 # three runs with different seeds
    python train.py --seeds 0 --timesteps 100000  # quick test run
    python train.py --seeds 0 1 2 --n-agents 4    # Task 3 example: 4 agents

Each run writes to results/<run name>/:
    progress.csv     evaluation during training (the learning curve)
    final_eval.json  evaluation of the final policy over 100 episodes
    model.zip        the trained policy
    config.json      the settings used

"timesteps" counts agent steps: one step of the game with 3 agents is 3 timesteps.
"""

import argparse
import json
from pathlib import Path

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from marl.callbacks import EvalLogger
from marl.envs import make_env, seed_vec_env
from marl.metrics import evaluate, format_table, save_json


def make_training_env(env_kwargs, n_envs):
    """Wrap the multi-agent game so Stable-Baselines3 can train on it."""
    env = make_env(**env_kwargs)
    # TODO(Task 2a): turn `env` into a vector environment for Stable-Baselines3.
    #   1. ss.pettingzoo_env_to_vec_env_v1(env) turns every agent into one "slot"
    #      of a vector environment.
    #   2. ss.concat_vec_envs_v1(..., n_envs, num_cpus=0, base_class="stable_baselines3")
    #      runs n_envs copies of the game side by side.
    # Question for your report: why does this give ONE policy shared by all agents?
    raise NotImplementedError("TODO(Task 2a): see the comments above")


def make_model(venv, args):
    """Create the PPO learner."""
    # TODO(Task 2b): return a PPO model with the "MlpPolicy" and the
    # hyperparameters in `args` (learning_rate, n_steps, batch_size, gamma, ent_coef).
    #   Docs: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    #   Do not pass seed=... (it fails with SuperSuit); main() seeds everything.
    raise NotImplementedError("TODO(Task 2b): see the comments above")


def make_policy_fn(model):
    """Return a function policy_fn(agent, observation, env) -> action."""
    # TODO(Task 2c): use the trained model to choose an action.
    #   model.predict(observation, deterministic=True) returns (action, state).
    #   The environment expects a plain int.
    raise NotImplementedError("TODO(Task 2c): see the comments above")


def train_one(args, seed):
    env_kwargs = {"N": args.n_agents, "local_ratio": args.local_ratio}
    run_name = args.run_name or f"ppo_N{args.n_agents}_lr{args.local_ratio}"
    run_dir = Path("results") / f"{run_name}_seed{seed}"
    print(f"\n=== {run_dir} ===")

    set_random_seed(seed)
    venv = make_training_env(env_kwargs, args.n_envs)
    seed_vec_env(venv, seed)
    model = make_model(venv, args)
    logger = EvalLogger(
        make_policy_fn, env_kwargs, run_dir / "progress.csv", args.eval_every
    )
    model.learn(total_timesteps=args.timesteps, callback=logger)
    venv.close()

    model.save(run_dir / "model.zip")
    config = {"seed": seed, "env_kwargs": env_kwargs, **vars(args)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    summary = evaluate(make_policy_fn(model), env_kwargs, n_episodes=100)
    save_json(summary, run_dir / "final_eval.json")
    print(format_table({"final policy": summary}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--run-name", help="default: ppo_N<agents>_lr<local ratio>")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--local-ratio", type=float, default=0.5)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    args = parser.parse_args()
    for seed in args.seeds:
        train_one(args, seed)


if __name__ == "__main__":
    main()
