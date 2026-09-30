"""Evaluation: run a policy for many episodes and measure how well the team does.

A policy function has the signature

    policy_fn(agent, observation, env) -> action

where `agent` is a name such as "agent_0", `observation` is that agent's
observation (a numpy array) and the action is an int from 0 to 4.
"""

import json
from pathlib import Path

import numpy as np

from marl.envs import make_env

# Seeds used for evaluation episodes. They are different from training seeds.
EVAL_SEED = 10_000

METRICS = {
    "team_return": "Team return (higher is better)",
    "final_distance": "Final landmark distance (lower is better)",
    "collisions": "Collisions per episode (lower is better)",
    "landmarks_covered": "Landmarks covered at the end (higher is better)",
}


def run_episode(policy_fn, env, seed):
    """Play one episode and return its metrics as a dict.

    team_return: sum over time of the average reward of the agents.
    final_distance: at the last step, for each landmark the distance to the
        closest agent, summed over landmarks.
    collisions: number of colliding agent pairs, summed over all steps.
    landmarks_covered: landmarks with an agent closer than 0.1 at the last step.
    """
    observations, _ = env.reset(seed=seed)
    for i, agent in enumerate(env.agents):
        env.action_space(agent).seed(seed + i)  # makes random policies repeatable
    team_return, collisions = 0.0, 0
    final_distance, landmarks_covered = float("nan"), 0
    while env.agents:
        actions = {
            agent: policy_fn(agent, observations[agent], env) for agent in env.agents
        }
        observations, rewards, _, _, infos = env.step(actions)
        team_return += float(np.mean(list(rewards.values())))
        # benchmark_data = (reward, collisions, sum of min distances, landmarks covered)
        data = [info["benchmark_data"] for info in infos.values()]
        # Each agent counts its own collisions, so every colliding pair is counted twice.
        collisions += sum(int(d[1]) for d in data) // 2
        final_distance, landmarks_covered = float(data[0][2]), int(data[0][3])
    return {
        "team_return": team_return,
        "final_distance": final_distance,
        "collisions": collisions,
        "landmarks_covered": landmarks_covered,
    }


def evaluate(policy_fn, env_kwargs, n_episodes=100, seed=EVAL_SEED):
    """Run `n_episodes` episodes and return the mean and std of every metric."""
    env = make_env(benchmark_data=True, **env_kwargs)
    episodes = [run_episode(policy_fn, env, seed + i) for i in range(n_episodes)]
    env.close()
    summary = {"n_episodes": n_episodes}
    for name in METRICS:
        values = np.array([episode[name] for episode in episodes], dtype=float)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std())
    return summary


def format_table(summaries):
    """Format {label: summary} as a small text table."""
    width = max(len(label) for label in summaries) + 2
    header = "".ljust(width) + "".join(name.rjust(22) for name in METRICS)
    lines = [header]
    for label, summary in summaries.items():
        cells = "".join(
            f"{summary[f'{name}_mean']:.2f} ± {summary[f'{name}_std']:.2f}".rjust(22)
            for name in METRICS
        )
        lines.append(label.ljust(width) + cells)
    return "\n".join(lines)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def render_episode(policy_fn, env_kwargs, seed=0, gif_path=None, fps=10):
    """Watch one episode in a window, or save it as a GIF if `gif_path` is given.

    Use a GIF on Google Colab or any machine without a display.
    """
    render_mode = "rgb_array" if gif_path else "human"
    env = make_env(render_mode=render_mode, **env_kwargs)
    observations, _ = env.reset(seed=seed)
    frames = [env.render()] if gif_path else []
    while env.agents:
        actions = {
            agent: policy_fn(agent, observations[agent], env) for agent in env.agents
        }
        observations, _, _, _, _ = env.step(actions)
        if gif_path:
            frames.append(env.render())
    env.close()
    if gif_path:
        from PIL import Image

        images = [Image.fromarray(frame) for frame in frames]
        Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"Saved {gif_path}")
