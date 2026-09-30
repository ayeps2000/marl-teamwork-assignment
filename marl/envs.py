"""Environment helpers for the cooperative navigation task (MPE simple_spread).

Documentation: https://mpe2.farama.org/mpe2/simple_spread/
"""

from mpe2 import simple_spread_v3

# The settings used in the assignment. Override any of them with keyword arguments.
DEFAULT_ENV_KWARGS = {
    "N": 3,  # number of agents, and also the number of landmarks
    "local_ratio": 0.5,  # weight of the local (collision) reward; see the docs
    "max_cycles": 25,  # steps per episode
    "continuous_actions": False,  # 5 discrete actions per agent
}

ACTION_NAMES = ["no_action", "move_left", "move_right", "move_down", "move_up"]


def make_env(render_mode=None, benchmark_data=False, **env_kwargs):
    """Return a PettingZoo ParallelEnv: every agent acts at the same time.

    render_mode: None, "human" (opens a window) or "rgb_array" (returns images).
    benchmark_data: if True, each step's `infos` include distance and collision
        numbers. The evaluation code in marl/metrics.py uses them.
    """
    kwargs = {**DEFAULT_ENV_KWARGS, **env_kwargs}
    return simple_spread_v3.parallel_env(
        render_mode=render_mode, benchmark_data=benchmark_data, **kwargs
    )


def seed_vec_env(venv, seed):
    """Seed a SuperSuit vector environment.

    Stable-Baselines3's PPO(seed=...) does not work with SuperSuit vector
    environments, so we reset the inner environment once with a seed instead.
    Later resets continue from the same random number generator, which makes
    runs with the same seed repeatable.
    """
    inner = getattr(venv, "venv", venv)
    inner.reset(seed=seed)
