"""Task 1: explore the cooperative navigation task with a random policy.

Examples:
    python explore.py                          # spaces, one observation, random baseline
    python explore.py --render                 # also watch one random episode
    python explore.py --gif results/random.gif # save one random episode as a GIF
    python explore.py --n-agents 4             # same, with 4 agents and 4 landmarks
"""

import argparse

from marl.envs import ACTION_NAMES, make_env
from marl.metrics import evaluate, format_table, render_episode, save_json


def random_policy(agent, observation, env):
    """Return a random valid action for `agent` (ignores the observation)."""
    # TODO(Task 1a): return a random action for this agent.
    # Hint: every agent has its own action space: env.action_space(agent).
    raise NotImplementedError("TODO(Task 1a): see the comments above")


def split_observation(observation, n_agents):
    """Split one agent's observation vector into named parts.

    The order of the parts is given in the documentation
    (https://mpe2.farama.org/mpe2/simple_spread/):
        self_vel, self_pos, landmark_rel_positions,
        other_agent_rel_positions, communication

    Return a dict {part name: numpy array}.
    """
    # TODO(Task 1b): work out how many numbers each part has, for any n_agents.
    # Check: the sizes must add up to len(observation) (18 when n_agents = 3).
    raise NotImplementedError("TODO(Task 1b): see the comments above")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--local-ratio", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="watch one episode")
    parser.add_argument("--gif", help="save one episode to this GIF file")
    args = parser.parse_args()
    env_kwargs = {"N": args.n_agents, "local_ratio": args.local_ratio}

    env = make_env(**env_kwargs)
    observations, _ = env.reset(seed=0)
    agent = env.agents[0]
    print("Agents:", env.agents)
    print("Observation space:", env.observation_space(agent))
    print("Action space:", env.action_space(agent), "=", ACTION_NAMES)

    print(f"\nFirst observation of {agent}, split into parts:")
    for name, values in split_observation(observations[agent], args.n_agents).items():
        print(f"  {name:28s} {values.round(2)}")

    actions = {a: int(random_policy(a, observations[a], env)) for a in env.agents}
    _, rewards, _, _, _ = env.step(actions)
    print("\nOne random step. Actions:", actions)
    print("Rewards:", {a: round(float(r), 3) for a, r in rewards.items()})
    env.close()

    if args.render or args.gif:
        render_episode(random_policy, env_kwargs, seed=0, gif_path=args.gif)

    print(f"\nRandom policy over {args.episodes} episodes:")
    summary = evaluate(random_policy, env_kwargs, n_episodes=args.episodes)
    print(format_table({"random": summary}))
    path = f"results/random_N{args.n_agents}_lr{args.local_ratio}.json"
    save_json({"env_kwargs": env_kwargs, **summary}, path)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
