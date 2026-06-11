"""Evaluate the already-trained q_table.pkl WITHOUT retraining.

Loads the saved MultiAgentTabularQ table and runs it on a held-out seed,
comparing greedy / epsilon-greedy / softmax action selection against the
Random baseline.

Usage:
    python evaluate_saved.py                  # default: eval seed, 100 episodes
    python evaluate_saved.py --episodes 200
    python evaluate_saved.py --seed 123
    python evaluate_saved.py --temp 2.0       # softmax temperature
"""

from __future__ import annotations

import argparse

import numpy as np

import config
from environment.complex_env import ComplexBuildingEnv, ComplexEnvConfig
from rl.multi_agent_tabular import MultiAgentTabularQ


def load_agent(env: ComplexBuildingEnv) -> MultiAgentTabularQ:
    agent = MultiAgentTabularQ(
        per_robot_action_size=env.per_robot_action_size,
        n_robots=env.cfg.n_robots,
        seed=config.TRAIN_SEED,
    )
    agent.load(config.Q_TABLE_PATH)
    return agent


def run(env_cfg, seed, mode, episodes, temp, eps, rng):
    """mode in {'greedy', 'eps', 'softmax', 'random'} -> success%, avg_reward."""
    env = ComplexBuildingEnv(config=env_cfg, seed=seed)
    agent = load_agent(env)
    n_act = agent.per_robot_action_size
    succ = 0
    total_r = 0.0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        success = False
        while not done and steps < env_cfg.max_steps:
            acts = []
            for o in obs:
                q = agent.get_q_values(o)
                if mode == "random":
                    a = int(rng.integers(n_act))
                elif mode == "greedy":
                    a = int(rng.choice(np.flatnonzero(q == q.max())))
                elif mode == "eps":
                    a = (int(rng.integers(n_act)) if rng.random() < eps
                         else int(rng.choice(np.flatnonzero(q == q.max()))))
                elif mode == "softmax":
                    z = (q - q.max()) / temp
                    p = np.exp(z)
                    p /= p.sum()
                    a = int(rng.choice(n_act, p=p))
                else:
                    raise ValueError(mode)
                acts.append(a)
            obs, r, done, info = env.step(tuple(acts))
            total_r += r
            steps += 1
            if isinstance(info, dict) and info.get("success"):
                success = True
        if not success:
            success = bool(getattr(env, "delivered", False))
        if success:
            succ += 1

    return succ / episodes * 100.0, total_r / episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved q_table.pkl.")
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED)
    parser.add_argument("--episodes", type=int, default=config.NUM_EVALUATION_EPISODES)
    parser.add_argument("--temp", type=float, default=2.0, help="softmax temperature")
    parser.add_argument("--eps", type=float, default=0.37, help="epsilon for eps-greedy")
    args = parser.parse_args()

    env_cfg = ComplexEnvConfig(obs_mode="per_robot", max_steps=500)

    print(f"Evaluating {config.Q_TABLE_PATH} | seed={args.seed} | "
          f"episodes={args.episodes}\n")
    print(f"{'Mode':<22} | {'Success%':>8} | {'AvgReward':>10}")
    print("-" * 48)
    modes = [
        ("greedy (argmax Q)", "greedy", {}),
        (f"eps-greedy (eps={args.eps})", "eps", {}),
        (f"softmax (temp={args.temp})", "softmax", {}),
        ("Random baseline", "random", {}),
    ]
    for label, mode, _ in modes:
        rng = np.random.default_rng(0)  # fixed RNG -> reproducible comparison
        sr, ar = run(env_cfg, args.seed, mode, args.episodes, args.temp, args.eps, rng)
        print(f"{label:<22} | {sr:>7.0f}% | {ar:>10.1f}")


if __name__ == "__main__":
    main()
