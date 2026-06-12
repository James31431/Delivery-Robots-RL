"""Evaluate an already-trained Q-table WITHOUT retraining.

Loads the saved table for the chosen environment and compares action-selection
modes (greedy / epsilon-greedy / softmax) against the Random baseline. For the
simple env the hand-coded Optimal ceiling is included too.

Artifacts are namespaced per env:
    q_table_simple.pkl    (produced by `python main.py --env simple`)
    q_table_complex.pkl   (produced by `python main.py --env complex`)

Usage:
    python evaluate_saved.py --env complex            # held-out seed, 100 eps
    python evaluate_saved.py --env simple
    python evaluate_saved.py --env complex --seed 42  # the TRAIN seed
    python evaluate_saved.py --env complex --temp 2.0 --episodes 200
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import config
from environment.complex_env import ComplexBuildingEnv, ComplexEnvConfig
from environment.simple_env import SimpleBuildingEnv
from rl.baselines import OptimalAgent
from rl.linear_agent import LinearMultiAgentQ, make_per_robot_featurizer
from rl.multi_agent_tabular import MultiAgentTabularQ
from rl.q_learning_agent import QLearningAgent


def _select(q, mode, n_act, temp, eps, rng):
    """Pick one action index from Q-values under the given selection mode."""
    if mode == "random":
        return int(rng.integers(n_act))
    if mode == "greedy":
        return int(rng.choice(np.flatnonzero(q == q.max())))
    if mode == "eps":
        if rng.random() < eps:
            return int(rng.integers(n_act))
        return int(rng.choice(np.flatnonzero(q == q.max())))
    if mode == "softmax":
        p = np.exp((q - q.max()) / temp)
        p /= p.sum()
        return int(rng.choice(n_act, p=p))
    raise ValueError(mode)


def _run_complex(seed, mode, episodes, temp, eps,
                 include_last_action=False, table_name="complex", linear=False):
    env_cfg = ComplexEnvConfig(obs_mode="per_robot", max_steps=500, include_last_action=include_last_action)
    env = ComplexBuildingEnv(config=env_cfg, seed=seed)
    if linear:
        featurize, n_features = make_per_robot_featurizer(env_cfg)
        agent = LinearMultiAgentQ(
            featurize=featurize,
            n_features=n_features,
            per_robot_action_size=env.per_robot_action_size,
            n_robots=env.cfg.n_robots,
            seed=config.TRAIN_SEED,
        )
        agent.load(config.q_table_path(table_name))
        q_of = agent.q_values            # linear: Q(obs,.) = phi(obs) @ W
    else:
        agent = MultiAgentTabularQ(
            per_robot_action_size=env.per_robot_action_size,
            n_robots=env.cfg.n_robots,
            seed=config.TRAIN_SEED,
        )
        agent.load(config.q_table_path(table_name))
        q_of = agent.get_q_values        # tabular: dict lookup
    n_act = agent.per_robot_action_size
    rng = np.random.default_rng(0)
    succ = 0
    total_r = 0.0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        success = False
        while not done and steps < env_cfg.max_steps:
            acts = tuple(
                _select(q_of(o), mode, n_act, temp, eps, rng)
                for o in obs
            )
            obs, r, done, info = env.step(acts)
            total_r += r
            steps += 1
            if isinstance(info, dict) and info.get("success"):
                success = True
        if not success:
            success = bool(getattr(env, "delivered", False))
        succ += int(success)
    return succ / episodes * 100.0, total_r / episodes


def _run_simple(seed, mode, episodes, temp, eps, optimal=False):
    env = SimpleBuildingEnv(max_steps=config.MAX_STEPS, seed=seed)
    if optimal:
        agent = OptimalAgent(env.action_space_size)
    else:
        agent = QLearningAgent(action_space_size=env.action_space_size, seed=config.TRAIN_SEED)
        agent.load(config.q_table_path("simple"))
    n_act = env.action_space_size
    rng = np.random.default_rng(0)
    succ = 0
    total_r = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        success = False
        while not done and steps < config.MAX_STEPS:
            if optimal:
                a = agent.choose_action(state)
            else:
                a = _select(agent.get_q_values(state), mode, n_act, temp, eps, rng)
            state, r, done, info = env.step(a)
            total_r += r
            steps += 1
            if isinstance(info, dict) and info.get("success"):
                success = True
        if not success:
            success = bool(getattr(env, "delivered", False))
        succ += int(success)
    return succ / episodes * 100.0, total_r / episodes


def _complex_variant(linear, memory):
    """Artifact name for a (linear, memory) combination of the complex env."""
    v = "complex"
    if linear:
        v += "_linear"
    if memory:
        v += "_mem"
    return v


def _train_cmd(variant):
    """The `main.py` command that produces a given variant's table."""
    if variant == "simple":
        return "python main.py --env simple"
    cmd = "python main.py --env complex"
    if "_linear" in variant:
        cmd += " --linear"
    if variant.endswith("_mem"):
        cmd += " --memory"
    return cmd


def _require_tables(parser, *variants):
    """Exit with a helpful message if any variant's saved table is missing."""
    missing = [v for v in variants if not os.path.exists(config.q_table_path(v))]
    if missing:
        lines = "\n".join(
            f"  {config.q_table_path(v)}  ->  train it with: {_train_cmd(v)}"
            for v in missing
        )
        parser.error(f"missing saved Q-table(s):\n{lines}")


def _print_complex_block(title, table_name, include_last_action, args, rows, linear):
    """Print one variant's full mode table for the complex env."""
    print(f"\n### {title}  ({config.q_table_path(table_name)})")
    print(f"{'Mode':<22} | {'Success%':>8} | {'AvgReward':>10}")
    print("-" * 48)
    for label, mode in rows:
        sr, ar = _run_complex(
            args.seed, mode, args.episodes, args.temp, args.eps,
            include_last_action=include_last_action, table_name=table_name, linear=linear,
        )
        print(f"{label:<22} | {sr:>7.0f}% | {ar:>10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved Q-table without retraining.")
    parser.add_argument("--env", choices=["simple", "complex"], default="complex")
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED)
    parser.add_argument("--episodes", type=int, default=config.NUM_EVALUATION_EPISODES)
    parser.add_argument("--temp", type=float, default=2.0, help="softmax temperature")
    parser.add_argument("--eps", type=float, default=0.37, help="epsilon for eps-greedy")
    parser.add_argument("--memory", action="store_true",
                        help="(complex) evaluate the include_last_action model "
                             "(the *_mem table) instead of the baseline.")
    parser.add_argument("--linear", action="store_true",
                        help="(complex) evaluate the linear function-approximation "
                             "agent (the *_linear table) instead of the tabular one.")
    parser.add_argument("--compare", action="store_true",
                        help="(complex) print the no-memory and memory variants "
                             "side by side. Combine with --linear to compare the "
                             "linear agent's two variants instead of the tabular one.")
    args = parser.parse_args()

    rows = [
        ("greedy (argmax Q)", "greedy"),
        (f"eps-greedy (eps={args.eps})", "eps"),
        (f"softmax (temp={args.temp})", "softmax"),
        ("Random baseline", "random"),
    ]

    # --- Side-by-side comparison (complex only) ----------------------------
    if args.compare:
        if args.env != "complex":
            parser.error("--compare is only supported for --env complex")
        base = _complex_variant(args.linear, memory=False)      # complex or complex_linear
        mem = _complex_variant(args.linear, memory=True)        # complex_mem or complex_linear_mem
        _require_tables(parser, base, mem)
        kind = "linear FA" if args.linear else "tabular"
        print(f"env=complex ({kind}) | seed={args.seed} | episodes={args.episodes}")
        _print_complex_block("no-memory baseline", base, False, args, rows, args.linear)
        _print_complex_block("1-step memory", mem, True, args, rows, args.linear)
        return

    # --- Single variant ----------------------------------------------------
    if args.env == "complex":
        table_name = _complex_variant(args.linear, args.memory)
    else:
        table_name = "simple"
    _require_tables(parser, table_name)
    print(f"Evaluating {config.q_table_path(table_name)} | env={args.env} | "
          f"seed={args.seed} | episodes={args.episodes}\n")
    print(f"{'Mode':<22} | {'Success%':>8} | {'AvgReward':>10}")
    print("-" * 48)

    for label, mode in rows:
        if args.env == "complex":
            sr, ar = _run_complex(
                args.seed, mode, args.episodes, args.temp, args.eps,
                include_last_action=args.memory, table_name=table_name, linear=args.linear,
            )
        else:
            sr, ar = _run_simple(args.seed, mode, args.episodes, args.temp, args.eps)
        print(f"{label:<22} | {sr:>7.0f}% | {ar:>10.1f}")

    # The Optimal ceiling is only defined for the simple env.
    if args.env == "simple":
        sr, ar = _run_simple(args.seed, "greedy", args.episodes, args.temp, args.eps, optimal=True)
        print(f"{'Optimal (ceiling)':<22} | {sr:>7.0f}% | {ar:>10.1f}")


if __name__ == "__main__":
    main()
