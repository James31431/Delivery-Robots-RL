"""Entry point: train, evaluate, and save artifacts for one of the two envs.

Two pipelines are kept here side-by-side so you can switch with one flag.

Usage:
    python main.py                  # default — SimpleBuildingEnv
    python main.py --env simple     # explicit
    python main.py --env complex    # the harder, stochastic, multi-robot env

Or change ``DEFAULT_ENV`` below to flip the default permanently.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Dict, Optional

# Ensure the project root is importable when run directly.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from environment.complex_env import ComplexBuildingEnv, ComplexEnvConfig
from environment.simple_env import SimpleBuildingEnv
from rl.baselines import OptimalAgent, RandomAgent
from rl.linear_agent import LinearMultiAgentQ, make_per_robot_featurizer
from rl.multi_agent_tabular import MultiAgentTabularQ
from rl.q_learning_agent import QLearningAgent
from rl.train import train_agent
from rl.evaluate import evaluate_agent
from utils.logger import save_training_stats
from utils.visualizer import EpisodeReplayRenderer
from utils.plot import plot_training_curves


# Change to "simple" to flip the default without editing the CLI invocation.
DEFAULT_ENV = "complex"


# --- Shared helpers --------------------------------------------------------

def _build_q_agent(action_space_size: int) -> QLearningAgent:
    return QLearningAgent(
        action_space_size=action_space_size,
        learning_rate=config.LEARNING_RATE,
        discount_factor=config.DISCOUNT_FACTOR,
        epsilon=config.EPSILON,
        epsilon_decay=config.EPSILON_DECAY,
        min_epsilon=config.MIN_EPSILON,
        seed=config.TRAIN_SEED,
    )


def _save_artifacts(agent: Any, stats: Dict[str, Any], env_name: str) -> str:
    q_path = os.path.join(PROJECT_ROOT, config.q_table_path(env_name))
    agent.save(q_path)
    print(f"\nSaved Q-table to {q_path}")
    stats_path = os.path.join(PROJECT_ROOT, config.training_stats_path(env_name))
    save_training_stats(stats, stats_path)
    print(f"Saved training stats to {stats_path}")
    return stats_path


def _evaluate_all(
    agents: Dict[str, Any],
    fresh_env_fn: Callable[[], Any],
    max_steps: int,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_agent(
            fresh_env_fn(),
            agent,
            episodes=config.NUM_EVALUATION_EPISODES,
            max_steps=max_steps,
        )
    return results


def _print_comparison(results: Dict[str, Dict[str, float]]) -> None:
    print("\n=== Baseline Comparison ===")
    print(f"{'Agent':<12} | {'AvgReward':>10} | {'Success%':>8} | {'AvgSteps':>9}")
    print("-" * 50)
    for name, r in results.items():
        steps = r["average_steps_successful"]
        steps_str = f"{steps:.2f}" if steps == steps else "n/a"  # NaN check
        print(
            f"{name:<12} | {r['average_reward']:>10.2f} | "
            f"{r['success_rate'] * 100:>7.1f}% | {steps_str:>9}"
        )
    print("=" * 50)


def _plot(stats_path: str, optimal_reward: Optional[float], env_name: str) -> None:
    plot_path = os.path.join(PROJECT_ROOT, config.training_curves_path(env_name))
    plot_training_curves(
        stats_path=stats_path,
        output_path=plot_path,
        optimal_reward=optimal_reward,
    )
    print(f"Saved training curves to {plot_path}")


def _record_and_replay_episode(
    env: Any,
    agent: Any,
    max_steps: int,
    record_path: str,
) -> None:
    print(f"\nRecording a single evaluation episode to {record_path}...")
    evaluate_agent(
        env,
        agent,
        episodes=1,
        max_steps=max_steps,
        record_episode_path=record_path,
    )
    try:
        print("Launching replay renderer...")
        EpisodeReplayRenderer(record_path).run(autoplay=True)
    except RuntimeError as exc:
        print(f"Skipping replay renderer: {exc}")


class _FullStateReplayAgent:
    """Adapter that lets the multi-agent policy run on a full complex state.

    The trained policy still operates on per-robot observations. For replay,
    we want a full-state recording so the renderer can reconstruct the entire
    scene. This adapter converts the recorded full state into the per-robot
    observation tuples the policy expects.
    """

    def __init__(self, agent: MultiAgentTabularQ, env_cfg: ComplexEnvConfig) -> None:
        self._agent = agent
        self._n_robots = env_cfg.n_robots
        self._n_elevators = env_cfg.n_elevators
        self._n_tasks = env_cfg.n_tasks

    def _to_per_robot_observations(self, state: Any) -> tuple[tuple[int, ...], ...]:
        flat_state: list[int] = []
        for item in state:
            if isinstance(item, (list, tuple)):
                flat_state.extend(int(value) for value in item)
            else:
                flat_state.append(int(item))

        observations: list[tuple[int, ...]] = []
        index = 0
        robot_chunks: list[tuple[int, int, int]] = []
        elevator_chunks: list[tuple[int, int, int]] = []
        task_chunks: list[tuple[int, int, int]] = []

        for _ in range(self._n_robots):
            robot_chunks.append(tuple(flat_state[index:index + 3]))
            index += 3
        for _ in range(self._n_elevators):
            elevator_chunks.append(tuple(flat_state[index:index + 3]))
            index += 3
        for _ in range(self._n_tasks):
            task_chunks.append(tuple(flat_state[index:index + 3]))
            index += 3

        for robot_idx in range(self._n_robots):
            robot_floor, inside_elevator, carrying_task = robot_chunks[robot_idx]
            obs: list[int] = [robot_floor, inside_elevator, carrying_task]
            for elevator_floor, direction, broken_remaining in elevator_chunks:
                obs.append(elevator_floor)
                obs.append(1 if broken_remaining > 0 else 0)

            if carrying_task >= 0:
                obs.append(-1)
                obs.append(task_chunks[carrying_task][1])
            else:
                best_pickup = -1
                best_dist: Optional[int] = None
                for pickup_floor, delivery_floor, status in task_chunks:
                    if status != 1:
                        continue
                    distance = abs(pickup_floor - robot_floor)
                    if best_dist is None or distance < best_dist:
                        best_pickup = pickup_floor
                        best_dist = distance
                obs.append(best_pickup)
                obs.append(-1)

            observations.append(tuple(obs))

        return tuple(observations)

    def choose_action(self, state: Any, greedy: bool = False) -> Any:
        return self._agent.choose_action(self._to_per_robot_observations(state), greedy=greedy)


# --- Pipelines -------------------------------------------------------------

def run_simple() -> None:
    """Pipeline against SimpleBuildingEnv: Q-Learning vs Random vs Optimal."""
    env = SimpleBuildingEnv(max_steps=config.MAX_STEPS, seed=config.TRAIN_SEED)
    agent = _build_q_agent(env.action_space_size)

    print("Training Q-learning agent on SimpleBuildingEnv...")
    agent, stats = train_agent(
        env,
        agent,
        episodes=config.NUM_TRAINING_EPISODES,
        max_steps=config.MAX_STEPS,
    )

    stats_path = _save_artifacts(agent, stats, "simple")

    def fresh_env() -> SimpleBuildingEnv:
        return SimpleBuildingEnv(max_steps=config.MAX_STEPS, seed=config.EVAL_SEED)

    agents = {
        "Q-Learning": agent,
        "Random": RandomAgent(env.action_space_size, seed=config.EVAL_SEED),
        "Optimal": OptimalAgent(env.action_space_size),
    }

    results = _evaluate_all(agents, fresh_env, config.MAX_STEPS)
    _print_comparison(results)
    _plot(stats_path, results["Optimal"]["average_reward"], "simple")


def run_complex(use_memory: bool = False, use_linear: bool = False) -> None:
    """Pipeline against ComplexBuildingEnv with per-robot partial observability.

    Uses ``obs_mode='per_robot'`` so each robot's observation is a small
    local tuple, and a multi-agent tabular Q-learner (shared Q-table) acts
    independently for each robot. The Random baseline keeps the same env
    config but emits a flat action int — the env decodes it into per-robot
    actions internally. OptimalAgent is intentionally omitted (hand-coded
    for SimpleBuildingEnv's state shape).

    ``use_memory=True`` enables ``include_last_action`` (1-step action memory).
    ``use_linear=True`` swaps the tabular shared Q-table for a linear
    function approximator (``LinearMultiAgentQ``) over a one-hot feature
    encoding of the same per-robot observation.

    Artifacts are namespaced per combination (``complex``, ``complex_mem``,
    ``complex_linear``, ``complex_linear_mem``) so variants never clobber each
    other and can be compared side by side.
    """
    variant = "complex"
    if use_linear:
        variant += "_linear"
    if use_memory:
        variant += "_mem"

    env_cfg = ComplexEnvConfig(obs_mode="per_robot", max_steps=500, include_last_action=use_memory)
    env = ComplexBuildingEnv(config=env_cfg, seed=config.TRAIN_SEED)
    if use_linear:
        featurize, n_features = make_per_robot_featurizer(env_cfg)
        agent = LinearMultiAgentQ(
            featurize=featurize,
            n_features=n_features,
            per_robot_action_size=env.per_robot_action_size,
            n_robots=env_cfg.n_robots,
            learning_rate=0.05,   # NOT config.LEARNING_RATE (0.3 diverges with FA)
            discount_factor=config.DISCOUNT_FACTOR,
            epsilon=config.EPSILON,
            epsilon_decay=config.EPSILON_DECAY,
            min_epsilon=config.MIN_EPSILON,
            seed=config.TRAIN_SEED,
        )
        print("Training LinearMultiAgentQ on ComplexBuildingEnv (per-robot obs)...")
        print(f"  n_features={n_features}, lr=0.05")
    else:
        agent = MultiAgentTabularQ(
            per_robot_action_size=env.per_robot_action_size,
            n_robots=env_cfg.n_robots,
            learning_rate=config.LEARNING_RATE,
            discount_factor=config.DISCOUNT_FACTOR,
            epsilon=config.EPSILON,
            epsilon_decay=config.EPSILON_DECAY,
            min_epsilon=config.MIN_EPSILON,
            seed=config.TRAIN_SEED,
        )
        print("Training MultiAgentTabularQ on ComplexBuildingEnv (per-robot obs)...")

    print(
        f"  per_robot_action_size={env.per_robot_action_size}, "
        f"flat_action_space_size={env.flat_action_space_size}, "
        f"max_steps={env_cfg.max_steps}"
    )
    agent, stats = train_agent(
        env,
        agent,
        episodes=config.NUM_TRAINING_EPISODES,
        max_steps=env_cfg.max_steps,
    )

    stats_path = _save_artifacts(agent, stats, variant)

    def fresh_env() -> ComplexBuildingEnv:
        return ComplexBuildingEnv(config=env_cfg, seed=config.EVAL_SEED)

    agents = {
        "MultiAgent-Q": agent,
        "Random": RandomAgent(env.action_space_size, seed=config.EVAL_SEED),
    }

    results = _evaluate_all(agents, fresh_env, env_cfg.max_steps)
    _print_comparison(results)
    _plot(stats_path, None, variant)

    replay_cfg = ComplexEnvConfig(
        n_floors=env_cfg.n_floors,
        n_robots=env_cfg.n_robots,
        n_elevators=env_cfg.n_elevators,
        n_tasks=env_cfg.n_tasks,
        total_task_budget=env_cfg.total_task_budget,
        max_steps=env_cfg.max_steps,
        p_elevator_delay=env_cfg.p_elevator_delay,
        p_breakdown=env_cfg.p_breakdown,
        breakdown_duration=env_cfg.breakdown_duration,
        p_new_task=env_cfg.p_new_task,
        obs_mode="full",
    )
    replay_env = ComplexBuildingEnv(config=replay_cfg, seed=config.EVAL_SEED)
    replay_path = os.path.join(PROJECT_ROOT, "complex_eval_episode.json")
    _record_and_replay_episode(
        replay_env,
        _FullStateReplayAgent(agent, env_cfg),
        env_cfg.max_steps,
        replay_path,
    )


# --- Entry point -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Q-learning on a delivery-robot env."
    )
    parser.add_argument(
        "--env",
        choices=["simple", "complex"],
        default=DEFAULT_ENV,
        help=f"Which environment to use (default: {DEFAULT_ENV}).",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="(complex env only) append each robot's last action to its "
             "observation and save under the 'complex_mem' artifacts, "
             "leaving the memoryless 'complex' baseline intact.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="(complex env only) use the linear function approximator "
             "(LinearMultiAgentQ) instead of the tabular shared Q-table; "
             "saves under the 'complex_linear' artifacts.",
    )
    args = parser.parse_args()

    if args.env == "simple":
        run_simple()
    else:
        run_complex(use_memory=args.memory, use_linear=args.linear)


if __name__ == "__main__":
    main()
