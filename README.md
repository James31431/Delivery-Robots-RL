# Delivery Robot Elevator RL

A tabular **Q-learning** project for a delivery robot navigating a multi-floor
building with an elevator. Compares the learned policy against **Random** and
**Optimal** baselines, plots training curves, and supports multi-seed
experiments.

> Educational / starter version. Pure Python + numpy + matplotlib — no
> PyTorch, no gymnasium.

---

## Project Goal

Train a single delivery robot to pick up the elevator, ride it to the correct
floor, exit, and deliver a package — using only reward signals.

The trained agent is compared against:

| Baseline       | Role                        |
|----------------|-----------------------------|
| `RandomAgent`  | Performance **floor**       |
| `OptimalAgent` | Performance **ceiling**     |

This sandwich tells you whether learning actually happened and how close it
got to optimal.

---

## Project Structure

```
Delivery-Robots-RL/
├── config.py                  # Hyperparameters, seeds, output paths
├── main.py                    # Train → save → compare → plot (simple env)
├── run_experiments.py         # Multi-seed sweep + multi-seed plots
├── demo_complex_env.py        # Demo: tabular Q-learning vs random on ComplexBuildingEnv
├── requirements.txt           # Runtime deps (numpy, matplotlib)
├── requirements-dev.txt       # +pytest
├── environment/
│   ├── simple_env.py          # SimpleBuildingEnv  (small, deterministic)
│   └── complex_env.py         # ComplexBuildingEnv (stochastic, multi-robot, multi-task)
├── rl/
│   ├── interfaces.py          # Lightweight Protocols (Env, Agent)
│   ├── q_learning_agent.py    # Tabular Q-learning
│   ├── baselines.py           # RandomAgent, OptimalAgent
│   ├── train.py               # train_agent(env, agent, ...)
│   └── evaluate.py            # evaluate_agent(env, agent, ...)
├── utils/
│   ├── logger.py              # save_training_stats → CSV
│   └── plot.py                # plot_training_curves (single + multi-seed)
└── tests/                     # pytest suite
```

---

## Environment Summary

- **5 floors** (`0..4`); 1 robot; 1 elevator; 1 delivery task per episode.
- Robot and elevator start at floor `0`. Target floor is randomly chosen from `1..4`.
- Episode ends on delivery, or after `max_steps` (default `50`).

**State** (5-tuple of ints):

```
(robot_floor, target_floor, elevator_floor, robot_inside_elevator, delivered)
```

**Actions**

| ID | Name              |
|----|-------------------|
| 0  | `wait`            |
| 1  | `enter_elevator`  |
| 2  | `exit_elevator`   |
| 3  | `elevator_up`     |
| 4  | `elevator_down`   |
| 5  | `deliver_package` |

**Reward design**

| Event               | Reward |
|---------------------|--------|
| Successful delivery | +100   |
| Each normal step    |  -1    |
| Invalid action      |  -5    |
| `wait`              |  -2    |

---

## How to Run

From the repository root:

```bash
# 1. Create a virtualenv (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install runtime deps
pip install -r requirements.txt

# 3. (Optional) Install dev deps if you want to run tests
pip install -r requirements-dev.txt

# 4. Train + evaluate + compare against baselines + save plot
python main.py

# 5. Run tests
pytest

# 6. Multi-seed experiment (writes to results/)
python run_experiments.py
```

### Outputs (gitignored)

| File                    | Produced by              | What it is                              |
|-------------------------|--------------------------|-----------------------------------------|
| `q_table.pkl`           | `main.py`                | Pickled Q-table                         |
| `training_stats.csv`    | `main.py`                | Per-episode reward / steps / success / ε |
| `training_curves.png`   | `main.py`                | 2×2 figure of training curves           |
| `results/<run>/`        | `run_experiments.py`     | One CSV per (seed, agent) + plots       |

All generated artifacts are gitignored.

### Plotting standalone

`utils/plot.py` is also a CLI for re-plotting an existing CSV:

```bash
python -m utils.plot --stats training_stats.csv --out training_curves.png --window 100
python -m utils.plot --stats training_stats.csv --optimal 95.52   # add a reference line
```

---

## Sample Results

After training for **2000 episodes** with `TRAIN_SEED=42` and evaluating for
**100 episodes** with `EVAL_SEED=123`:

```
=== Baseline Comparison ===
Agent        |  AvgReward | Success% |  AvgSteps
--------------------------------------------------
Q-Learning   |      95.52 |   100.0% |      5.48
Random       |    -126.07 |    18.0% |     26.78
Optimal      |      95.52 |   100.0% |      5.48
==================================================
```

The learned Q-policy matches the hand-coded optimal policy on this
environment — a sanity check that tabular Q-learning is sufficient here.

After running `main.py`, see `training_curves.png` for reward / success-rate /
steps / ε-decay curves over training.

---

## Complex Environment (`ComplexBuildingEnv`)

A harder sibling to `SimpleBuildingEnv` lives in
[environment/complex_env.py](environment/complex_env.py). It is **not** a
replacement — both envs satisfy `EnvironmentProtocol` so `train_agent` and
`evaluate_agent` accept either unchanged.

### Why two environments?

| Aspect              | `SimpleBuildingEnv`                  | `ComplexBuildingEnv`                                     |
|---------------------|--------------------------------------|-----------------------------------------------------------|
| Floors              | 5                                    | 8 (configurable)                                          |
| Robots              | 1                                    | 2 (configurable, 1–4)                                     |
| Elevators           | 1                                    | 2 (configurable)                                          |
| Tasks               | 1 per episode                        | Refilling queue (default 3 active, 6 total per episode)   |
| Dynamics            | Deterministic                        | Stochastic: elevator delay, breakdowns, task spawn        |
| State space         | ~400                                 | Combinatorially large                                     |
| Flat action space   | 6                                    | `per_robot_actions ** n_robots` (576 with defaults)       |
| Suited for          | Tabular Q-learning (provably solves) | Function approximation (DQN) — tabular fails here         |

### Stochastic dynamics

- **Elevator delay** (`p_elevator_delay`, default `0.10`): elevator move turns
  into a no-op for one step.
- **Breakdowns** (`p_breakdown`, default `0.01`): elevator becomes unusable
  for `breakdown_duration` steps; any robot inside is ejected at the current
  floor (extra penalty if they were carrying a task).
- **Task spawn** (`p_new_task`, default `0.20`): when a slot is empty, a new
  task spawns with this probability per step, up to `total_task_budget`.

### State and action shape

```python
from environment.complex_env import ComplexBuildingEnv, ComplexEnvConfig

env = ComplexBuildingEnv(config=ComplexEnvConfig(), seed=42)
state = env.reset()       # tuple of tuples — hashable, dict-friendly
vec   = env.to_vector()   # np.ndarray, float32 — for function approximation
```

Actions can be passed two ways:

- A flat `int` in `[0, flat_action_space_size)` — agents that already speak
  the `AgentProtocol` (e.g. the tabular Q-agent) just keep working
- A per-robot sequence of length `n_robots` — natural for multi-agent agents

### Reward shape

`+100` per delivery, `+10` per pickup (sparse-reward shaping),
`-0.5` per robot per step, `-2` per `wait`, `-5` per invalid sub-action,
`-20` if a breakdown ejects a robot that was carrying a task.

### Try it

```bash
python demo_complex_env.py
```

Trains the existing tabular Q-agent for 300 episodes on `ComplexBuildingEnv`
and compares it to a random baseline. Expected outcome: Q-learning barely
beats (or ties) random — the state space is too big for a tabular table to
generalize. This is the motivation for a follow-up DQN issue.

---

## Reproducibility

Seeds are set in [config.py](config.py):

- `TRAIN_SEED = 42` — controls env target sequence + agent exploration RNG during training
- `EVAL_SEED  = 123` — fresh seed for evaluation so we're not testing on training trajectories

`run_experiments.py` sweeps across `[42, 43, 44, ...]` for variance estimates.

---

## Design Notes

- **Dependency injection** — `train_agent` / `evaluate_agent` accept any
  env/agent satisfying [`rl/interfaces.py`](rl/interfaces.py). Swapping in a
  DQN agent or `ComplexBuildingEnv` later won't require pipeline changes.
- **Protocol-based typing** (no ABCs) — structural typing keeps the contract
  loose and inheritance-free.
- **Tabular Q-table** as `dict[state_tuple → np.ndarray]` — exactly right for
  this ~400-state problem; lazy zero-init on unseen states.
