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
├── main.py                    # Train → save → compare → plot
├── run_experiments.py         # Multi-seed sweep + multi-seed plots
├── requirements.txt           # Runtime deps (numpy, matplotlib)
├── requirements-dev.txt       # +pytest
├── environment/
│   └── simple_env.py          # SimpleBuildingEnv
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
