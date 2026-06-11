# Delivery Robot Elevator RL

A tabular **Q-learning** project for delivery robots navigating multi-floor
buildings with elevators. Ships with two environments:

- **`SimpleBuildingEnv`** — 5 floors, 1 robot, 1 elevator, 1 task,
  deterministic. Tabular Q-learning provably solves it.
- **`ComplexBuildingEnv`** — 8 floors, 2 robots, 2 elevators, refilling task
  queue, stochastic elevator delays and breakdowns. Solved with **per-robot
  partial observation** + a **shared multi-agent Q-table**.

The pipeline compares learned policies against **Random** (floor) and
**Optimal** (ceiling, simple env only) baselines, plots training curves, and
supports multi-seed experiments.

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
│   ├── q_learning_agent.py    # Tabular Q-learning (single agent)
│   ├── multi_agent_tabular.py # Multi-agent tabular Q (shared Q-table)
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
python main.py --env simple      # default
python main.py --env complex     # the harder, stochastic, multi-robot env

# 5. Evaluate an ALREADY-trained table (no retraining) and compare
#    greedy / epsilon-greedy / softmax action selection vs Random
python evaluate_saved.py --env simple
python evaluate_saved.py --env complex
python evaluate_saved.py --env complex --seed 42 --episodes 200 --temp 2.0

# 6. Run tests
pytest

# 7. Multi-seed experiment (writes to results/)
python run_experiments.py
```

### Outputs (gitignored)

Artifacts are **namespaced per environment** so the simple and complex
pipelines never overwrite each other (`{env}` is `simple` or `complex`):

| File                        | Produced by              | What it is                               |
|-----------------------------|--------------------------|------------------------------------------|
| `q_table_{env}.pkl`         | `main.py`                | Pickled Q-table                          |
| `training_stats_{env}.csv`  | `main.py`                | Per-episode reward / steps / success / ε |
| `training_curves_{env}.png` | `main.py`                | 2×2 figure of training curves            |
| `results/<run>/`            | `run_experiments.py`     | One CSV per (seed, agent) + plots        |

The path helpers `config.q_table_path(env)`, `config.training_stats_path(env)`,
and `config.training_curves_path(env)` produce these names. All generated
artifacts are gitignored.

### Plotting standalone

`utils/plot.py` is also a CLI for re-plotting an existing CSV:

```bash
python -m utils.plot --stats training_stats_complex.csv --out training_curves_complex.png --window 100
python -m utils.plot --stats training_stats_simple.csv --optimal 95.52   # add a reference line
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

`SimpleBuildingEnv` exists to verify that the pipeline works end-to-end and
to give tabular Q-learning a problem it can fully solve. `ComplexBuildingEnv`
exists to break that comfort zone — to show what happens when state and
action spaces explode, and to motivate the architectural choices (partial
observability, multi-agent decomposition) needed to keep tabular methods
viable.

| Aspect              | `SimpleBuildingEnv`                  | `ComplexBuildingEnv`                                     |
|---------------------|--------------------------------------|-----------------------------------------------------------|
| Floors              | 5                                    | 8 (configurable)                                          |
| Robots              | 1                                    | 2 (configurable, 1–4)                                     |
| Elevators           | 1                                    | 2 (configurable)                                          |
| Tasks               | 1 per episode                        | Refilling queue: 3 active slots, 6 per episode (defaults) |
| Dynamics            | Deterministic                        | Stochastic: elevator delay, breakdowns, task spawn        |
| State space (full)  | ~400                                 | Combinatorially huge (>10⁶ reachable)                     |
| Flat action space   | 6                                    | `per_robot_actions ** n_robots` (576 with defaults)       |
| Observation mode    | Always full state                    | `full` or `per_robot` (POMDP); the latter is the one that actually works for tabular |
| Suited for          | Tabular Q-learning (provably solves) | Multi-agent tabular under PO; full DQN later              |

### The world

A building with `n_floors` floors (`0..n_floors-1`). `n_robots` robots and
`n_elevators` elevators live inside it. There is a **task queue** of up to
`n_tasks` active tasks. Each task has a pickup floor and a delivery floor.
A robot picks up a task by being on its pickup floor (outside any elevator)
and selecting the `pickup_task_<id>` action; later, it delivers by being on
the task's delivery floor (also outside) and selecting `deliver`.

### Stochastic dynamics

Three independent sources of randomness make this env interesting:

| Source                       | Default            | Effect                                                                                            |
|------------------------------|--------------------|---------------------------------------------------------------------------------------------------|
| **Elevator delay**           | `p_elevator_delay = 0.10` | An elevator's planned movement turns into a no-op for that step (think slow door close).         |
| **Breakdown**                | `p_breakdown = 0.01`      | An elevator becomes unusable for `breakdown_duration = 5` steps. Any robot inside is ejected at the current floor; if it was carrying a task it eats a `-20` penalty.   |
| **Task spawn**               | `p_new_task = 0.20`       | When a task slot is empty (delivered → INACTIVE), refill it with a new random `(pickup_floor, delivery_floor)`. Capped at `total_task_budget` spawns per episode.        |

All three probabilities are independent per step, per elevator (or per slot
for spawn). They're all dial-able through `ComplexEnvConfig`.

### Full state representation (`obs_mode='full'`)

State is a tuple of small tuples — hashable, dict-friendly. The number of
entries equals `n_robots + n_elevators + n_tasks`:

```python
(
    # n_robots entries
    (robot_floor, inside_elevator_id_or_-1, carrying_task_slot_or_-1),
    ...

    # n_elevators entries
    (floor, direction, broken_remaining),
    # direction: 0=idle, 1=up, 2=down

    # n_tasks entries (one per slot in the queue)
    (pickup_floor, delivery_floor, status),
    # status: 0=INACTIVE, 1=PENDING, 2=PICKED_UP
)
```

A continuous form is available too via `env.to_vector()` (a `float32`
ndarray of length `state_vector_size`), intended for future
function-approximation agents — it additionally includes `self.steps`
because neural nets can use a time signal productively whereas tabular
tables would only inflate from it.

> **Why the step counter is *not* in the hashable state.** Including a
> monotonic counter made every state unique. After 5000 episodes the
> Q-table held 5 million entries and ate 18 GB. Dropping it lets revisits
> happen, which is the whole point of tabular methods.

### Per-robot partial observation (`obs_mode='per_robot'`)

The full state has so many degrees of freedom (`>10⁶` reachable
configurations) that a tabular Q-table can't reuse experience. The fix is
to give each robot a small **local observation** instead of the joint
state. This turns the MDP into a **POMDP** — Q-learning loses its formal
convergence guarantees but in practice still finds a good policy.

Each robot's observation is the same shape regardless of which robot it
is, so a single Q-table can be shared across the team:

```python
# Per-robot observation tuple (9 ints for n_elevators=2)
(
    own_floor,                  # 0..n_floors-1
    own_inside_elevator,        # -1 or elevator_id
    own_carrying_task,          # -1 or task slot

    elev_0_floor,
    elev_0_broken,              # 0 or 1
    elev_1_floor,
    elev_1_broken,

    pickup_target,              # nearest pending pickup floor, or -1 if carrying
    delivery_target,            # carried task's delivery floor, or -1 if not carrying
)
```

Reachable observation space per robot: **on the order of 50,000 unique
tuples in practice**, fully tractable for tabular methods. Memory savings
versus full-state mode: **~2000× smaller Q-table.**

`env.get_state()` returns the per-robot tuple-of-observations directly when
`obs_mode='per_robot'`, so existing `train_agent` / `evaluate_agent` work
without modification.

### Action space

Each robot has its own per-robot action set (`24` actions for defaults):

| Per-robot action            | Effect                                                                       |
|-----------------------------|------------------------------------------------------------------------------|
| `wait`                      | Robot does nothing this step (`-2` reward).                                  |
| `enter_elevator_<id>`       | Enter elevator `<id>` (valid only if same floor, robot outside, not broken).  |
| `exit_elevator`             | Step out at the elevator's current floor.                                    |
| `request_elevator_<e>_to_floor_<f>` | Set elevator `e`'s target to floor `f`. Elevator then advances one step toward target on subsequent ticks (subject to delay / breakdown). |
| `pickup_task_<slot>`        | Pick up the task in slot `<slot>` (valid if pending, robot outside, on its pickup floor, not already carrying). |
| `deliver`                   | Deliver the currently-carried task (valid if outside elevator and on its delivery floor). |

Per-robot action size:
`1 + n_elevators + 1 + n_elevators*n_floors + n_tasks + 1`. With defaults:
`1 + 2 + 1 + 16 + 3 + 1 = 24`.

The **flat** (team-joint) action space is `per_robot_actions ** n_robots`
(`576` for defaults). `env.step()` accepts either:

- A flat `int` in `[0, flat_action_space_size)` — auto-decoded into per-robot
  actions via mixed-radix base conversion. This keeps `RandomAgent` and any
  single-int `AgentProtocol` agent working.
- A `Sequence[int]` of length `n_robots` — what multi-agent agents emit
  naturally. `MultiAgentTabularQ.choose_action` returns this form.

### Reward design

| Event                                            | Reward          |
|--------------------------------------------------|-----------------|
| Successful delivery                              | `+100` per task |
| Pickup (intermediate, combats reward sparsity)   | `+10`           |
| Per robot, per step                              | `-0.5`          |
| `wait` per robot                                 | `-2`            |
| Any invalid sub-action                           | `-5`            |
| Breakdown ejecting a carrying robot              | `-20`           |

The `+10` pickup bonus is reward shaping — it gives the agent a denser
gradient than a delivery-only signal would. Cooperative team reward: all
robots see the same reward each step, so credit assignment is implicit.

### Episode termination

The episode ends when **either** of:

1. All `total_task_budget` tasks (default 6) have been delivered, **or**
2. `max_steps` (default 300, or whatever `ComplexEnvConfig(max_steps=…)`
   is set to) is reached.

`env.delivered` is `True` iff all budget tasks were delivered — this is
what `evaluate_agent` reports as "success rate." Note the high bar: in a
stochastic env with breakdowns, full-budget success is rare even for a
well-trained policy. Watch the **reward** column for honest learning signal.

### `ComplexEnvConfig` knobs

Every dial you'd want to twiddle lives on a single dataclass:

```python
from environment.complex_env import ComplexEnvConfig

cfg = ComplexEnvConfig(
    # Shape
    n_floors=8, n_robots=2, n_elevators=2,
    n_tasks=3, total_task_budget=6, max_steps=300,

    # Observation / training mode
    obs_mode="per_robot",   # "full" or "per_robot"

    # Stochastic dynamics — set to 0.0 for a deterministic variant
    p_elevator_delay=0.10,
    p_breakdown=0.01,
    breakdown_duration=5,
    p_new_task=0.20,

    # Reward weights (all overridable)
    reward_delivery=100.0,
    reward_pickup=10.0,
    reward_step=-0.5,
    reward_wait=-2.0,
    reward_invalid=-5.0,
    reward_breakdown_eject=-20.0,
)
```

### The agent for it — `MultiAgentTabularQ`

The complex env is paired with [`rl/multi_agent_tabular.py`](rl/multi_agent_tabular.py):

- **One shared Q-table** across all robots (sample-efficient — experience
  collected by any robot updates the same parameters).
- **Independent learners** — each robot picks its action via epsilon-greedy
  lookup on its own observation; no explicit coordination.
- **Cooperative team reward** — each step's reward updates every robot's
  transition into the shared table.

It satisfies the existing `AgentProtocol`, so `train_agent` and
`evaluate_agent` consume it unchanged.

### Running the complex pipeline

```bash
python main.py --env complex
```

This launches the full PO + multi-agent pipeline using values from
[config.py](config.py). With default tuning (`50,000` episodes, ε-decay
`0.9999`, learning rate `0.3`, discount `0.9`) it runs in a few minutes
and produces a comparison table plus `training_curves.png`.

### Empirical results on the full complex env

After 20,000 training episodes (defaults: 8 floors, 2 robots, 2 elevators,
stochastic delays + breakdowns, 6-task budget per episode, 300 max steps):

| Agent          | Avg reward | Mean deliveries / episode | Q-table size |
|----------------|------------|---------------------------|--------------|
| `MultiAgent-Q` | **-340**   | **0.27**                  | ~50k obs (~9 MB) |
| `Random`       | -1260      | 0.11                      | n/a          |

The trained policy is ~2.4× better than random in deliveries per episode
and ~73% better in reward. Full-budget completion (6/6 tasks delivered
within 300 steps) is rare under defaults — that bar is intentionally
unforgiving so the metric scales with future improvements (more training,
reward shaping, DQN, …).

### Action selection matters more than the seed

Evaluating the trained table (`evaluate_saved.py --env complex`, 50k-episode
run, 500 max steps, held-out seed 123) reveals that **how you query the
policy dominates the result**:

| Mode                  | Success % | Avg reward |
|-----------------------|-----------|------------|
| greedy (argmax-Q)     | 11%       | -1174      |
| ε-greedy (ε=0.37)     | 80%       | -283       |
| **softmax (temp≈2)**  | **93%**   | **-73**    |
| Random baseline       | 0%        | -2074      |

Greedy collapses while softmax sampling reaches 93%. This is **not** an
overfitting/generalization gap: holding the mode fixed, the train seed (42)
and held-out seed (123) score almost identically (greedy 14% vs 11%;
ε-greedy 79% vs 80%), and 0% of observations are unseen at evaluation. The
cause is the **POMDP**: under partial observation a memoryless policy suffers
perceptual aliasing, where one observation maps to several true states, so a
deterministic greedy policy gets trapped in stall/oscillation loops. A
*stochastic* policy (ε-greedy, and especially Q-weighted softmax) hedges
across the aliased states and escapes them. By contrast, on the fully-observed
`SimpleBuildingEnv`, greedy is already optimal and stochasticity only hurts —
exactly the MDP-vs-POMDP distinction. Run `evaluate_saved.py` on both envs to
reproduce.

### POMDP caveat

Per-robot observation hides things the full state knows: the other
robots' positions and intentions, the other elevators' detailed status
beyond floor/broken, the full task queue. A robot acting only on its own
observation can't *guarantee* optimal team behavior — two robots might
both head for the same elevator, etc. In practice this works fine for
this env because:

- The shared Q-table absorbs both robots' experience together.
- The observation includes both elevators (so robots can prefer the
  closer/working one).
- The task signal (`pickup_target` / `delivery_target`) gives each robot
  a coherent local objective.

If you need provably-optimal multi-agent coordination, the right fix is
function approximation with a centralized critic — out of scope here.

### Try the demo

```bash
python demo_complex_env.py
```

A quick standalone script that trains a vanilla single-agent
`QLearningAgent` against `Random` on the complex env in **full-state**
mode. Expected outcome: both ~0% success — the demo's whole purpose is to
show *why* `obs_mode='per_robot'` + `MultiAgentTabularQ` exist.

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
  DQN agent or a new env later won't require pipeline changes.
- **Protocol-based typing** (no ABCs) — structural typing keeps the contract
  loose and inheritance-free.
- **Tabular Q-table** as `dict[obs → np.ndarray]` — lazy zero-init on unseen
  observations. Shape unchanged across single- and multi-agent variants.
- **Multi-agent decomposition without inheritance** — `MultiAgentTabularQ`
  is a separate class that happens to satisfy the same protocol as
  `QLearningAgent`. No shared base class. New agent types (DQN, COMA, …)
  can join the same way.
- **Observation projection lives on the env, not the agent** — `obs_mode`
  is a config knob on `ComplexBuildingEnv`. Agents stay simple; the env
  decides what each agent can see. This makes it easy to A/B-test
  different observation designs without touching agent code.

---

## AI Disclaimer

This project was developed with the assistance of [Claude Code](https://claude.com/claude-code),
Anthropic's agentic coding tool. It was used to help with implementation,
debugging, running experiments, and documentation. All design decisions,
results, and conclusions were reviewed and validated by the author.
