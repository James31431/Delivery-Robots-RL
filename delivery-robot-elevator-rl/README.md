# Delivery Robot Elevator RL (Starter)

A minimal **tabular Q-learning** starter for a delivery robot that must use an
elevator to move between floors in a multi-floor building.

This is the **scratch / starter version** — not the final polished system.

## Project Goal

Train a single delivery robot in a small simulated building to learn how to
pick up an elevator, ride it to the correct floor, exit, and deliver a package.
The agent learns purely from reward signals using tabular Q-learning (no neural
networks).

## Simple Environment Assumptions

- A single building with **5 floors**, numbered `0` to `4`.
- **One robot**, starting at floor `0`.
- **One elevator**, starting at floor `0`.
- **One delivery task** per episode.
- Target floor is randomly chosen from floors `1..4` on each reset.
- The robot must exit the elevator before delivering the package.
- Episode ends when the package is delivered OR `max_steps` (default `50`) is reached.

## State Representation

State is a 5-tuple of integers:

```
(robot_floor, target_floor, elevator_floor, robot_inside_elevator, delivered)
```

- `robot_floor`: `0..4`
- `target_floor`: `1..4`
- `elevator_floor`: `0..4`
- `robot_inside_elevator`: `0` or `1`
- `delivered`: `0` or `1`

## Action Space

| ID | Action            | Description                                                   |
|----|-------------------|---------------------------------------------------------------|
| 0  | `wait`            | Do nothing for this step                                      |
| 1  | `enter_elevator`  | Robot enters elevator (requires same floor & not already in)  |
| 2  | `exit_elevator`   | Robot exits elevator (requires being inside)                  |
| 3  | `elevator_up`     | Elevator moves up one floor (cannot exceed floor 4)           |
| 4  | `elevator_down`   | Elevator moves down one floor (cannot go below floor 0)       |
| 5  | `deliver_package` | Deliver at current floor (must match target, must be outside) |

## Reward Design

| Event                     | Reward |
|---------------------------|--------|
| Successful delivery       | +100   |
| Each normal step          |  -1    |
| Invalid action            |  -5    |
| `wait` action             |  -2    |

Episode terminates on a successful delivery OR when `max_steps` is reached.

## Project Layout

```
delivery-robot-elevator-rl/
├── README.md
├── requirements.txt
├── main.py
├── environment/
│   ├── __init__.py
│   └── simple_env.py
├── rl/
│   ├── __init__.py
│   ├── q_learning_agent.py
│   ├── train.py
│   └── evaluate.py
└── tests/
    ├── __init__.py
    └── test_simple_env.py
```

## How to Run

From the `delivery-robot-elevator-rl/` directory:

```bash
# 1. (Optional) create a venv
python -m venv .venv
source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Train + evaluate + save Q-table
python main.py

# 4. Run tests
pytest
```

After running `main.py`, the learned Q-table is saved to `q_table.pkl`.
