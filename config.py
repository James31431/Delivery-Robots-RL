"""Central configuration for training/evaluation runs.

These values are tuned for the per-robot partial-observation pipeline
(``--env complex`` with MultiAgentTabularQ). The simple env pipeline still
works fine with them too — it's smaller and easily converges within this
episode budget.
"""

NUM_TRAINING_EPISODES = 50_000
NUM_EVALUATION_EPISODES = 100
MAX_STEPS = 50  # used by the SimpleBuildingEnv pipeline; complex pipeline
                # overrides via ComplexEnvConfig(max_steps=...)

TRAIN_SEED = 42
EVAL_SEED = 123

LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.90
EPSILON = 1.0
EPSILON_DECAY = 0.99998
MIN_EPSILON = 0.2

# Artifacts are namespaced per environment so the simple and complex
# pipelines don't overwrite each other (e.g. q_table_simple.pkl vs
# q_table_complex.pkl).
def q_table_path(env_name: str) -> str:
    return f"q_table_{env_name}.pkl"


def training_stats_path(env_name: str) -> str:
    return f"training_stats_{env_name}.csv"


def training_curves_path(env_name: str) -> str:
    return f"training_curves_{env_name}.png"
