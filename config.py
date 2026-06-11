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

Q_TABLE_PATH = "q_table.pkl"
TRAINING_STATS_PATH = "training_stats.csv"
