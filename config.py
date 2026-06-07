"""Central configuration for training/evaluation runs."""

NUM_TRAINING_EPISODES = 1_000_000
NUM_EVALUATION_EPISODES = 100
MAX_STEPS = 50  # used by the SimpleBuildingEnv pipeline; complex pipeline
                # overrides via ComplexEnvConfig(max_steps=...)

TRAIN_SEED = 42
EVAL_SEED = 123

LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.90
EPSILON = 1.0
EPSILON_DECAY = 0.999_999
MIN_EPSILON = 0.05

Q_TABLE_PATH = "q_table.pkl"
TRAINING_STATS_PATH = "training_stats.csv"
