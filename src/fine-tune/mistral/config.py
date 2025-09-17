import os

# Randomization / reproducibility
SEED = 42

# Data
SAMPLE_SIZE = 40000
MAX_LENGTH = 512
EVAL_RATIO = 0.05

# Training
BATCH_SIZE = 2
GRAD_ACCUM = 16
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LOG_STEPS = 10
SAVE_STEPS = 1000

# Model
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./mistral-diabetes-lora"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Datasets", "diabetes_unified.jsonl")
