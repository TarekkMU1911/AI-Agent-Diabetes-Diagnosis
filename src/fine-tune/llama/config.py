import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE_DIR, "Datasets", "diabetes_unified.json")

MODEL_NAME = "openlm-research/open_llama_7b"
OUTPUT_DIR = "./fine_tuned_llama_test"

TRAIN_PARAMS = {
    "max_length": 512,
    "batch_size": 2,
    "grad_accum": 4,
    "epochs": 1,
    "lr": 2e-4,
    "warmup": 50,
    "save_steps": 100,
    "logging_steps": 20,
    "dtype": torch.float16
}
