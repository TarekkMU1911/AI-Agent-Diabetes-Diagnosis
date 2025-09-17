import os
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.environ.get("HF_TOKEN")

MODELS = {
    "BioGPT": {"type": "local", "name": "microsoft/biogpt"},
    "BioGPT-PubMedQA": {"type": "local", "name": "microsoft/BioGPT-Large-PubMedQA"},
    "LLaMA Fine-Tuned (API)": {"type": "api", "url": "http://127.0.0.1:8000/generate"}
}

pipelines = {}
