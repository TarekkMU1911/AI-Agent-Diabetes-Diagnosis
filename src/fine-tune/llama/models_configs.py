import gradio as gr
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BioGptTokenizer, BioGptForCausalLM
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.environ.get("HF_TOKEN")

# Configs
MODELS = {
    "BioGPT": {"type": "local", "name": "microsoft/biogpt"},
    "BioGPT-PubMedQA": {"type": "local", "name": "microsoft/BioGPT-Large-PubMedQA"},
    "LLaMA Fine-Tuned (API)": {"type": "api", "url": "http://127.0.0.1:8000/generate"}
}
