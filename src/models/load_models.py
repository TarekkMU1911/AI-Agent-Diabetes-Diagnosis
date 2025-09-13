from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.pipelines import pipeline 
from pathlib import Path
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# === Load environment variables ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# === Check for GPU ===
USE_CUDA = torch.cuda.is_available()

# === Optional 4-bit quantization config (only useful with GPU) ===
BNB_CONFIG = (
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    if USE_CUDA else None
)

# === Universal pipeline loader ===
def _make_pipeline(model_id, use_token=False, local=False):
    if local:
        model_path = Path(model_id).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Local model path not found: {model_path}")
    else:
        model_path = model_id

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_auth_token=HF_TOKEN if use_token else None,
        local_files_only=local
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        use_auth_token=HF_TOKEN if use_token else None,
        local_files_only=local,
        device_map="auto" if USE_CUDA else None,
        torch_dtype=torch.float16 if USE_CUDA else torch.float32,
        quantization_config=BNB_CONFIG
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Individual loaders ===

def load_mistral():
    """
    Load fine-tuned Mistral model from Hugging Face.
    Requires valid HF_TOKEN in .env.
    """
    return _make_pipeline("Reham1/mistral-diabetes-lora1", use_token=True)

def load_biogpt():
    return InferenceClient(model="microsoft/BioGPT-Large", token=HF_TOKEN)

def load_llama2():
    model_path = "/teamspace/studios/this_studio/AI-Agent-Diabetes-Diagnosis/fine_tuned_llama_test"  
    # path to your cloned repo folder

    tok = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True   
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True   
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok
    )
    return gen