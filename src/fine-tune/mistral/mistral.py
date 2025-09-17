import os, json, random
from typing import List, Dict, Any
from dotenv import load_dotenv
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Datasets", "diabetes_unified.jsonl")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./mistral-diabetes-lora"
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
SEED = 42
SAMPLE_SIZE = 40000
MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 16
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LOG_STEPS = 10
SAVE_STEPS = 1000
EVAL_RATIO = 0.05
# Utilities
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl_sample(path: str, n: int, seed: int) -> List[Dict[str, Any]]:
    sample = []
    rng = random.Random(seed)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(sample) < n:
                sample.append(obj)
            else:
                j = rng.randint(1, i)
                if j <= n:
                    sample[j - 1] = obj
    rng.shuffle(sample)
    return sample


def guess_text(ex: Dict[str, Any]) -> str:
    instr_keys = ["instruction", "prompt", "task", "question"]
    input_keys = ["input", "context", "details"]
    output_keys = ["output", "response", "answer", "label", "target"]

    def first_key(keys):
        for k in keys:
            if k in ex and isinstance(ex[k], (str, int, float)):
                return k
        return None

    ik = first_key(instr_keys)
    ok = first_key(output_keys)
    ck = first_key(input_keys)

    # fallback: try to detect label-like field
    if not ik and isinstance(ex, dict):
        label_like = None
        for k in ["diagnosis", "risk", "class", "category", "readout", "y", "target", "label", "answer", "output"]:
            if k in ex and isinstance(ex[k], (str, int, float)):
                label_like = k
                break
        context_str = json.dumps({k: v for k, v in ex.items() if k != label_like}, ensure_ascii=False)
        if label_like:
            return f"""[INST] Based on the patient data below, provide the correct output.\n\nPatient Data:\n{context_str}\n\nAnswer: [/INST] {ex[label_like]}"""
        else:
            return f"""[INST] Summarize the key insights from the following patient data:\n\n{context_str}\n\nSummary: [/INST]"""

    instruction = ex.get(ik, "You are a helpful AI assistant.")
    inp = ex.get(ck, None)
    out = ex.get(ok, None)

    if inp and out is not None:
        return f"[INST] {instruction}\n\nInput:\n{inp} [/INST] {out}"
    elif out is not None:
        return f"[INST] {instruction} [/INST] {out}"
    elif inp:
        return f"[INST] {instruction}\n\nInput:\n{inp} [/INST]"
    else:
        dumped = json.dumps(ex, ensure_ascii=False)
        return f"[INST] Read and summarize the following JSON record.\n\n{dumped}\n\nSummary: [/INST]"

# Load data
set_seed(SEED)
records = read_jsonl_sample(DATA_PATH, SAMPLE_SIZE, SEED)
texts = [guess_text(r) for r in records]

eval_size = max(1, int(len(texts) * EVAL_RATIO))
train_texts = texts[eval_size:]
eval_texts = texts[:eval_size]

train_ds = Dataset.from_dict({"text": train_texts})
eval_ds = Dataset.from_dict({"text": eval_texts})

# Tokenizer & Model (QLoRA)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None,
    )

train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Training
training_args = TrainingArguments(
    output_dir="./mistral-lora-test",
    per_device_train_batch_size=2,    
    gradient_accumulation_steps=4,   
    warmup_steps=50,
    num_train_epochs=1,                
    learning_rate=2e-4,
    fp16=True,                        
    save_strategy="steps",
    save_steps=100,
    logging_steps=20,
    report_to="none",                 
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Training complete. Adapter saved to:", OUTPUT_DIR)

# Quick inference test
def generate(prompt: str, max_new_tokens: int = 256):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


test_prompt = "[INST] Based on the patient data, predict the likely diagnosis:\n\n{ 'glucose': { 'avg': 170, 'min': 60, 'max': 300 }, 'steps_total': 3500, 'heart_rate_avg': 92 } [/INST]"
print("\n🔎 Sample generation:\n", generate(test_prompt))
