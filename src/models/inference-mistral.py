import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
OUTPUT_DIR = "./mistral-diabetes-lora"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

# generate function 
def generate(prompt: str, max_new_tokens: int = 256):
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

# clean function 
def clean_answer(raw_answer: str) -> str:
    """
    Answer STRICTLY with only one word: "Yes" OR "No" OR "Unknown".
    Do NOT output anything else.
    """
    for token in ["Yes", "No", "Unknown"]:
        if token.lower() in raw_answer.lower():
            return token
    return "Unknown" 

# test 
test_prompt = """[INST] Answer STRICTLY with only one word: "Yes" OR "No" OR "Unknown".
Do NOT output anything else.

{ 'glucose': { 'avg': 170, 'min': 60, 'max': 300 }, 'steps_total': 3500, 'heart_rate_avg': 92 } [/INST]"""

raw = generate(test_prompt)
print("🔎 Raw model output:\n", raw)
print("✅ Cleaned answer:", clean_answer(raw))
