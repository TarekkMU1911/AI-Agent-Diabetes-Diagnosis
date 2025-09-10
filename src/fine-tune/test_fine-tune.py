import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import LlamaTokenizer, AutoModelForCausalLM

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
model_dir = os.path.join(repo_root, "fine_tuned_llama_test")
data_file = os.path.join(repo_root, "Datasets", "diabetes_unified.json")

# --- Load tokenizer and model ---
tokenizer = LlamaTokenizer.from_pretrained(model_dir, legacy=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)
model.eval()

# Handle missing pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load dataset ---
try:
    with open(data_file, "r") as f:
        data = json.load(f)
    if isinstance(data, list) and isinstance(data[0], dict):
        dataset = Dataset.from_list(data)
    else:
        raise ValueError
except Exception:
    dataset = load_dataset("json", data_files=data_file)["train"]

# --- Select a few examples for testing ---
test_dataset = dataset.select(range(5))

def make_prompt(entry):
    instruction = entry.get("instruction", "")
    input_text = entry.get("input", "")
    return f"Instruction: {instruction}\nInput: {input_text}\nOutput:"

# --- Generate outputs for test dataset ---
for i, entry in enumerate(test_dataset):
    prompt = make_prompt(entry)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\n=== Example {i+1} ===")
    print("Prompt:")
    print(prompt)
    print("\nGenerated:")
    print(generated_text)
    print("\nExpected Output:")
    print(entry.get("output", "N/A"))
    print("-" * 50)

# --- Optional: Additional fixed prompts ---
additional_prompts = [
    "Instruction: Summarize the patient's daily health metrics.\nInput: Patient: HUPA9999 | Date: 2021-08-15\nGlucose (avg/min/max): 120/60/200, Heart rate avg: 80, Steps: 15000, Carbs: 10, Insulin: 6, Events: Meals=3, Activities=100, Hypoglycemia=1, Hyperglycemia=5\nOutput:",
    "Instruction: Suggest a healthy daily routine for someone who wants to lose weight safely.\nInput:\nOutput:",
    "Instruction: Is this patient at risk of cardiovascular disease? Answer 0 = No risk, 1 = At risk.\nInput: 65-year-old female, BMI 28, HbA1c 7.2, glucose 140, history of smoking and hypertension.\nOutput:",
    "Instruction: Is this patient at risk of cardiovascular disease? Answer 0 = No risk, 1 = At risk.\nInput: 55-year-old male, BMI 24, HbA1c 6.8, glucose 120, no smoking history, no hypertension.\nOutput:",
    "Instruction: Summarize this medical research paper in simple terms.\nInput: Effect of novel diabetes drug XYZ on blood sugar control in type 2 diabetes patients.\nOutput:"
]

for prompt in additional_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-" * 50)
