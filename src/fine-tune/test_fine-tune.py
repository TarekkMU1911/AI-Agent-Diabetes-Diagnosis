import json
import os
import torch
from datasets import load_dataset, Dataset
from transformers import LlamaTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Paths
model_dir = "./fine_tuned_llama_test"
base_model = "openlm-research/open_llama_7b"

# 2. Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

# 3. Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 4. Attach LoRA adapters
model = PeftModel.from_pretrained(model, model_dir)
model.eval()

# 5. Load some test data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_file = os.path.join(BASE_DIR, "Datasets", "diabetes_unified.json")

try:
    with open(data_file, "r") as f:
        data = json.load(f)
    if isinstance(data, list) and isinstance(data[0], dict):
        dataset = Dataset.from_list(data)
    else:
        raise ValueError
except Exception:
    dataset = load_dataset("json", data_files=data_file)["train"]

test_dataset = dataset.select(range(5))

def make_prompt(entry):
    instruction = entry["instruction"]
    input_text = entry["input"]
    return f"Instruction: {instruction}\nInput: {input_text}\nOutput:"

# 6. Run inference
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
    print(entry["output"])
