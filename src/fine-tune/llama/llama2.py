import json
from datasets import Dataset, load_dataset
from transformers import LlamaTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_file = os.path.join(BASE_DIR, "Datasets", "diabetes_unified.json")

# Load dataset
try:
    with open(data_file, "r") as f:
        data = json.load(f)
    if isinstance(data, list) and isinstance(data[0], dict):
        dataset = Dataset.from_list(data)  # convert list of dicts to Dataset object
    else:
        raise ValueError
except Exception:
    dataset = load_dataset("json", data_files=data_file)["train"]

# Shuffle dataset and select subset
dataset = dataset.shuffle(seed=42).select(range(30000))
print(f"Shuffled + selected dataset size: {len(dataset)}")
print("Sample entry:", dataset[0])

# Tokenizer
model_name = "openlm-research/open_llama_7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # use eos_token as padding

def tokenize(batch):
    inputs = [
        f"Instruction: {instr}\nInput: {inp}\nOutput: {out}"
        for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    tokenized = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Tokenization on all data
tokenized_dataset = dataset.map(tokenize, batched=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # query and value
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama_test",
    per_device_train_batch_size=2,  # for each GPU
    gradient_accumulation_steps=4,
    warmup_steps=50,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,  # to save memory
    save_strategy="steps",
    save_steps=100,  # save every 100 steps (checkpoint)
    logging_steps=20,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# Save model and tokenizer
model.save_pretrained("./fine_tuned_llama_test")
tokenizer.save_pretrained("./fine_tuned_llama_test")

print("Test fine-tuning completed!")
