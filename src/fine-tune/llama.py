import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

data_file = "Datasets/diabetes_unified.json"

try:
    with open(data_file, "r") as f:
        data = json.load(f)
    if isinstance(data, list) and isinstance(data[0], dict):
        dataset = Dataset.from_list(data)
    else:
        raise ValueError
except Exception:
    dataset = load_dataset("json", data_files=data_file)["train"]

print("Sample entry:", dataset[0])

model_name = "openlm-research/open_llama_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Tokenize function ---
def tokenize(batch):
    inputs = [
        f"Instruction: {instr}\nInput: {inp}\nOutput: {out}"
        for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    return tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# --- Load model and apply LoRA ---
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, lora_config)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=100,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()


model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")

print("Fine-tuning completed! Model saved at './fine_tuned_llama'")
