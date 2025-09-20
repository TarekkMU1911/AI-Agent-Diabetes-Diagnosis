# imports and libraries
import os
import json
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset



raw_path = "Datasets/diabetes_unified.jsonl"
ft_path = "Datasets/fine-tuning/ftData.jsonl"
tokenized_path = "Datasets/fine-tuning/tokenized/tokenizedData.jsonl"
data = pd.read_json(raw_path, lines=True)



ft_data = [
    {
      "instruction": "Summarize the patient's health metrics for the given day.",
        "input": row["input"],
        "output": row["output"]}
    
    for _, row in data.iterrows()
]





# tokenization
model_id = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

with open(ft_path, "r", encoding="utf-8") as f_in, open(tokenized_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        row = json.loads(line)
        text = row["instruction"].strip() + "\n" + row["input"].strip()
        tokens = tokenizer(text, truncation=False, padding=False, return_attention_mask=True)

        record = {
            "instruction": row["instruction"],
            "input": row["input"],
            "text": text,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")





dataset = load_dataset("json", data_files=tokenized_path, split="train")



def tokenize_fn(batch):
    text = batch["instruction"] + "\n" + batch["input"]
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

sampled = random.sample(range(len(dataset)), min(50000, len(dataset)))
tokenized_samples = [tokenize_fn(dataset[i]) for i in sampled]

class FineTuneDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = {k: torch.tensor(v) for k, v in self.samples[idx].items() if k in ["input_ids", "attention_mask"]}
        s["labels"] = s["input_ids"].clone()
        return s

train_loader = DataLoader(FineTuneDataset(tokenized_samples), batch_size=1, shuffle=True)


# fine-tuning

device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device)
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()

model.save_pretrained("GPTneo-finetuned")
tokenizer.save_pretrained("GPTneo-finetuned")
