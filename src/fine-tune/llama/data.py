import json
from datasets import Dataset, load_dataset
from transformers import LlamaTokenizer
from config import DATA_FILE, MODEL_NAME, TRAIN_PARAMS

def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and isinstance(data[0], dict):
            dataset = Dataset.from_list(data)
        else:
            raise ValueError
    except Exception:
        dataset = load_dataset("json", data_files=DATA_FILE)["train"]

    dataset = dataset.shuffle(seed=42).select(range(30000))
    print(f"Dataset size: {len(dataset)}")
    return dataset

def get_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, legacy=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        inputs = [
            f"Instruction: {instr}\nInput: {inp}\nOutput: {out}"
            for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
        ]
        tokenized = tokenizer(inputs, truncation=True, padding="max_length", max_length=TRAIN_PARAMS["max_length"])
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(tokenize, batched=True)
