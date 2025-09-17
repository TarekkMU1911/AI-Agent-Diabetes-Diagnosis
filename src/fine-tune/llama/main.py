from data import load_data, get_tokenizer, tokenize_dataset
from model import load_model
from train import train_model
from config import OUTPUT_DIR

if __name__ == "__main__":
    dataset = load_data()
    tokenizer = get_tokenizer()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    model = load_model()
    model = train_model(model, tokenized_dataset)

    # Save outputs
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Fine-tuning completed and model saved!")
