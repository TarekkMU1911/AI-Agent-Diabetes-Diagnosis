from transformers import TrainingArguments, Trainer
from config import OUTPUT_DIR, TRAIN_PARAMS

def train_model(model, tokenized_dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_PARAMS["batch_size"],
        gradient_accumulation_steps=TRAIN_PARAMS["grad_accum"],
        warmup_steps=TRAIN_PARAMS["warmup"],
        num_train_epochs=TRAIN_PARAMS["epochs"],
        learning_rate=TRAIN_PARAMS["lr"],
        fp16=True,
        save_strategy="steps",
        save_steps=TRAIN_PARAMS["save_steps"],
        logging_steps=TRAIN_PARAMS["logging_steps"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()
    return model
