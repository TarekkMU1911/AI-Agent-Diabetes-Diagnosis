from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import MODEL_NAME, TRAIN_PARAMS

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=TRAIN_PARAMS["dtype"],
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    return get_peft_model(model, lora_config)
