from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
FINETUNED_MODEL = "https://huggingface.co/Reham1/mistral-diabetes-lora"
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
