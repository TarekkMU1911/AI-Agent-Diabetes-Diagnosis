from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

API_TOKEN = os.environ.get("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(
    "Reham1/mistral-diabetes-lora1", 
    token=API_TOKEN  
)

model = AutoModelForCausalLM.from_pretrained(
    "Reham1/mistral-diabetes-lora1",
    token=API_TOKEN,
    device_map="auto",  
    dtype="auto"        
)

generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer
)

prompt = """[INST] Based on patient data, provide possible symptoms and 3 short medical advice suggestions.

{ 'glucose': { 'avg': 170, 'min': 60, 'max': 300 }, 'steps_total': 3500, 'heart_rate_avg': 92 } [/INST]"""

output = generator(prompt, max_new_tokens=128)
print(output[0]["generated_text"])
