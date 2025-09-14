
# AI Agent Diabetes Diagnosis

An AI-powered system for diabetes diagnosis and patient data analysis using multiple language models, including LLaMA, Mistral, and BioGPT.

## Features

* **Multi-Model Support**: BioGPT, BioGPT-PubMedQA, LLaMA, Mistral-7B
* **Fine-Tuned Models**: Custom diabetes-focused models
* **Web Interface**: Gradio-based UI
* **API Backend**: FastAPI server
* **Vector Search**: Pinecone integration for medical knowledge retrieval
* **Medical Text Processing**: Patient data summarization and risk assessment

## Installation

```bash
# Python 3.8+
pip install torch torchvision torchaudio transformers datasets peft
pip install gradio fastapi uvicorn sentence-transformers pinecone-client python-dotenv requests
```

Create a `.env` file with your API keys:

```env
HF_TOKEN=your_huggingface_token
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
```

## Model Setup & Inference

### Load Fine-Tuned Mistral Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
FINETUNED_MODEL = "https://huggingface.co/Reham1/mistral-diabetes-lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

# Load LoRA fine-tuned model
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
```

### Generate Predictions

```python
prompt = """[INST] Based on patient data, assess diabetes risk:

Patient Data: {
    'glucose': {'avg': 170, 'min': 60, 'max': 300},
    'steps_total': 3500,
    'heart_rate_avg': 92,
    'age': 45,
    'bmi': 28.5
} [/INST]"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

> You can also try the model online via the [Hugging Face Space](https://huggingface.co/spaces/Reham1/mistral-diabetes-app).

## Usage

### Start FastAPI Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Run Web Interface

```bash
python app.py   # Multi-model interface
python ui.py    # Simple interface
```

### API Example

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "instruction": "Analyze patient glucose levels",
        "input": "Glucose readings: 120, 140, 180, 200 mg/dL over 4 hours"
    }
)
print(response.json()["response"])
```

## Contributors

* **Tarek Muhammed** – [@TarekkMU1911](https://github.com/TarekkMU1911)
* **Reham Hassan** – [@RehamHassan1](https://github.com/RehamHassan1)
* **Farida El Shenawy** – [@Farida-EL-Shenawy](https://github.com/Farida-EL-Shenawy)
* **Rana Helal** – [@ranaehelal](https://github.com/ranaehelal)

