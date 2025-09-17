# AI Agent Diabetes Diagnosis

An AI-powered system for diabetes diagnosis and patient data analysis using **LLaMA**, **Mistral**, and **BioGPT**.

## Features

* Multi-Model Support: BioGPT, BioGPT-PubMedQA, LLaMA, Mistral-7B
* Fine-Tuned Models: Custom diabetes-focused models
* Web Interface: Gradio UI
* API Backend: FastAPI server (LLaMA only)
* Vector Search: Pinecone for knowledge retrieval
* Medical Text Processing: Patient data summarization & risk assessment

## Installation

```bash
pip install torch torchvision torchaudio transformers datasets peft
pip install gradio fastapi uvicorn sentence-transformers pinecone-client python-dotenv requests
```

Create a `.env` file with your API keys:

```env
HF_TOKEN=your_huggingface_token
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
```

---

## Model Usage

### LLaMA Fine-Tuned (API Backend)

* The FastAPI server serves only the **LLaMA fine-tuned model**.
* Run the server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

* Example API call:

```python
import requests

response = requests.post("http://localhost:8000/generate",
    json={"instruction": "Analyze patient glucose levels",
          "input": "Glucose readings: 120, 140, 180, 200 mg/dL"})
print(response.json()["response"])
```

### Mistral Fine-Tuned (Local / Hugging Face Space)

* Load locally or use the [Hugging Face Space](https://huggingface.co/spaces/Reham1/mistral-diabetes-app)
* Local inference script available in `scripts/mistral_inference.py`

---
## Contributors

* **Tarek Muhammed** – [@TarekkMU1911](https://github.com/TarekkMU1911)
* **Reham Hassan** – [@RehamHassan1](https://github.com/RehamHassan1)
* **Farida El Shenawy** – [@Farida-EL-Shenawy](https://github.com/Farida-EL-Shenawy)
* **Rana Helal** – [@ranaehelal](https://github.com/ranaehelal)


