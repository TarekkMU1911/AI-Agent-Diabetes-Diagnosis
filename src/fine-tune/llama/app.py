import gradio as gr
import requests
import torch
from pinecone_plugins.assistant.models.assistant_model import MODELS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BioGptTokenizer, BioGptForCausalLM
from models_configs import API_TOKEN, pipelines



def load_biogpt():

    if "BioGPT" in pipelines:
        return pipelines["BioGPT"]
    print(" Loading BioGPT ...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt", token=API_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/biogpt",
        token=API_TOKEN,
        device_map="auto",
        torch_dtype="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipelines["BioGPT"] = pipe
    return pipe

def load_biogpt_pubmedqa():

    if "BioGPT-PubMedQA" in pipelines:
        return pipelines["BioGPT-PubMedQA"]
    print(" Loading BioGPT-Large-PubMedQA ...")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
    model = BioGptForCausalLM.from_pretrained(
        "microsoft/BioGPT-Large-PubMedQA",
        device_map="auto",
        torch_dtype="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipelines["BioGPT-PubMedQA"] = pipe
    return pipe


def generate_text(model_name, instruction, input_text):
    try:
        model_info = MODELS[model_name]

        if model_info["type"] == "local":
            if model_name == "BioGPT":
                pipe = load_biogpt()
            elif model_name == "BioGPT-PubMedQA":
                pipe = load_biogpt_pubmedqa()
            prompt = f"{instruction}\n{input_text}"
            output = pipe(prompt, max_new_tokens=256, do_sample=True)
            result = output[0]["generated_text"]
            cleaned_result = result.replace(prompt, "").strip()
            return cleaned_result

        elif model_info["type"] == "api":
            response = requests.post(
                model_info["url"],
                json={"instruction": instruction, "input": input_text},
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "No response from API")

    except Exception as e:
        return f"Error: {str(e)}"

#UI
diabetes_assistant_ui = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Dropdown(choices=list(MODELS.keys()), value="BioGPT", label="Choose a Model"),
        gr.Textbox(label="Instruction", placeholder="Example: Summarize this paper...", lines=2),
        gr.Textbox(label="Input", placeholder="Patient data or research text...", lines=4),
    ],
    outputs=gr.Textbox(label="Model Output", lines=12),
    title="🩺 Multi-Model Medical Assistant",
    description="Select between BioGPT (local), BioGPT-PubMedQA (local) or LLaMA (API)."
)

if __name__ == "__main__":
    diabetes_assistant_ui.launch(server_name="0.0.0.0", server_port=7863)
