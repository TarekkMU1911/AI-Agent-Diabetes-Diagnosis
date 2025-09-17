import gradio as gr
import requests
import os

# API endpoint
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/generate")

def generate_text(instruction, input_text):
    try:
        # send req to api
        response = requests.post(
            API_URL,
            json={"instruction": instruction, "input": input_text},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("response", "No response from API")
    except Exception as e:
        return f" Request failed: {str(e)}"

# UI
diabetes_assistant_ui = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Instruction",
            placeholder="Example: Summarize this medical research paper in simple terms.",
            lines=2
        ),
        gr.Textbox(
            label="Input",
            placeholder="Effect of novel diabetes drug XYZ on blood sugar control ...",
            lines=4
        )
    ],
    outputs=gr.Textbox(
        label="Model Output",
        lines=10
    ),
    title="🩺 Diabetes Assistant",
    description="A simple interface to interact with the fine-tuned medical model API."
)

if __name__ == "__main__":
    diabetes_assistant_ui.launch(server_name="0.0.0.0", server_port=7861)