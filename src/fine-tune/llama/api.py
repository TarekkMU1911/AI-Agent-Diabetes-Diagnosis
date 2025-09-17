import os
import torch
from fastapi import FastAPI, Request
from transformers import LlamaTokenizer, AutoModelForCausalLM

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
model_dir = os.path.join(repo_root, "fine_tuned_llama_test")

# convert to tokens
tokenizer = LlamaTokenizer.from_pretrained(model_dir, legacy=True)

# get the fine tuned model from the dir
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto", # in gpu if there
    torch_dtype=torch.float16 # half of the model precesion
)
# add padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#  API
app = FastAPI()

@app.post("/generate")
async def generate_text(request: Request):
    data = await request.json()
    instruction = data.get("instruction", "")
    input_text = data.get("input", "")

    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"


    # convert input to tensor , for the model to understand
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # take the output only
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Output:" in full_response:
        clean_response = full_response.split("Output:")[-1].strip()
    else:
        clean_response = full_response

    return {"response": clean_response}

