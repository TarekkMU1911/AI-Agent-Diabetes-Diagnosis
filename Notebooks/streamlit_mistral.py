import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
FINETUNED_MODEL = "Reham1/mistral-diabetes-lora"

st.title("🩺 Diabetes Diagnosis AI Agent")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
    return tokenizer, model

tokenizer, model = load_model()

user_input = st.text_area("Enter patient data:")

if st.button("Predict"):
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success(f"Diagnosis: {result}")
