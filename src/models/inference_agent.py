from src.models.load_models import load_mistral, load_llama2, load_biogpt

mistral = load_mistral()
llama = load_llama2()
biogpt = load_biogpt()

def query_mistral(patient_data: str):
    prompt = f"""[INST] Based on patient data, provide symptoms and medical advice:\n{patient_data} [/INST]"""
    return mistral(prompt, max_new_tokens=128)[0]['generated_text']

def query_llama(patient_data: str):
    prompt = f"""[INST] Analyze this patient and predict risk levels with reasoning:\n{patient_data} [/INST]"""
    return llama(prompt, max_new_tokens=128)[0]['generated_text']

def query_biogpt(question: str):
    # BioGPT is an InferenceClient, so we call text_generation()
    return biogpt.text_generation(question, max_new_tokens=128)
