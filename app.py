import streamlit as st
import src.models.inference_agent as agent

st.set_page_config(page_title="Diabetes Diagnosis AI", layout="wide")
st.title("🩺 Diabetes Diagnosis Assistant (LLaMA2 + Mistral + BioGPT)")

st.markdown("""
Upload patient data or write below. The assistant will:
- Analyze risk (LLaMA2)
- Suggest treatment (Mistral)
- Provide explanations (BioGPT)
""")

patient_input = st.text_area("Enter patient data or prescription text", height=250)

if st.button("🧠 Diagnose Patient"):
    with st.spinner("Running Mistral..."):
        mistral_out = agent.query_mistral(patient_input)

    with st.spinner("Running LLaMA2..."):
        llama_out = agent.query_llama(patient_input)

    with st.spinner("Running BioGPT..."):
        biogpt_out = agent.query_biogpt("Explain diabetes risks for: " + patient_input)

    st.success("Results Ready!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌪️ Mistral (Suggestions)")
        st.write(mistral_out)

    with col2:
        st.subheader("🦙 LLaMA2 (Risk Analysis)")
        st.write(llama_out)

    with col3:
        st.subheader("🧬 BioGPT (Medical Knowledge)")
        st.write(biogpt_out)