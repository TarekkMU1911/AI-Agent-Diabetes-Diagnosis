---
base_model: openlm-research/open_llama_7b
library_name: peft
tags:
- base_model:adapter:openlm-research/open_llama_7b
- lora
- transformers
- diabetes
- medical
---

# Model Card for Fine-Tuned LLaMA Diabetes Assistant

## Model Details

### Model Description

This model is a **LoRA fine-tuned version of OpenLLaMA-7B** for medical and healthcare-related tasks, specifically focusing on **diabetes** data.  
It can summarize research articles, analyze patient health metrics, classify clinical risk, and answer questions from EHR-style data.

- **Developed by:** Ranaehelal  
- **Base model:** `openlm-research/open_llama_7b`  
- **Model type:** Causal Language Model with LoRA adapters  
- **Language(s):** English  
- **License:** Same license as base model (OpenLLaMA license)  
- **Repository:** GitHub (see below)  

---

## Model Sources

- **Repository:** https://github.com/TarekkMU1911/AI-Agent-Diabetes-Diagnosis  
- **Base model: OpenLLaMA-7B**  
- **Demo / Usage:** (not yet published)  

---

## Uses

### Direct Use
- Summarizing diabetes-related research.  
- Converting daily health / glucose metrics into readable summaries.  
- Predicting readmission from EHR-like patient data.  
- Basic risk classification tasks.  

### Out-of-Scope Use
- Not a substitute for medical advice.  
- Should not be used for real‐life clinical decision making without expert oversight.  

---

## Training Details

- **Dataset:** Unified diabetes dataset (EHR, abstracts, health logs, risk labels)  
- **Preprocessing:** Instruction-Input-Output formatting; dataset shuffled; subset selected (e.g. 30,000 examples)  
- **Tokenizer:** LLaMA tokenizer; EOS token used as pad if pad_token not set  
- **LoRA config:** r=8, alpha=16, dropout=0.05 on `q_proj` and `v_proj` modules  

### Hyperparameters
- Batch size: 2 per device (with gradient accumulation)  
- Learning rate: 2e-4  
- Epochs: 1 (test run)  
- Precision: FP16  

---

## Technical Specs

- **Frameworks:** PyTorch, Transformers, PEFT, Datasets  
- **Environment:** Python 3.x  
- **Hardware:**  
  - GPU Type: NVIDIA L4  
  - Number of GPUs: 8  
  - GPU Memory (per GPU): 24 GB  
  - Total GPU Memory: 192 GB  
  - CPU Cores: 121  
  - System RAM: 1 TB  
- **Software versions:**  
  - PEFT ≥ 0.17  
  - Transformers ≥ 4.40  
  - Datasets ≥ 2.x  


---

## Bias, Risks, and Limitations

- May not generalize outside of diabetes / medical data similar to training set.  
- Potential for hallucination or incorrect summaries.  
- Outputs should be verified by medical experts.  

---

## Citation

If you use or build upon this model, please cite:

**BibTeX:**
```bibtex
@misc{diabetes_llama_lora_ranaehelal_2025,
  title={AI Agent Diabetes Diagnosis: LoRA Fine-Tuned LLaMA-7B},
  author={Ranaehelal},
  year={2025},
  howpublished={GitHub, https://github.com/TarekkMU1911/AI-Agent-Diabetes-Diagnosis}
}
