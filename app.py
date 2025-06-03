import os
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load components once at startup
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
peft_model_id = "linear_merged_adapter/merge"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

@st.cache_resource
def load_models():
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    ).eval()
    
    peft_model = PeftModel.from_pretrained(
        base_model,
        peft_model_id,
        adapter_name="merge"
    )
    
    return base_model, peft_model

# Streamlit UI
st.title("TinyLlama Philosophy Assistant")
user_input = st.text_area("Ask a philosophical question:")

if st.button("Generate Responses"):
    if user_input.strip():
        # Prepare ChatML prompt
        prompt = f"""<|im_start|>system
You are a philosophical AI assistant.<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        # Load models (cached)
        base_model, peft_model = load_models()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Original model response
            inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
            with torch.inference_mode():
                outputs = base_model.generate(**inputs, max_new_tokens=200)
                original_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.subheader("Fine tuned Model")
            st.write(original_text.split("<|im_start|>assistant")[-1].strip())

        with col2:
            # Fine-tuned response
            with peft_model.disable_adapter():
                peft_model.set_adapter("merge")
                inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)
                with torch.inference_mode():
                    outputs = peft_model.generate(
                        **inputs,
                        max_new_tokens=200
                    )
                    finetuned_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                st.subheader("Base model")
                st.write(finetuned_text.split("<|im_start|>assistant")[-1].strip())
                
    else:
        st.warning("Please enter a question.")
