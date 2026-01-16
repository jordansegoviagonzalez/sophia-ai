import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import shutil

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "models/sophia-finetuned"
OUTPUT_DIR = "models/sophia-v1-standalone"

def merge():
    print(f"--- Starting Merge Process ---")
    print(f"Base: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_DIR}")
    
    # 1. Load Base Model (CPU to save memory)
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 2. Load Adapter
    print("Loading LoRA adapter...")
    model_to_merge = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    # 3. Merge
    print("Merging weights (This fuses your training into the base model)...")
    merged_model = model_to_merge.merge_and_unload()
    
    # 4. Save
    print(f"Saving standalone model to: {OUTPUT_DIR}")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("SUCCESS: Model merged and saved.")

if __name__ == "__main__":
    merge()
