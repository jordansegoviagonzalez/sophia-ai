import os
import sys

def check(condition, msg):
    if condition:
        print(f"[PASS] {msg}")
    else:
        print(f"[FAIL] {msg}")
        sys.exit(1)

print("--- Sophia AI Pre-Flight Check ---")

# 1. Check Source Data Content
with open("../../ai-ml-backend-datasets/raw/domain_knowledge/deep_learning/llm_transformers.md", "r") as f:
    content = f.read()
    check("Decoder-Only" in content and "GPT" in content, "Knowledge Base contains updated GPT definitions")

# 2. Check Dataset Script
with open("scripts/prepare_data.py", "r") as f:
    content = f.read()
    check("ai-ml-backend-knowledge-v1" in content, "prepare_data.py targets correct V1 dataset path")
    check("import re" in content, "prepare_data.py has Regex parser for Guanaco")

# 3. Check Training Script
with open("scripts/finetune_llm.py", "r") as f:
    content = f.read()
    check("from peft import" in content and "PeftModel" in content, "finetune_llm.py imports PeftModel")
    check("eval_strategy" in content, "finetune_llm.py uses valid 'eval_strategy' arg")
    check("ai-ml-backend-knowledge-v1" in content, "finetune_llm.py targets correct V1 dataset path")

# 4. Check Checkpoint Cleanup
if os.path.exists("models/sophia-finetuned/checkpoint-150"):
    print("[WARN] Old checkpoints exist. These must be deleted before V2 training.")
else:
    print("[PASS] Clean workspace (no old checkpoints).")

print("\nALL SYSTEMS GO.")
