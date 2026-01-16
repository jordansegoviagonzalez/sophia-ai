import os
import glob
import re
import json
import hashlib
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets

# Configuration
PUBLIC_DATASET = "mlabonne/guanaco-llama2-1k" 
NOTES_DIR = "../../ai-ml-backend-datasets/raw/domain_knowledge"
SYNTHETIC_DIR = "../../ai-ml-backend-datasets/synthetic_corpus"
OUTPUT_PATH = "../../ai-ml-backend-datasets/processed/ai-ml-backend-knowledge-v3"

def get_row_hash(text):
    """Generate SHA256 hash for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def process_local_notes():
    """
    Feature Engineering Step 1: 
    Extract knowledge from Markdown files recursively.
    """
    abs_notes_dir = os.path.abspath(NOTES_DIR)
    print(f"--- Processing Local Notes from Taxonomy: {abs_notes_dir} ---")
    
    data_samples = []
    files = glob.glob(os.path.join(NOTES_DIR, "**/*.md"), recursive=True)
    
    for f_path in files:
        path_obj = Path(f_path)
        topic = path_obj.stem.replace("_", " ").title()
        domain_category = path_obj.parent.name
        
        print(f"Ingesting: {topic} (Domain: {domain_category})")
        
        with open(f_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 1. Full Document Authority
        data_samples.append({
            "instruction": f"Explain the concept of {topic} based on the internal knowledge base.",
            "input": "",
            "output": content,
            "source": "internal_kb",
            "category": domain_category
        })
        
        # 2. Section Chunking
        lines = content.split('\n')
        current_header = ""
        current_chunk = []
        for line in lines:
            if line.startswith("#"):
                if current_header and current_chunk:
                    text = "\n".join(current_chunk).strip()
                    if len(text) > 50:
                        data_samples.append({
                            "instruction": f"What does the knowledge base say about {current_header} in the context of {topic}?",
                            "input": "",
                            "output": text,
                            "source": "internal_kb",
                            "category": domain_category
                        })
                current_header = line.lstrip("#").strip()
                current_chunk = []
            else:
                current_chunk.append(line)
        
        # Catch last chunk
        if current_header and current_chunk:
             text = "\n".join(current_chunk).strip()
             if len(text) > 50:
                data_samples.append({
                    "instruction": f"Detail {current_header} regarding {topic}.",
                    "input": "",
                    "output": text,
                    "source": "internal_kb",
                    "category": domain_category
                })

    print(f"Generated {len(data_samples)} structured samples from domain taxonomy.")
    return Dataset.from_list(data_samples)

def process_synthetic_data():
    """
    Feature Engineering Step 3:
    Ingest the massive 5k synthetic corpus.
    """
    print(f"--- Processing Synthetic Corpus from {SYNTHETIC_DIR} ---")
    data_samples = []
    
    files = glob.glob(os.path.join(SYNTHETIC_DIR, "*.jsonl"))
    for f_path in files:
        with open(f_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data_samples.append(json.loads(line))
                    
    print(f"Generated {len(data_samples)} synthetic samples.")
    if not data_samples:
        return None
    return Dataset.from_list(data_samples)

def process_public_data():
    """
    Feature Engineering Step 2:
    Load external 'General Intelligence' data and standardize columns.
    """
    print(f"--- Loading Public Dataset: {PUBLIC_DATASET} ---")
    ds = load_dataset(PUBLIC_DATASET, split="train")
    
    # PARSING LOGIC
    def parse_guanaco(example):
        text = example.get('text', '')
        match = re.search(r'\[INST\](.*?)(\[/INST\])(.*)', text, re.DOTALL)
        if match:
            return {
                'instruction': match.group(1).strip(),
                'input': '',
                'output': match.group(3).replace('</s>', '').strip(),
                'text': text
            }
        return {'instruction': '', 'input': '', 'output': '', 'text': text}

    ds = ds.map(parse_guanaco)

    # FILTER 1: English
    def is_english_clean(example):
        text = example.get('text', '')
        if any('\u0400' <= char <= '\u04FF' for char in text):
            return False
        return True

    # FILTER 2: Quality
    def is_high_quality(example):
        out_text = example.get('output', '')
        if not out_text or len(out_text.strip()) < 20: 
            return False
        instr_text = example.get('instruction', '')
        if not instr_text or len(instr_text.strip()) < 5:
            return False
        return True

    original_len = len(ds)
    ds = ds.filter(is_english_clean)
    ds = ds.filter(is_high_quality)
    print(f"Quality Filter: Removed {original_len - len(ds)} low-quality/non-English rows.")

    # DEDUPLICATION
    unique_hashes = set()
    def is_unique(example):
        content = (example.get('instruction', '') + example.get('input', '')).strip()
        h = get_row_hash(content)
        if h in unique_hashes:
            return False
        unique_hashes.add(h)
        return True
        
    ds = ds.filter(is_unique)

    # Metadata
    def add_metadata(example):
        example["source"] = "external_guanaco"
        example["category"] = "general_instruction"
        return example
        
    ds = ds.map(add_metadata)
    return ds

def main():
    # 1. Load Data Sources
    local_ds = process_local_notes()
    synthetic_ds = process_synthetic_data()
    public_ds = process_public_data()
    
    # 2. Data Augmentation (Mixing)
    datasets_to_merge = [local_ds, public_ds]
    if synthetic_ds:
        datasets_to_merge.append(synthetic_ds)
        
    combined_ds = concatenate_datasets(datasets_to_merge)
    
    # 3. Strict Train/Validation Split (90/10)
    split_ds = combined_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]
    
    print(f"\n--- Data Pipeline Complete (V3) ---")
    print(f"Total Training Samples:   {len(train_ds)}")
    print(f"Total Validation Samples: {len(val_ds)}")
    
    # 4. Serialization
    train_ds.save_to_disk(os.path.join(OUTPUT_PATH, "train"))
    val_ds.save_to_disk(os.path.join(OUTPUT_PATH, "validation"))
    print(f"Datasets saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()