import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def train():
    # 1. Configuration
    # We use Qwen2.5-1.5B-Instruct (SOTA for small models)
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    OUTPUT_DIR = "models/sophia-finetuned"
    DATA_PATH = "../../ai-ml-backend-datasets/processed/ai-ml-backend-knowledge-v3"

    print(f"Loading model: {MODEL_NAME}")
    
    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Qwen/Llama usually don't set a default pad token for training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in half-precision (float16) to save memory if on GPU, otherwise float32 on CPU
    # checking for MPS (Mac Metal) support
    device_map = "auto"
    if torch.backends.mps.is_available():
        print("Using Mac MPS (Metal Performance Shaders) acceleration!")
        device_map = None # Accelerate handles MPS better usually, or we let Trainer handle it
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map
    )

    # 3. Apply LoRA (Low-Rank Adaptation)
    # This allows us to fine-tune only a tiny fraction of parameters (~1%)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16, # Increased rank for Qwen
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all linear layers for better results
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Prepare Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run scripts/prepare_data.py first.")
        
    # Load the pre-split datasets
    print(f"Loading datasets from {DATA_PATH}...")
    train_dataset = load_from_disk(os.path.join(DATA_PATH, "train"))
    eval_dataset = load_from_disk(os.path.join(DATA_PATH, "validation"))
    
    def tokenize_function(examples):
        text_samples = []
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            # Robustly format using the model's own chat template
            messages = [
                {"role": "system", "content": "You are Sophia, an expert AI assistant."},
                {"role": "user", "content": f"{instr}\n{inp}".strip()},
                {"role": "assistant", "content": out}
            ]
            # apply_chat_template handles the specific tags (e.g. <|im_start|>) for us
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            text_samples.append(prompt)
            
        return tokenizer(text_samples, padding="max_length", truncation=True, max_length=512)

    print("Tokenizing training data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    print("Tokenizing validation data...")
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # Reduced for larger Qwen model on Mac
        gradient_accumulation_steps=16, # Increased to 16 for effective batch size of 16 (smoother)
        num_train_epochs=3,
        learning_rate=5e-5,          # Lower learning rate for precision
        logging_steps=10,            # More frequent logs for small dataset
        save_steps=100,              # Save periodically
        eval_strategy="steps",       
        eval_steps=100,              # Check quality periodically
        save_total_limit=2,          # Keep only last 2 checkpoints to save space
        use_mps_device=torch.backends.mps.is_available(),
        fp16=False,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval, # Pass the validation set here
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    # Check if we have a checkpoint to resume from
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by number to get the latest
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print(f"Saving temporary adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 7. MERGE STEP (The "Ownership" Step)
    # We reload the base model in full precision, merge the adapter, and save the standalone result.
    print("Merging weights to create standalone Sophia model...")
    
    # Clean up memory first
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Reload base in 16-bit (or 32-bit CPU)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        return_dict=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu" # Load on CPU to avoid memory fragmentation during merge
    )
    
    # Load the adapter we just trained
    model_to_merge = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    
    # Merge!
    merged_model = model_to_merge.merge_and_unload()
    
    # Save the final standalone model
    FINAL_DIR = "models/sophia-v1-standalone"
    merged_model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    
    print(f"SUCCESS: Your standalone model is ready at: {FINAL_DIR}")
    print("You can now delete the 'models/sophia-finetuned' folder if you wish.")

if __name__ == "__main__":
    train()
