from __future__ import annotations

import torch
from pathlib import Path
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from .knowledge_base import NoteChunk
from .schemas import SophiaResponse, SourceType


class LLMClient:
    """
    Local LLM client using Hugging Face Transformers.
    """

    def __init__(self, model_path: str = "models/sophia-v1-standalone", base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        
        print(f"Loading Sophia AI from {model_path}...")
        
        # Check if our custom merged model exists
        if Path(model_path).exists():
            print(f"Loading custom standalone model: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # Load the standalone model directly
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
        else:
            print(f"Custom model not found at {model_path}. Falling back to base {base_model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_id, 
                dtype=torch.float16 if self.device != "cpu" else torch.float32
            )

        self.model.to(self.device)
        self.model.eval()

    def build_prompt(self, question: str, topic: str, notes: List[NoteChunk]) -> str:
        notes_text = "\n".join(f"- {n.text}" for n in notes)
        
        # Robust chat format using tokenizer template
        messages = [
            {"role": "system", "content": f"You are Sophia, an expert AI assistant specialized in {topic}. Use the provided context to answer accurately."},
            {"role": "user", "content": f"Context:\n{notes_text}\n\nQuestion: {question}"}
        ]
        
        # Apply template but do not tokenize yet, just return string
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def ask(self, question: str, topic: str, notes: List[NoteChunk]) -> SophiaResponse:
        source_type: SourceType = "personal_notes" if notes else "model_only"
        prompt = self.build_prompt(question, topic, notes)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.7, 
                do_sample=True
            )
            
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's reply. 
        # Since we use apply_chat_template, the output will contain the user prompt + response.
        # We find the last "assistant" marker or similar logic depending on the tokenizer decoding.
        # A simple robust way for many chat models:
        # 1. Remove the prompt text from the full_response
        # This is tricky with special tokens, so we often just split.
        
        # Heuristic: split by the last system/user/assistant token if visible, or just take the text after prompt length.
        prompt_len = len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
        generated_text = full_response[prompt_len:].strip()

        # In a future version, we can prompt the LLM to output JSON with specific fields.
        # For now, since the model generates a single text block, we put it all in technical_answer.
        
        return SophiaResponse(
            topic=topic,
            source_type=source_type,
            technical_answer=generated_text,
            simple_explanation="Refer to the detailed technical answer above.",
            follow_up_question="Would you like to dive deeper into a specific concept mentioned?"
        )
