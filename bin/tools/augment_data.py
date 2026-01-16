import os
import random

# Target Knowledge to Reinforce
FACTS = [
    {
        "q_patterns": [
            "Does GPT use an Encoder?",
            "Is GPT an encoder-decoder model?",
            "What is the architecture of GPT-3?",
            "Does Llama use an encoder?",
            "Difference between BERT and GPT architecture?",
            "Explain the decoder-only architecture.",
            "Does GPT-4 have bidirectional attention?",
            "Is Qwen an encoder model?"
        ],
        "a_patterns": [
            "No, GPT uses a Decoder-Only architecture. It does not have an encoder stack.",
            "GPT models (and Llama/Qwen) are Decoder-Only. They process text autoregressively.",
            "Unlike BERT (Encoder-only), GPT is Decoder-Only. It predicts the next token based on previous ones.",
            "GPT is strictly Decoder-Only. The original Transformer had both, but GPT dropped the Encoder.",
            "No encoder here. GPT uses a stack of masked self-attention decoder blocks.",
            "It is a common misconception, but GPT is Decoder-Only. It does not use the Encoder half of the Transformer."
        ]
    }
]

OUTPUT_FILE = "../../ai-ml-backend-datasets/raw/domain_knowledge/deep_learning/gpt_architecture_augmented.md"

def generate_augmented_data():
    print(f"--- Generating Augmented Data for: {OUTPUT_FILE} ---")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("# GPT Architecture (Augmented Knowledge)\n\n")
        
        count = 0
        # Generate 50 combinations
        for _ in range(50):
            fact = FACTS[0]
            q = random.choice(fact["q_patterns"])
            a = random.choice(fact["a_patterns"])
            
            # Format as a Q&A block for our pipeline
            # Our pipeline reads paragraphs, so we write conversational blocks
            f.write(f"### Question: {q}\n")
            f.write(f"{a}\n\n")
            count += 1
            
    print(f"Successfully generated {count} reinforced examples.")

if __name__ == "__main__":
    generate_augmented_data()
