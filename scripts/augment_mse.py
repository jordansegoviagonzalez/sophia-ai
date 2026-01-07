import os
import random

# Target Knowledge to Reinforce: Loss Functions
FACTS = [
    {
        "q_patterns": [
            "Can I use MSE for classification?",
            "Is Mean Squared Error good for Cat vs Dog?",
            "Should I use MSE or Cross-Entropy for classification?",
            "What loss function for binary classification?",
            "Why not use MSE for classification?",
            "I am building a classifier. Should I use MSE?"
        ],
        "a_patterns": [
            "No, do not use MSE for classification. Use Cross-Entropy (Binary or Categorical). MSE is for Regression (predicting numbers).",
            "MSE is a poor choice for classification because it assumes a Gaussian distribution. Use Cross-Entropy instead.",
            "For classification (like Cat vs Dog), you must use Cross-Entropy (Log Loss). MSE is specifically for Regression tasks like predicting prices.",
            "Never use MSE for classification outputs. It leads to vanishing gradients. Always use Cross-Entropy.",
            "MSE is for Regression. Cross-Entropy is for Classification. Do not mix them.",
            "You should strongly prefer Cross-Entropy. MSE is designed for continuous values, not probability classes."
        ]
    }
]

OUTPUT_FILE = "../../ai-ml-backend-datasets/raw/domain_knowledge/machine_learning/loss_functions_augmented.md"

def generate_augmented_data():
    print(f"--- Generating Augmented Data for: {OUTPUT_FILE} ---")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Loss Functions (Augmented Rules)\n\n")
        
        count = 0
        # Generate 50 combinations
        for _ in range(50):
            fact = FACTS[0]
            q = random.choice(fact["q_patterns"])
            a = random.choice(fact["a_patterns"])
            
            f.write(f"### Question: {q}\n")
            f.write(f"{a}\n\n")
            count += 1
            
    print(f"Successfully generated {count} reinforced MSE examples.")

if __name__ == "__main__":
    generate_augmented_data()
