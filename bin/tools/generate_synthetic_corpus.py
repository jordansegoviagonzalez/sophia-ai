import json
import random
import os

# Configuration
OUTPUT_DIR = "../../ai-ml-backend-datasets/synthetic_corpus"
TARGET_SIZE = 750

# 1. Knowledge Graph (Facts)
KNOWLEDGE_GRAPH = {
    "loss_functions": [
        ("What is Cross-Entropy?", "Cross-Entropy Loss (Log Loss) measures the performance of a classification model whose output is a probability value between 0 and 1."),
        ("When to use Focal Loss?", "Use Focal Loss when you have a severe class imbalance, like fraud detection where 99% of cases are negative."),
        ("Explain MSE for classification.", "Do NOT use MSE for classification. It assumes a Gaussian distribution of errors, which is incorrect for categorical probabilities."),
        ("Can I use MAE for regression?", "Yes, Mean Absolute Error (MAE) is robust to outliers compared to MSE.")
    ],
    "mlops": [
        ("Why use Docker?", "Docker ensures that the environment (OS, libraries, drivers) is identical in Dev, Staging, and Production."),
        ("What is Data Drift?", "Data Drift occurs when the statistical properties of the input data change over time, degrading model performance."),
        ("Explain Concept Drift.", "Concept Drift happens when the relationship between inputs and outputs changes (e.g., spam emails evolve)."),
        ("Why not run scripts directly?", "Running raw scripts leads to 'Dependency Hell' and makes rollbacks difficult. Always containerize.")
    ],
    "architecture": [
        ("Is GPT an Encoder?", "No. GPT is a Decoder-Only architecture designed for autoregressive text generation."),
        ("How does BERT differ from GPT?", "BERT is Encoder-Only (bidirectional, good for understanding). GPT is Decoder-Only (unidirectional, good for creating)."),
        ("What is RAG?", "RAG (Retrieval-Augmented Generation) combines a retriever (Vector DB) with a generator (LLM) to reduce hallucinations.")
    ]
}

# 2. Persona Templates (High Variance)
TEMPLATES = {
    "Senior Engineer": [
        "In production environments, {fact}",
        "Technically speaking, {fact}",
        "The standard pattern is: {fact}",
        "We prefer this because {fact}",
        "Avoid the alternative. {fact}",
        "From an engineering perspective, {fact}"
    ],
    "Mentor": [
        "Think of it this way: {fact}",
        "The reason we do this is because {fact}",
        "A good rule of thumb: {fact}",
        "Don't get confused. {fact}",
        "Remember: {fact}",
        "To build robust systems, know that {fact}"
    ],
    "Skeptical": [
        "You might assume otherwise, but {fact}",
        "It's a common misconception. Actually, {fact}",
        "Contrary to popular belief, {fact}",
        "Don't make the rookie mistake. {fact}",
        "Be careful. {fact}",
        "Strictly speaking, {fact}"
    ]
}

def generate_entry():
    domain = random.choice(list(KNOWLEDGE_GRAPH.keys()))
    q, core_fact = random.choice(KNOWLEDGE_GRAPH[domain])
    
    persona_name = random.choice(list(TEMPLATES.keys()))
    template = random.choice(TEMPLATES[persona_name])
    
    # 50% chance to use the template prefix, 50% chance to use the fact raw (to prevent pattern matching)
    if random.random() > 0.5:
        final_answer = template.format(fact=core_fact)
    else:
        final_answer = core_fact

    return {
        "instruction": q,
        "input": "", # Empty input to force model to rely on instruction
        "output": final_answer,
        "category": f"{domain}_synthetic",
        "source": "synthetic_generator_v2"
    }

def main():
    print(f"--- Launching High-Variance Synthetic Engine ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "synthetic_5k.jsonl")
    
    with open(output_file, "w") as f:
        for i in range(TARGET_SIZE):
            entry = generate_entry()
            f.write(json.dumps(entry) + "\n")
            
    print(f"Generated {TARGET_SIZE} diverse samples.")

if __name__ == "__main__":
    main()