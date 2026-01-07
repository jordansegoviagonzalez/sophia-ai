# Sophia-V2 (Fine-Tuned LLM)

## Model Description
**Sophia-V2** is a domain-specific instruction-tuned Large Language Model (LLM) designed to act as an expert Technical Interview Coach for AI/ML and Backend Engineering roles.

It is fine-tuned on a curated corpus of high-density technical knowledge, specifically optimized to correct common misconceptions (e.g., GPT architecture nuances) and provide "Best Practice" engineering advice (e.g., Docker containerization).

## Model Details
*   **Base Model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
*   **Architecture:** Decoder-Only Transformer
*   **Parameters:** 1.54 Billion
*   **Quantization:** FP16 (Training), FP16 (Inference)
*   **Context Window:** 32k (Trained on 512 max seq length)

## Training Configuration
*   **Method:** QLoRA (Low-Rank Adaptation)
*   **Rank (r):** 16
*   **Alpha:** 32
*   **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (All Linear Layers)
*   **Epochs:** 3.0
*   **Batch Size:** 1 (Effective Batch Size = 8 via Gradient Accumulation)
*   **Learning Rate:** 1e-4
*   **Loss Function:** Cross-Entropy

## Dataset Lineage
*   **Corpus:** `ai-ml-backend-knowledge-v2`
*   **Composition:**
    *   **Domain Knowledge (Platinum):** Internal Expert Notes (ML Basics, Loss Functions, Deployment, RAG, Transformers).
    *   **General Intelligence (Gold):** `mlabonne/guanaco-llama2-1k` (Filtered for English, Deduplicated).
*   **Size:** ~840 Training Samples, ~90 Validation Samples.

## Usage
Load using the provided `LLMClient` in `src/sophia/llm_client.py`.

```python
from sophia.llm_client import LLMClient
client = LLMClient(model_path="models/sophia-v1-standalone")
response = client.ask("Does GPT use an Encoder?", topic="DL", notes=[])
print(response.technical_answer)
```
