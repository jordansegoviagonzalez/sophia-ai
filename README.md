![alt text](Sophia-AI.png)
# Sophia AI - Enterprise Interview Coach

**Sophia** is a domain-specific, fine-tuned Large Language Model (LLM) designed to act as an expert technical interviewer for AI/ML, Backend, and MLOps roles.

Unlike generic wrappers, Sophia operates on a **Full Ownership** model. She is not an API call to OpenAI; she is a standalone, 1.5B parameter neural network fine tuned on a proprietary **Enterprise Knowledge Lake**.

---

## 🚀 Key Features

*   **🧠 Custom Fine-Tuned Brain:** Built on a 1.5B parameter foundation, fine-tuned using **QLoRA** on a high-density technical corpus.
*   **🤗 Model Weights:** [Download Sophia-V1 from Hugging Face](https://huggingface.co/jordansegovia/sophia-v1)
*   **📚 Enterprise Knowledge Lake:** A curated ontology of engineering wisdom (`ai-ml-backend-datasets`), structured into domains like MLOps, Deep Learning, and System Design.
*   **🛠️ Synthetic Alignment:** Uses **Targeted Data Augmentation** to correct common base-model hallucinations (e.g., enforcing "GPT is Decoder-Only", "No MSE for Classification").
*   **⚡ Local Inference:** Runs 100% locally on Apple Silicon (MPS/Metal) or CUDA. No data leaves your machine.

---

## 🏗️ Architecture

### Model Architecture (Transformer)
```mermaid
graph TD
    %% Cyberpunk / Scientific Color Palette
    classDef input fill:#161b22,stroke:#58a6ff,stroke-width:2px,color:#a5d6ff;
    classDef attn fill:#161b22,stroke:#bc8cff,stroke-width:2px,color:#d2a8ff;
    classDef ffn fill:#161b22,stroke:#ff7b72,stroke-width:2px,color:#ffa198;
    classDef norm fill:#161b22,stroke:#8b949e,stroke-width:1px,stroke-dasharray: 3 3,color:#c9d1d9;
    classDef output fill:#161b22,stroke:#3fb950,stroke-width:2px,color:#56d364;
    classDef residual fill:#1f6feb,stroke:none,color:#ffffff,font-weight:bold;

    subgraph "Context Window (32k Tokens)"
        TOK["Tokens (Input)"]:::input --> EMB["Embedding Layer<br/>(d=1536)"]:::input
        ROPE["RoPE (Rotary Pos)"]:::input -.-> EMB
    end

    EMB --> PRE_NORM_1[RMSNorm]:::norm

    subgraph "Transformer Layer (x28)"
        direction TB
        
        PRE_NORM_1 --> MHA["Masked Multi-Head Attention<br/>(12 Heads)"]:::attn
        MHA --> ADD_1((+)):::residual
        EMB -.->|Residual| ADD_1
        
        ADD_1 --> PRE_NORM_2[RMSNorm]:::norm
        PRE_NORM_2 --> FFN["SwiGLU FFN<br/>(Intermediate=8960)"]:::ffn
        
        FFN --> ADD_2((+)):::residual
        ADD_1 -.->|Residual| ADD_2
    end

    ADD_2 --> FINAL_NORM[RMSNorm]:::norm
    FINAL_NORM --> LM_HEAD["Linear Head<br/>(Vocab=151k)"]:::output
    LM_HEAD --> SOFT[Softmax]:::output
    SOFT --> PROB["Next Token Output"]:::output
```

The system follows a modern **Instruction Tuning** pipeline:

1.  **Ingestion:** Raw Markdown notes are ingested from `ai-ml-backend-datasets`.
2.  **Feature Engineering:**
    *   Data is deduplicated (SHA256 hashing).
    *   Filtered for quality and language.
    *   Augmented with synthetic "Hard Negatives" to fix reasoning gaps.
3.  **Training:**
    *   **Base:** Qwen 2.5 (1.5B).
    *   **Adapter:** LoRA (Rank 16, Alpha 32).
    *   **Merge:** Weights are fused into a standalone artifact (`models/sophia-v1-standalone`).

---

## 📂 Project Structure

### System Map
```mermaid
graph LR
    subgraph "User Interface"
        CLI["CLI (bin/sophia)"]
        API["API Server (Future)"]
    end

    subgraph "Core Library (src/sophia)"
        PL["Pipeline Controller"]
        SCH["Schemas (Pydantic)"]
        CFG["Config Manager"]
        
        subgraph "Engine Room"
            TC["Topic Classifier"]
            KB["Knowledge Base (RAG)"]
            LLM["LLM Client (Inference)"]
        end
    end

    subgraph "Data Layer"
        RAW[("Raw Markdown")]
        DB[("Processed Dataset")]
        VEC[("Vector Store")]
    end

    %% Connections
    CLI -->|Request| PL
    PL -->|Validate| SCH
    PL -->|Load| CFG
    
    PL -->|1. Predict| TC
    PL -->|2. Retrieve| KB
    PL -->|3. Generate| LLM

    KB -.->|Read| VEC
    TC -.->|Load| DB
```

```text
sophia-ai/
├── ai-ml-backend-datasets/      # (External) The Source of Truth
│   ├── raw/domain_knowledge/    # The "Gold" Corpus (Markdown)
│   └── processed/               # The "Fuel" (Tokenized Arrow Files)
│
├── models/
│   ├── sophia-v1-standalone/    # The Fused, Production-Ready Model
│   └── README.md                # Model Card (Hyperparameters)
│
├── scripts/
│   ├── prepare_data.py          # ETL Pipeline (Clean -> Hash -> Split)
│   ├── finetune_llm.py          # Training Loop (Hugging Face Trainer)
│   ├── augment_data.py          # Synthetic Data Generator
│   └── audit_data_quality.py    # Forensic Data Unit Tests
│
├── src/
│   └── sophia/
│       └── llm_client.py        # Inference Engine
```

---

## 🛠️ Quick Start

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
If you want to reproduce the weights from the raw knowledge base:
```bash
# 1. Build the Dataset
python scripts/prepare_data.py

# 2. Run Fine-Tuning
python scripts/finetune_llm.py
```

### 3. Chat with Sophia
Launch the interactive shell to interview the model.
```bash
PYTHONPATH=src python scripts/chat.py
```

---

## 📊 Performance & Governance

*   **Hallucination Rate:** Minimally observed on core topics due to "Sledgehammer" augmentation.
*   **Data Lineage:** All training data is versioned in `ai-ml-backend-datasets`.
*   **PII Safety:** Automated audit scripts run prior to every training job.

---

**Author:** DJ Jordan
**License:** Apache 2.0