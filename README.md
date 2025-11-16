# Sophia – Interview Question Coach (ML + LLM System)

Sophia is a **mini interview coach** for ML / LLM / backend roles.

You give Sophia an interview question in plain language. Behind the scenes she:

1. Classifies the question into a topic (ML basics, loss functions, Transformers, RAG, deployment, etc.).  
2. Retrieves your **own notes** for that topic from simple `.md` files.  
3. Builds a structured prompt and (later) sends it to an LLM (local or cloud).  
4. Returns a response with:
   - A technical answer (interview-style)  
   - A simple explanation (to check understanding)  
   - One follow-up question

This project is designed to show **end-to-end ML + LLM system design**:

- Classical ML (scikit-learn classifier, train/test, persisted model)  
- Lightweight RAG (note-based knowledge store)  
- LLM orchestration (prompt building, response schema)  
- Clean architecture, tests, and documentation  

---

## Project Structure

```text
sophia/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── questions_labeled.csv        # labeled interview questions
│
├── notes/                           # your curated study notes
│   ├── ml_basics.md
│   ├── loss_functions.md
│   ├── llm_transformers.md
│   ├── rag.md
│   └── deployment.md
│
├── models/
│   └── topic_classifier.joblib      # trained topic classifier
│
├── src/
│   └── sophia/
│       ├── __init__.py
│       ├── config.py
│       ├── topics.py
│       ├── schemas.py
│       ├── classifier.py
│       ├── knowledge_base.py
│       ├── llm_client.py
│       ├── pipeline.py
│       └── cli.py
│
├── scripts/
│   └── train_classifier.py
│
└── tests/
    ├── test_classifier.py
    ├── test_knowledge_base.py
    └── test_pipeline.py
```

---

## Setup

```bash
cd sophia

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Data: Labeled Questions

Create `data/questions_labeled.csv` with at least a few examples per topic:

```csv
question,topic
"What is overfitting and how do you prevent it?",ml_basics
"Explain cross-entropy loss.",loss_functions
"How does attention work in Transformers?",llm_transformers
"Describe a RAG (Retrieval-Augmented Generation) system.",rag
"How would you deploy an LLM-backed API to production?",deployment
```

You can expand this file over time.

---

## Notes: Your Knowledge Base

Each `.md` file in `notes/` is your “golden” summary for that topic. For example:

`notes/ml_basics.md`:

```markdown
# ML Basics

- Overfitting: model fits noise instead of signal.
- Prevent with: more data, regularization, dropout, simpler model, early stopping.
- Always keep a held-out validation set for tuning.
```

Sophia will search these notes to provide grounded answers.

---

## Training the Topic Classifier

The topic classifier is a simple **TF-IDF + Logistic Regression** model trained with scikit-learn.

Train it via:

```bash
python -m scripts.train_classifier
```

This will:

- Load `data/questions_labeled.csv`  
- Split into train/test  
- Train the pipeline  
- Print a classification report  
- Save the model to `models/topic_classifier.joblib`  

---

## Running Sophia (CLI)

Once the classifier is trained and you have notes, run:

```bash
PYTHONPATH=src python -m sophia.cli
```

Example session:

```text
Sophia 🧠 – Interview Question Coach
Type 'exit' to quit.

Your interview question> How do you prevent overfitting in a neural network?

============================================================
[Topic: ml_basics]  [Source: model_only]

Technical answer:
...

Simple explanation:
...

Follow-up question:
...
============================================================
```

Out of the box, the LLM client is a **stub** that echoes where the LLM answer will go.  
You can later plug in:

- Local models (Ollama, llama.cpp, etc.)  
- Cloud LLMs (Azure OpenAI, OpenAI, etc.)  

by editing `llm_client.py` and using your own keys (not committed to git).

---

## Architecture

High-level pipeline:

```text
Question
  ↓
TopicClassifier (scikit-learn)
  ↓
KnowledgeBase (notes/*.md retrieval)
  ↓
LLMClient (prompt construction + LLM call)
  ↓
SophiaResponse (technical answer, simple answer, follow-up, source_type)
```

Key modules:

- `classifier.py` – trains & runs the topic classifier.  
- `knowledge_base.py` – loads your notes and performs simple keyword-based retrieval.  
- `llm_client.py` – centralizes prompt building + future LLM API calls.  
- `pipeline.py` – orchestrates classify → retrieve → answer.  
- `cli.py` – interactive terminal UX.  

---

## Tests

Run tests with:

```bash
pytest
```

Tests cover:

- Classifier: trains on a tiny sample and predicts a known topic.  
- Knowledge base: returns relevant chunks given a topic hint.  
- Pipeline: runs with a dummy LLMClient and returns a well-formed `SophiaResponse`.  

---

## Next Steps

- Add more topics and richer notes.  
- Replace the stub LLM client with a real model.  
- Add a simple web API (FastAPI/Flask) on top of `SophiaPipeline`.  
- Log sessions to a DB to track your progress over time.  

Sophia is intentionally small but complete: it shows you understand **ML, LLMs, RAG, and system design** in one focused project.
