
---

# RAG Debugger & Evaluation Tool

A practical **Retrieval-Augmented Generation (RAG) debugging and evaluation tool** for inspecting retrieval quality, detecting weak grounding, tracking experiments, and comparing retrieval configurations over time.

This project focuses on **observability and evaluation**, not just answer generation.

This project was built incrementally to better understand how RAG systems fail in practice, not just how they work when everything goes right.

---

## Why this project exists

Most RAG demos stop at:

> “Ask a question → get an answer.”

In practice, that’s not enough.

RAG systems often **fail silently**:

* retrieval may be weak
* answers may sound confident but be poorly grounded
* small configuration changes can degrade performance without obvious signs

This tool exists to answer harder questions:

* *Why did the system answer this way?*
* *Did it retrieve the right information?*
* *How confident should we be in this answer?*
* *Did a recent change improve or regress performance?*

---

## What this tool can do

### 🔍 Debug Query Inspection

* Ask a single question
* Inspect retrieved chunks, sources, pages, and similarity scores
* See exactly **what the model saw** before answering
* View answers with explicit citations

### ⚠️ Weak Retrieval Detection

* Flags queries where the best retrieved chunk is beyond a configurable distance threshold
* Highlights answers that may be **correct but weakly grounded**
* Helps detect potential hallucination risk

> Note: retrieval distance measures semantic proximity, not reasoning quality

### 📊 Batch Evaluation (Hit@K)

* Run evaluation over a set of test questions
* Measure whether the expected source appears in the top-K retrieved chunks
* Compute aggregate metrics (Hit@K, weak retrieval rate, latency)

### 🧪 Run Tracking & Experiment Logging

* Each batch evaluation run is saved with:

  * retrieval configuration (chunk size, overlap, K, thresholds)
  * dataset identity
  * index identity
  * per-query results
  * aggregate metrics
* Enables reproducible experimentation instead of ad-hoc tuning

### 🔁 Run Comparison & Regression Analysis

* Compare two runs side-by-side
* Warns when datasets or indexes differ (fairness checks)
* Shows metric deltas
* Highlights **which specific queries changed**
* Drill down into per-query differences

This enables **regression testing for RAG systems**.

---

## Example use cases

* Evaluating chunk size and overlap trade-offs
* Detecting silent retrieval regressions
* Understanding why certain queries are weakly grounded
* Comparing retrieval configurations before deploying changes
* Teaching or learning how RAG systems behave in practice

---

## Project structure

```
rag-debugger/
├── app.py                 # Streamlit application
├── rag_core.py            # Core RAG logic (retrieval + answering)
├── eval_core.py           # Batch evaluation helpers (Hit@K, summaries)
├── ingest.py              # PDF loading and preprocessing
├── run_store.py           # SQLite-backed run tracking & comparison
├── utils.py               # Utilities
│
├── data/
│   ├── docs/              # PDFs (ignored)
│   ├── indexes/           # FAISS indexes (ignored)
│   ├── logs/              # Logs (ignored)
│   └── testset.example.csv
│
├── runs/                  # Saved evaluation runs (ignored)
│
├── assets/
│   └── screenshots/       # UI screenshots (optional)
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-debugger.git
cd rag-debugger
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy the example file:

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=your_key_here
```

### 4. Add documents

Place your PDF files in:

```
data/docs/
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Batch evaluation dataset format

The file `data/testset.example.csv` shows the required format:

```csv
id,question,expected_source,expected_page
1,What is X?,sample.pdf,2
2,Why does Y happen?,sample.pdf,4
```

To run batch evaluation:

* Create your own `data/testset.csv`
* Follow the same column structure
* This file is intentionally **ignored** by git

---

## Key ideas demonstrated

* Retrieval quality ≠ answer quality
* Reasoning-heavy questions often have weaker semantic proximity
* Evaluation must be systematic, not anecdotal
* Small configuration changes can cause silent regressions
* RAG systems need observability, not just generation

---

## What this is *not*

* Not a production deployment
* Not a prompt-engineering demo
* Not a chatbot showcase

This is an **engineering and evaluation tool**.

---

## Future directions (ideas)

* Tiered confidence levels instead of a single threshold
* Additional metrics (MRR, Recall@K)
* Regression alerts
* Larger evaluation datasets
* Re-ranking strategies (MMR, hybrid retrieval)

---

## License

MIT License

---

## Notes & Observations

While building this, a few things stood out:

- Retrieval distance is a poor proxy for answer correctness on reasoning-heavy questions
- Many correct answers were flagged as weakly grounded due to low lexical proximity
- Small retrieval changes often had little visible effect on averages, but did affect specific queries
- Comparing runs helped surface subtle regressions that would be easy to miss otherwise

This reinforced the importance of inspecting systems at the query level, not just relying on aggregate metrics.
