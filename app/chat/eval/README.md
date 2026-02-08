# RAG evaluation

This module provides **retrieval**, **answer-quality**, and **latency** metrics for your RAG pipeline.

**In-depth guide:** See [EVAL_STRATEGY.md](./EVAL_STRATEGY.md) for what evals are, how each metric works, and how to interpret results.

## Metrics

| Metric | Description |
|--------|-------------|
| **context_relevance** | Average embedding similarity between the question and each retrieved chunk (0–1). Higher = better retrieval. |
| **faithfulness** | LLM-as-judge: is the answer fully supported by the retrieved context? (0–1). |
| **answer_relevance** | LLM-as-judge: does the answer address the question? (0–1). |
| **retrieval_latency_sec** | Time spent in retrieval (estimated as half of total when not instrumented). |
| **generation_latency_sec** | Time spent in generation (estimated). |
| **total_latency_sec** | End-to-end RAG latency in seconds. |

## API

- **`GET /api/eval/metrics`** (auth required) – Returns this metrics description and usage.
- **`POST /api/eval/run`** (auth required) – Run evaluation.

  Body (JSON):

  - `pdf_id` (required) – PDF to run evals for.
  - `questions` (optional) – List of questions to run. If omitted, recent human messages for this PDF are used.
  - `max_samples` (optional) – When not using `questions`, how many recent messages to use (default 10, max 50).

  Example:

  ```json
  { "pdf_id": 1, "questions": ["What is the main topic?", "Summarize section 2."] }
  ```

  Or to evaluate on recent traffic:

  ```json
  { "pdf_id": 1, "max_samples": 20 }
  ```

  Response:

  ```json
  {
    "summary": {
      "context_relevance": { "mean": 0.82, "std": 0.05, "count": 5 },
      "faithfulness": { "mean": 0.9, "std": 0.1, "count": 5 },
      "answer_relevance": { "mean": 0.88, "std": 0.07, "count": 5 },
      "total_latency_sec": { "mean": 2.3, "std": 0.4, "count": 5 },
      ...
    },
    "results": [ { "context_relevance": 0.85, "faithfulness": 0.9, ... }, ... ],
    "count": 5
  }
  ```

## Python usage

```python
from app.chat.eval import evaluate_single_run, run_eval_dataset

# Single run (you have question, answer, source_documents, and total_latency_sec)
result = evaluate_single_run(
    question="...",
    answer="...",
    source_documents=docs,
    total_latency_sec=1.5,
    embeddings=embeddings,
    judge_llm=judge_llm,
)
print(result.to_dict())

# Dataset: run chain on each row and aggregate
report = run_eval_dataset(
    dataset=[{"question": "q1", "pdf_id": 1}, ...],
    invoke_fn=lambda row: (answer, docs),  # your chain invoke
    embeddings=embeddings,
    judge_llm=judge_llm,
)
print(report["summary"])
```

## Judge LLM

Faithfulness and answer_relevance use an LLM judge. The app tries `gpt-4o-mini` (OpenAI) first, then falls back to `llama3.2` (Ollama). Set your `OPENAI_API_KEY` or run Ollama for best results.
