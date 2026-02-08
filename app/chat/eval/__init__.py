"""RAG evaluation: retrieval, answer quality, and latency metrics."""

from app.chat.eval.metrics import (
    RAGEvalResult,
    compute_answer_relevance,
    compute_context_relevance,
    compute_faithfulness,
)
from app.chat.eval.runner import evaluate_single_run, run_eval_dataset

__all__ = [
    "RAGEvalResult",
    "compute_answer_relevance",
    "compute_context_relevance",
    "compute_faithfulness",
    "evaluate_single_run",
    "run_eval_dataset",
]
