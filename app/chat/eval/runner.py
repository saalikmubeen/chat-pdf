"""Run RAG evaluation on a single run or a dataset."""

import time
from collections.abc import Callable

from langchain_core.documents import Document

from app.chat.eval.metrics import (
    RAGEvalResult,
    compute_answer_relevance,
    compute_context_relevance,
    compute_faithfulness,
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def evaluate_single_run(
    question: str,
    answer: str,
    source_documents: list[Document],
    total_latency_sec: float,
    embeddings,
    judge_llm,
    *,
    retrieval_latency_sec: float | None = None,
    generation_latency_sec: float | None = None,
) -> RAGEvalResult:
    """
    Compute all RAG metrics for one Q&A run.

    Args:
        question: User question (standalone).
        answer: Model answer.
        source_documents: Retrieved documents.
        total_latency_sec: Total time for the RAG call.
        embeddings: Embedding model for context_relevance.
        judge_llm: LLM for faithfulness and answer_relevance.
        retrieval_latency_sec: Optional; if not set, half of total is used for retrieval.
        generation_latency_sec: Optional; if not set, half of total is used for generation.

    Returns:
        RAGEvalResult with all metrics.

    """
    context = _format_docs(source_documents)

    if retrieval_latency_sec is None and generation_latency_sec is None:
        retrieval_latency_sec = total_latency_sec / 2
        generation_latency_sec = total_latency_sec / 2
    elif retrieval_latency_sec is None:
        retrieval_latency_sec = max(0, total_latency_sec - generation_latency_sec)
    elif generation_latency_sec is None:
        generation_latency_sec = max(0, total_latency_sec - retrieval_latency_sec)

    context_relevance = compute_context_relevance(
        question,
        source_documents,
        embeddings,
    )
    faithfulness = compute_faithfulness(context, answer, judge_llm)
    answer_relevance = compute_answer_relevance(question, answer, judge_llm)

    return RAGEvalResult(
        context_relevance=context_relevance,
        retrieval_latency_sec=retrieval_latency_sec,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        generation_latency_sec=generation_latency_sec,
        total_latency_sec=total_latency_sec,
    )


def run_eval_dataset(
    dataset: list[dict],
    invoke_fn: Callable[[dict], tuple[str, list[Document]]],
    embeddings,
    judge_llm,
) -> dict:
    """
    Run evaluation on a dataset of (question, conversation_id, pdf_id) and aggregate.

    invoke_fn(row) should return (answer, source_documents) for that row.
    Row dict must have "question" key; other keys (e.g. conversation_id, pdf_id) are for invoke_fn.

    Returns:
        {
            "summary": { "context_relevance": {"mean": ..., "std": ..., "count": ...}, ... },
            "results": [ RAGEvalResult.to_dict() per row ],
            "count": N,
        }

    """
    results: list[RAGEvalResult] = []
    debug_list: list[dict] = []
    for row in dataset:
        question = row.get("question", "").strip()
        if not question:
            continue
        start = time.perf_counter()
        try:
            answer, source_documents = invoke_fn(row)
        except Exception as e:
            total_sec = time.perf_counter() - start
            raise RuntimeError(
                f"RAG invoke failed for question {question!r}: {e}",
            ) from e
        total_sec = time.perf_counter() - start
        # Normalize: ensure we have a list of objects with page_content (LangChain Document-like)
        docs = []
        for d in source_documents or []:
            if hasattr(d, "page_content"):
                docs.append(d)
            elif isinstance(d, dict) and "page_content" in d:
                from langchain_core.documents import Document

                docs.append(
                    Document(
                        page_content=d["page_content"],
                        metadata=d.get("metadata", {}),
                    ),
                )
        result = evaluate_single_run(
            question=question,
            answer=answer or "",
            source_documents=docs,
            total_latency_sec=total_sec,
            embeddings=embeddings,
            judge_llm=judge_llm,
        )
        results.append(result)
        debug_list.append(
            {
                "n_source_documents": len(docs),
                "answer_length": len(answer or ""),
                "answer_preview": (
                    answer[:200] + "..." if len(answer or "") > 200 else (answer or "")
                )
                or "(empty)",
            },
        )

    if not results:
        return {
            "summary": _empty_summary(),
            "results": [],
            "count": 0,
        }

    def agg(key: str) -> dict:
        values = [getattr(r, key) for r in results]
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n if n else 0
        return {"mean": round(mean, 4), "std": round(variance**0.5, 4), "count": n}

    summary = {
        "context_relevance": agg("context_relevance"),
        "retrieval_latency_sec": agg("retrieval_latency_sec"),
        "faithfulness": agg("faithfulness"),
        "answer_relevance": agg("answer_relevance"),
        "generation_latency_sec": agg("generation_latency_sec"),
        "total_latency_sec": agg("total_latency_sec"),
    }
    any_zero_retrieval = any(d.get("n_source_documents", 0) == 0 for d in debug_list)
    return {
        "summary": summary,
        "results": [
            {**r.to_dict(), "debug": d}
            for r, d in zip(results, debug_list, strict=True)
        ],
        "count": len(results),
        "warnings": [
            "No documents were retrieved for one or more questions (n_source_documents=0). "
            "Context relevance and faithfulness will be 0. Ensure this PDF has been embedded "
            "into the vector store (e.g. run the Celery worker and re-upload or re-process the PDF).",
        ]
        if any_zero_retrieval
        else [],
    }


def _empty_summary() -> dict:
    keys = [
        "context_relevance",
        "retrieval_latency_sec",
        "faithfulness",
        "answer_relevance",
        "generation_latency_sec",
        "total_latency_sec",
    ]
    return {k: {"mean": 0, "std": 0, "count": 0} for k in keys}
