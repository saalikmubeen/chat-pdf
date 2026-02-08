"""RAG evaluation metrics: retrieval, answer quality, and latency."""

import re
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class RAGEvalResult:
    """Result of evaluating a single RAG run."""

    # Retrieval
    context_relevance: float  # 0-1, mean similarity of retrieved docs to question
    retrieval_latency_sec: float

    # Generation
    faithfulness: float  # 0-1, answer grounded in context (LLM judge)
    answer_relevance: float  # 0-1, answer addresses question (LLM judge)
    generation_latency_sec: float

    # Overall
    total_latency_sec: float

    def to_dict(self) -> dict:
        return {
            "context_relevance": round(self.context_relevance, 4),
            "retrieval_latency_sec": round(self.retrieval_latency_sec, 4),
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "generation_latency_sec": round(self.generation_latency_sec, 4),
            "total_latency_sec": round(self.total_latency_sec, 4),
        }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Assumes vectors may be normalized."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_context_relevance(
    question: str,
    source_documents: list[Document],
    embeddings,
) -> float:
    """
    Average embedding similarity between the question and each retrieved document.
    Higher = retrieved context is more relevant to the question.
    """
    if not source_documents:
        return 0.0
    try:
        q_embedding = embeddings.embed_query(question)
        doc_texts = [doc.page_content for doc in source_documents]
        doc_embeddings = embeddings.embed_documents(doc_texts)
        sims = [_cosine_similarity(q_embedding, d) for d in doc_embeddings]
        return sum(sims) / len(sims) if sims else 0.0
    except Exception:
        return 0.0


def _parse_float_from_llm_response(text: str) -> float | None:
    """Extract a single float from LLM output (e.g. '0.85' or 'Score: 0.9')."""
    if not text:
        return None
    text = text.strip()
    # Try last token that looks like a number
    numbers = re.findall(r"0?\.\d+|\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    try:
        return float(text)
    except ValueError:
        return None


def compute_faithfulness(context: str, answer: str, llm) -> float:
    """
    LLM-as-judge: is the answer fully supported by the context? Returns 0-1.
    """
    if not context or not answer:
        return 0.0
    from langchain_core.prompts import PromptTemplate

    from app.chat.eval.prompts import FAITHFULNESS_PROMPT

    prompt = PromptTemplate.from_template(FAITHFULNESS_PROMPT)
    try:
        response = llm.invoke(
            prompt.format(context=context[:8000], answer=answer[:4000]),
        )
        text = response.content if hasattr(response, "content") else str(response)
        score = _parse_float_from_llm_response(text)
        if score is not None and 0 <= score <= 1:
            return score
        return 0.5
    except Exception:
        return 0.0


def compute_answer_relevance(question: str, answer: str, llm) -> float:
    """
    LLM-as-judge: does the answer address the question? Returns 0-1.
    """
    if not question or not answer:
        return 0.0
    from langchain_core.prompts import PromptTemplate

    from app.chat.eval.prompts import ANSWER_RELEVANCE_PROMPT

    prompt = PromptTemplate.from_template(ANSWER_RELEVANCE_PROMPT)
    try:
        response = llm.invoke(
            prompt.format(question=question[:2000], answer=answer[:4000]),
        )
        text = response.content if hasattr(response, "content") else str(response)
        score = _parse_float_from_llm_response(text)
        if score is not None and 0 <= score <= 1:
            return score
        return 0.5
    except Exception:
        return 0.0
