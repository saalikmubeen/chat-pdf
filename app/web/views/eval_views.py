"""API for running RAG evaluation and fetching metrics."""

import uuid

from flask import Blueprint, g, jsonify, request
from werkzeug.exceptions import BadRequest

from app.chat import build_chat_base
from app.chat.embeddings.openai import embeddings
from app.chat.eval import run_eval_dataset
from app.chat.models import ChatArgs
from app.web.db import db
from app.web.db.models import Conversation, Message, Pdf
from app.web.hooks import load_model, login_required

bp = Blueprint("eval", __name__, url_prefix="/api/eval")


def _get_judge_llm():
    """LLM for faithfulness and answer_relevance scoring. Non-streaming."""
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", streaming=False)
    except Exception:
        from langchain_ollama import ChatOllama

        return ChatOllama(model="llama3.2", streaming=False)


@bp.route("/run", methods=["POST"])
@login_required
@load_model(Pdf, lambda r: (request.json or {}).get("pdf_id"))
def run_eval(pdf):
    """
    Run RAG evaluation on a set of questions for the given PDF.

    Body:
        pdf_id: int (required, also in URL/load_model)
        questions: list[str] (optional) - use these questions
        max_samples: int (optional) - if no questions, sample this many recent human messages for this PDF

    Returns:
        { "summary": {...}, "results": [...], "count": N }

    """
    data = request.json or {}
    questions = data.get("questions")
    max_samples = data.get("max_samples", 10)

    if questions is not None and not isinstance(questions, list):
        raise BadRequest("questions must be a list of strings")
    if questions and not all(isinstance(q, str) for q in questions):
        raise BadRequest("questions must be a list of strings")

    if questions:
        dataset = [
            {"question": q.strip(), "pdf_id": pdf.id}
            for q in questions
            if q and isinstance(q, str) and q.strip()
        ]
    else:
        # Sample recent human messages for this PDF
        messages = (
            Message.query.join(
                Conversation,
                Message.conversation_id == Conversation.id,
            )
            .filter(Conversation.pdf_id == pdf.id)
            .filter(Message.role == "human")
            .order_by(Message.created_on.desc())
            .limit(max(min(int(max_samples), 50), 1))
            .all()
        )
        dataset = [
            {"question": m.content.strip(), "pdf_id": pdf.id}
            for m in messages
            if m.content and m.content.strip()
        ]

    if not dataset:
        return jsonify(
            {
                "summary": _empty_summary(),
                "results": [],
                "count": 0,
                "message": "No questions to evaluate. Provide 'questions' or ensure there are human messages for this PDF.",
            },
        )

    judge_llm = _get_judge_llm()

    def invoke_fn(row):
        conversation_id = str(uuid.uuid4())
        # build_chat() calls get_conversation_components(conversation_id), which requires
        # a Conversation row to exist. Create a temporary one for this eval run.
        Conversation.create(
            id=conversation_id,
            pdf_id=row["pdf_id"],
            user_id=g.user.id,
            llm="gpt-3.5-turbo",
            retriever="pinecone_2",
            memory="SQL",
        )
        try:
            chat_args = ChatArgs(
                conversation_id=conversation_id,
                pdf_id=row["pdf_id"],
                streaming=False,
                metadata={
                    "conversation_id": conversation_id,
                    "user_id": g.user.id,
                    "pdf_id": row["pdf_id"],
                },
            )
            # Use base chain (no history) so we get full dict with answer + source_documents
            chain = build_chat_base(chat_args)
            result = chain.invoke(
                {
                    "question": row["question"],
                    "chat_history": [],
                }
            )
            if isinstance(result, dict):
                answer = result.get("answer", "") or ""
                docs = result.get("source_documents", [])
            else:
                answer = str(result)
                docs = []
            return answer, docs
        finally:
            # Remove the temporary conversation and its messages to avoid clutter
            try:
                conv = Conversation.find_by(id=conversation_id)
                if conv:
                    for msg in list(conv.messages):
                        db.session.delete(msg)
                    db.session.delete(conv)
                    db.session.commit()
            except Exception:
                db.session.rollback()

    try:
        report = run_eval_dataset(
            dataset=dataset,
            invoke_fn=invoke_fn,
            embeddings=embeddings,
            judge_llm=judge_llm,
        )
    except Exception as e:
        return jsonify(
            {
                "error": str(e),
                "detail": getattr(e, "__cause__", None) and str(e.__cause__),
            },
        ), 500
    return jsonify(report)


def _empty_summary():
    keys = [
        "context_relevance",
        "retrieval_latency_sec",
        "faithfulness",
        "answer_relevance",
        "generation_latency_sec",
        "total_latency_sec",
    ]
    return {k: {"mean": 0, "std": 0, "count": 0} for k in keys}


@bp.route("/metrics", methods=["GET"])
# @login_required
def get_metrics_doc():
    """Return a short description of the metrics used for RAG evaluation."""
    return jsonify(
        {
            "metrics": {
                "context_relevance": "Average embedding similarity between the question and each retrieved chunk (0-1). Higher = better retrieval.",
                "faithfulness": "LLM judge: is the answer fully supported by the retrieved context? (0-1).",
                "answer_relevance": "LLM judge: does the answer address the question? (0-1).",
                "retrieval_latency_sec": "Time spent in retrieval (estimated).",
                "generation_latency_sec": "Time spent in generation (estimated).",
                "total_latency_sec": "End-to-end RAG latency in seconds.",
            },
            "usage": 'POST /api/eval/run with JSON body: { "pdf_id": <int>, "questions": ["..."] } or { "pdf_id": <int>, "max_samples": 10 } to run on recent messages.',
        },
    )
