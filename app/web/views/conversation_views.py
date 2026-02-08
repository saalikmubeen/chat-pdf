from flask import Blueprint, Response, g, jsonify, request, stream_with_context

from app.chat import ChatArgs, build_chat
from app.chat.tracing.langfuse import langfuse
from app.web.db.models import Conversation, Pdf
from app.web.hooks import load_model, login_required

bp = Blueprint("conversation", __name__, url_prefix="/api/conversations")


@bp.route("/", methods=["GET"])
@login_required
@load_model(Pdf, lambda r: r.args.get("pdf_id"))
def list_conversations(pdf):
    return [c.as_dict() for c in pdf.conversations]


@bp.route("/", methods=["POST"])
@login_required
@load_model(Pdf, lambda r: r.args.get("pdf_id"))
def create_conversation(pdf):
    conversation = Conversation.create(user_id=g.user.id, pdf_id=pdf.id)

    return conversation.as_dict()


@bp.route("/<string:conversation_id>/messages", methods=["POST"])
@login_required
@load_model(Conversation)
def create_message(conversation):
    user_input = request.json.get("input") if request.json else None
    streaming = request.args.get("stream", False)

    pdf = conversation.pdf

    chat_args = ChatArgs(
        conversation_id=conversation.id,
        pdf_id=pdf.id,
        streaming=streaming,
        metadata={
            "conversation_id": conversation.id,
            "user_id": g.user.id,
            "pdf_id": pdf.id,
        },
    )

    chat = build_chat(chat_args)

    if not chat:
        return "Chat not yet implemented!"

    config = {"configurable": {"session_id": conversation.id}}

    # Capture all SQLAlchemy attributes before entering generator context
    user_id = str(g.user.id)
    pdf_id = pdf.id
    conversation_id = conversation.id

    # Langfuse trace input - ensure it's never None/undefined for proper tracing
    question = user_input if user_input is not None else ""
    trace_input = {"question": question}

    if streaming:

        def generate():
            # Create trace first with explicit input (fixes undefined input in Langfuse)
            trace = langfuse.trace(
                name="rag_conversation",
                session_id=conversation_id,
                user_id=user_id,
                metadata={
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "pdf_id": pdf_id,
                    "streaming": True,
                },
                input=trace_input,
            )
            # Create generation under the trace
            generation = langfuse.generation(
                name="rag_conversation",
                trace_id=trace.id,
                metadata={"pdf_id": pdf_id, "streaming": True},
                input=trace_input,
            )

            full_response = ""
            try:
                for chunk in chat.stream({"question": question}, config=config):
                    if isinstance(chunk, dict):
                        text = chunk.get("answer", "")
                    else:
                        text = str(chunk)
                    if text:
                        full_response += text
                        yield text

                output_data = {"answer": full_response}
                generation.end(output=output_data)
                trace.update(output=output_data)
            except Exception as e:
                output_data = {"error": str(e)}
                generation.end(output=output_data, level="ERROR")
                trace.update(output=output_data)
                raise
            finally:
                langfuse.flush()

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
        )

    # Non-streaming: Create trace first with explicit input
    trace = langfuse.trace(
        name="rag_conversation",
        session_id=conversation_id,
        user_id=user_id,
        metadata={
            "conversation_id": conversation_id,
            "user_id": user_id,
            "pdf_id": pdf_id,
            "streaming": False,
        },
        input=trace_input,
    )
    generation = langfuse.generation(
        name="rag_conversation",
        trace_id=trace.id,
        metadata={"pdf_id": pdf_id, "streaming": False},
        input=trace_input,
    )

    try:
        result = chat.invoke({"question": question}, config=config)
        if isinstance(result, dict):
            content = result.get("answer") or result.get("content") or str(result)
        else:
            content = str(result)

        output_data = {"answer": content}
        generation.end(output=output_data)
        trace.update(output=output_data)
        langfuse.flush()
        return jsonify({"role": "assistant", "content": content})
    except Exception as e:
        output_data = {"error": str(e)}
        generation.end(output=output_data, level="ERROR")
        trace.update(output=output_data)
        langfuse.flush()
        raise
