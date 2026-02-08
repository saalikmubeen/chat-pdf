from langchain_ollama import ChatOllama

from app.chat.chains.conversational_rag import (
    build_conversational_rag_chain,
    build_conversational_rag_chain_with_history,
)
from app.chat.llms import llm_map
from app.chat.memories.histories.sql_history import SqlMessageHistory
from app.chat.models import ChatArgs
from app.chat.score import random_component_by_score
from app.chat.tracing.langfuse import langfuse
from app.chat.vector_stores import retriever_map
from app.web.api import (
    get_conversation_components,
    set_conversation_components,
)


def get_langfuse_metadata(chat_args: ChatArgs) -> dict:
    """
    Get metadata for Langfuse tracing.

    Args:
        chat_args: Chat arguments containing metadata

    Returns:
        dict with session_id, user_id, and metadata for tracing

    """
    return {
        "session_id": chat_args.conversation_id,
        "user_id": str(chat_args.metadata.user_id),
        "metadata": chat_args.metadata.model_dump(),  # Convert Pydantic model to dict
    }


def create_langfuse_handler(chat_args: ChatArgs):
    """
    Create a Langfuse callback handler for detailed chain tracing.

    Args:
        chat_args: Chat arguments containing metadata

    Returns:
        Langfuse callback handler that traces all chain operations

    """
    # Create a trace and get its handler for LangChain callbacks
    trace = langfuse.trace(
        name="conversation",
        session_id=chat_args.conversation_id,
        user_id=str(chat_args.metadata.user_id),
        metadata=chat_args.metadata.model_dump(),
    )
    # Get the callback handler from the trace
    return trace.get_langchain_handler()


def select_component(
    component_type,
    component_map,
    chat_args,
):
    components = get_conversation_components(
        chat_args.conversation_id,
    )
    previous_component = components[component_type]

    if previous_component:
        builder = component_map[previous_component]
        return previous_component, builder(chat_args)
    random_name = random_component_by_score(component_type, component_map)
    builder = component_map[random_name]
    return random_name, builder(chat_args)


def get_session_history(session_id: str) -> SqlMessageHistory:
    """
    Get SQL message history for a session.

    This function signature matches what RunnableWithMessageHistory expects.
    """
    return SqlMessageHistory(conversation_id=session_id)


def build_chat(chat_args: ChatArgs):
    """
    Build the conversational RAG chain.

    Args:
        chat_args: Chat arguments containing conversation and metadata

    Returns:
        The conversational RAG chain

    """
    retriever_name, retriever = select_component(
        "retriever",
        retriever_map,
        chat_args,
    )
    llm_name, llm = select_component(
        "llm",
        llm_map,
        chat_args,
    )
    # memory_name, memory = select_component(
    #     "memory",
    #     memory_map,
    #     chat_args
    # )
    set_conversation_components(
        chat_args.conversation_id,
        llm=llm_name,
        retriever=retriever_name,
        memory="SQL",
    )

    # condense_question_llm = ChatOpenAI(streaming=False)
    condense_question_llm = ChatOllama(model="llama3.2", streaming=False)

    chain = build_conversational_rag_chain_with_history(
        llm=llm,
        retriever=retriever,
        get_session_history=get_session_history,
        condense_llm=condense_question_llm,
    )

    return chain


def build_chat_base(chat_args: ChatArgs):
    """
    Build the base conversational RAG chain without message history.
    Use for eval so invoke() returns the full dict with answer and source_documents.
    """
    retriever_name, retriever = select_component(
        "retriever",
        retriever_map,
        chat_args,
    )
    llm_name, llm = select_component(
        "llm",
        llm_map,
        chat_args,
    )
    set_conversation_components(
        chat_args.conversation_id,
        llm=llm_name,
        retriever=retriever_name,
        memory="SQL",
    )
    condense_question_llm = ChatOllama(model="llama3.2", streaming=False)
    return build_conversational_rag_chain(
        llm=llm,
        retriever=retriever,
        condense_llm=condense_question_llm,
    )

    # return StreamingConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     condense_question_llm=condense_question_llm,
    #     memory=memory,
    #     retriever=retriever,
    #     metadata=chat_args.metadata
    # )
