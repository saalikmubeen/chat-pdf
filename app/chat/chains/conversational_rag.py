from collections.abc import Callable, Iterator

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from app.chat.memories.histories.sql_history import SqlMessageHistory

# === Prompts ===

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:""",
)

QA_PROMPT = PromptTemplate(
    template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Helpful Answer:""",
    input_variables=["context", "question"],
)


# === Helper Functions ===


def format_chat_history(messages: list[BaseMessage]) -> str:
    """Format chat history into a string."""
    if not messages:
        return ""

    formatted = []
    for msg in messages:
        if msg.type == "human":
            formatted.append(f"Human: {msg.content}")
        elif msg.type == "ai":
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)


def format_docs(docs) -> str:
    """Format retrieved documents."""
    return "\n\n".join(doc.page_content for doc in docs)


# === Streamable RAG Chain (invoke = full pipeline; stream = condense + retrieve, then token-level QA) ===


class StreamingConversationalRAGChain(Runnable[dict, dict]):
    """
    RAG chain that supports token-level streaming.

    - invoke(): runs condense -> retrieve -> QA (invoke), returns {"answer", "source_documents"}.
    - stream(): runs condense -> retrieve, then streams QA tokens as {"answer": token} per chunk.
    """

    def __init__(self, condense_chain, retrieval_chain, qa_chain, format_docs_fn):
        self.condense_chain = condense_chain
        self.retrieval_chain = retrieval_chain
        self.qa_chain = qa_chain
        self.format_docs_fn = format_docs_fn

    def invoke(
        self,
        inputs: dict,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict:
        standalone_question = self.condense_chain.invoke(
            inputs,
            config=config,
            **kwargs,
        )
        docs = self.retrieval_chain.invoke(
            standalone_question,
            config=config,
            **kwargs,
        )
        answer = self.qa_chain.invoke(
            {
                "context": self.format_docs_fn(docs),
                "question": standalone_question,
            },
            config=config,
            **kwargs,
        )
        return {"answer": answer, "source_documents": docs}

    def stream(
        self,
        inputs: dict,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> Iterator[dict]:
        standalone_question = self.condense_chain.invoke(
            inputs,
            config=config,
            **kwargs,
        )
        docs = self.retrieval_chain.invoke(
            standalone_question,
            config=config,
            **kwargs,
        )
        qa_input = {
            "context": self.format_docs_fn(docs),
            "question": standalone_question,
        }
        for chunk in self.qa_chain.stream(qa_input, config=config, **kwargs):
            yield {"answer": chunk}


# === Main RAG Chain ===


def build_conversational_rag_chain(llm, retriever, condense_llm=None):
    """
    Build a conversational RAG chain using LCEL.

    This chain:
    1. Takes input as: {"question": str, "chat_history": List[BaseMessage]}
    2. Condenses the question based on chat history
    3. Retrieves relevant documents
    4. Generates an answer
    5. Returns: {"answer": str, "source_documents": List[Document]}

    Args:
        llm: Main language model for answering
        retriever: Vector store retriever
        condense_llm: Optional LLM for condensing questions (non-streaming)

    Returns:
        Runnable chain

    """
    if condense_llm is None:
        condense_llm = ChatOpenAI(streaming=False)

    # === Condense Question Chain ===
    def condense_question(inputs: dict) -> str:
        """Condense question based on chat history."""
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        # If no history, return original question
        if not chat_history:
            return question

        # Condense the question
        formatted_history = format_chat_history(chat_history)
        result = condense_llm.invoke(
            CONDENSE_QUESTION_PROMPT.format(
                chat_history=formatted_history,
                question=question,
            ),
        )
        return result.content if hasattr(result, "content") else str(result)

    condense_chain = RunnableLambda(condense_question)

    # === Retrieval Chain ===
    def retrieve_documents(standalone_question: str):
        """Retrieve relevant documents."""
        return retriever.invoke(standalone_question)

    retrieval_chain = RunnableLambda(retrieve_documents)

    # === QA Chain ===
    qa_chain = QA_PROMPT | llm | StrOutputParser()

    # Use streamable chain: invoke() = full pipeline; stream() = condense+retrieve then token-level QA
    return StreamingConversationalRAGChain(
        condense_chain=condense_chain,
        retrieval_chain=retrieval_chain,
        qa_chain=qa_chain,
        format_docs_fn=format_docs,
    )


# === With Message History ===


def build_conversational_rag_chain_with_history(
    llm,
    retriever,
    get_session_history: Callable[[str], SqlMessageHistory],
    condense_llm=None,
):
    """
    Build conversational RAG chain with automatic history management.

    Args:
        llm: Main language model
        retriever: Vector store retriever
        get_session_history: Function that returns SqlMessageHistory for a session_id
        condense_llm: Optional LLM for condensing questions

    Returns:
        Chain with automatic conversation history management

    """
    # Build base chain
    base_chain = build_conversational_rag_chain(llm, retriever, condense_llm)

    # Wrap with message history
    # The chain expects: {"question": str, "chat_history": List[BaseMessage]}
    # RunnableWithMessageHistory will inject chat_history automatically
    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",  # Where the new user message is
        history_messages_key="chat_history",  # Where to inject history
        output_messages_key="answer",  # Where the AI response is
    )

    return chain_with_history
