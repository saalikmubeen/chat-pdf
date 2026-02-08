from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


# def build_llm(chat_args, model_name):
#     return ChatOpenAI(
#         disable_streaming=not chat_args.streaming,
#         model_name=model_name
#     )


def build_llm(chat_args, model_name):
    return ChatOllama(
        disable_streaming=not chat_args.streaming,
        model="llama3.2"
    )
