"""LLM-as-judge prompts for RAG evaluation."""

FAITHFULNESS_PROMPT = """You are an evaluator. Given the CONTEXT (retrieved source text) and the ANSWER (model output), determine if the answer is fully supported by the context.

Rules:
- The answer must only contain information that is present in the context.
- If the answer adds information not in the context, or contradicts the context, it is not faithful.
- If the answer says "I don't know" or similar when the context has no relevant info, that is faithful.

Output ONLY a single number from 0 to 1:
- 1.0 = The answer is fully supported by the context (faithful).
- 0.0 = The answer contains unsupported or contradictory information (not faithful).
- Use values in between for partial faithfulness.

CONTEXT:
{context}

ANSWER:
{answer}

Score (0-1):"""

ANSWER_RELEVANCE_PROMPT = """You are an evaluator. Given the QUESTION and the ANSWER, determine how well the answer addresses the question.

Output ONLY a single number from 0 to 1:
- 1.0 = The answer fully and directly addresses the question.
- 0.0 = The answer is irrelevant or does not address the question at all.
- Use values in between for partial relevance.

QUESTION:
{question}

ANSWER:
{answer}

Score (0-1):"""
