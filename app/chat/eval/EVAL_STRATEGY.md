# RAG Evaluation Strategy: In-Depth Guide

This document explains **what evals are**, **why we use them**, and **exactly what our RAG eval pipeline is doing**

---

## 1. What are evals?

**Evals** (evaluations) are **repeatable checks** that tell you how well a system is doing. For a RAG (Retrieval-Augmented Generation) app, we don’t have a single “correct” answer for every question, so we can’t just compare to a gold label. Instead we define **metrics** that measure different aspects of quality and performance.

Think of evals as:

- **Retrieval evals** – Did we fetch the right chunks?
- **Generation evals** – Did we produce a good answer (faithful, relevant)?
- **Performance evals** – How fast did we run?

We run the same pipeline on many questions (or the same question over time), compute these metrics, and aggregate (e.g. mean, std) to see if the system is good enough or if a change made things better or worse.

---

## 2. What is RAG (quick recap)?

A typical RAG flow:

1. **User** asks a question.
2. **Retriever** finds relevant chunks from a vector store (e.g. Pinecone) using the question (and maybe chat history).
3. **LLM** gets the question + those chunks as context and generates an answer.

So the final answer depends on **both** retrieval and generation. If retrieval is bad (wrong or empty chunks), the answer will be bad even if the model is great. If retrieval is good but the model ignores or invents from context, the answer can still be bad. Evals help us see **where** the pipeline is weak.

---

## 3. What can go wrong? → What we need to measure

| What can go wrong | What we measure |
|-------------------|------------------|
| Wrong or irrelevant chunks retrieved | **Context relevance** – do retrieved chunks match the question? |
| No chunks retrieved (e.g. PDF not embedded) | **Context relevance** (0) + debug `n_source_documents` |
| Model invents or contradicts the context | **Faithfulness** – is the answer grounded in the context? |
| Model doesn’t answer the question | **Answer relevance** – does the answer address the question? |
| Pipeline too slow | **Latency** – retrieval, generation, total time |

Our eval strategy is built around these three pillars: **retrieval quality**, **answer quality**, and **latency**.

---

## 4. The three pillars of our eval strategy

### Pillar 1: Retrieval quality

**Metric: context_relevance (0–1)**

- **Idea:** The chunks we retrieved should be *about* what the user asked. We measure that with **embedding similarity**.
- **How it’s computed:**
  1. Embed the **question** with the same embedding model used for the index (e.g. OpenAI or Ollama).
  2. Embed each **retrieved chunk** (its text).
  3. Compute **cosine similarity** between the question embedding and each chunk embedding.
  4. **Average** those similarities → that’s `context_relevance`.

- **Interpretation:**
  - **High (e.g. 0.6–1.0):** Retrieved chunks are semantically close to the question; retrieval is doing its job.
  - **Low (e.g. 0–0.3):** Chunks are not very related to the question; retrieval or chunking might need improvement.
  - **0:** No chunks retrieved (or embedding failed). Check `n_source_documents` in debug.

- **Limitation:** This measures “similarity to the question,” not “does this chunk contain the *answer*?” So you can have high context relevance but still miss the right passage. For a deeper retrieval eval you’d add things like recall@k with labeled data; our metric is a simple, label-free proxy.

---

### Pillar 2: Answer quality

We use two metrics, both scored by an **LLM-as-judge**: faithfulness and answer relevance.

#### Metric: faithfulness (0–1)

- **Idea:** The answer should be **supported by** the retrieved context: no invention, no contradiction.
- **How it’s computed:**
  1. We send the **context** (concatenated retrieved chunks) and the **answer** to a judge LLM (e.g. gpt-4o-mini or Ollama).
  2. The prompt tells the judge: “Score 0–1: 1 = fully supported by context, 0 = unsupported or contradictory.”
  3. We parse a single number from the judge’s reply and clamp it to [0, 1].

- **Interpretation:**
  - **1.0:** Answer only uses information from the context (and doesn’t contradict it).
  - **0.0:** Answer invents or contradicts the context.
  - **In between:** Partial support (e.g. some parts grounded, some not).

- **Important:** If the model says “I don’t know” and the context really has no relevant info, the judge is instructed to treat that as **faithful** (honest). If the model says “I don’t know” but the context *did* have the answer, the judge can reasonably score low (answer didn’t use the context).

#### Metric: answer_relevance (0–1)

- **Idea:** The answer should **address the user’s question** (on-topic, responsive).
- **How it’s computed:**
  1. We send the **question** and the **answer** to the judge LLM.
  2. The prompt: “Score 0–1: 1 = fully addresses the question, 0 = irrelevant or doesn’t address it.”
  3. We parse the number from the judge’s reply.

- **Interpretation:**
  - **High:** The user’s question is clearly answered.
  - **Low:** The answer is off-topic or evasive (e.g. “I don’t know” when the user asked for a deep dive).

**What is “LLM-as-judge”?**

- We use **another LLM** to score quality instead of hand-written rules. The judge gets clear instructions and outputs a single number.
- **Pros:** Flexible, no need for labeled data, can capture nuance (e.g. “partially faithful”).
- **Cons:** Slight cost/latency, and the score can vary a bit between runs; we try to parse robustly (e.g. “Score: 0.8” or “0.8” both work).

---

### Pillar 3: Latency

**Metrics: retrieval_latency_sec, generation_latency_sec, total_latency_sec**

- **Idea:** We care how long the full RAG call takes and (roughly) how much time is retrieval vs generation.
- **How it’s computed:**
  - We **time the whole RAG run** (question in → answer + docs out). That’s `total_latency_sec`.
  - We don’t instrument the chain step-by-step, so we **estimate** retrieval vs generation by **splitting total time in half** for each. You could later add real retrieval/generation timers for finer detail.

- **Use:** Spot regressions (e.g. after adding more chunks or a heavier model), and set expectations for user experience.

---

## 5. End-to-end: what happens when you run an eval?

High-level flow:

1. **Input:** You send a list of **questions** (and a `pdf_id`). Optionally you can sample recent messages instead of supplying questions.
2. **Per question:**
   - A **temporary conversation** is created in the DB (so the existing chat pipeline can load components).
   - The **base RAG chain** (no history) is run: question + empty chat history → **answer** + **source_documents**.
   - **Timing:** We record total time for this run.
   - **Context relevance:** We embed the question and each retrieved doc, compute cosine similarities, average → one number.
   - **Faithfulness:** Judge LLM gets (context, answer) → we parse a 0–1 score.
   - **Answer relevance:** Judge LLM gets (question, answer) → we parse a 0–1 score.
   - Temporary conversation is deleted.
3. **Aggregation:** For each metric we compute **mean** and **std** over all questions, and return **per-question results** + **summary** + **debug** (e.g. `n_source_documents`, `answer_preview`).

So in one eval run we get:

- **Retrieval quality:** context_relevance (and whether we got any docs).
- **Answer quality:** faithfulness, answer_relevance.
- **Performance:** latency numbers.
- **Debug:** so we can see empty retrieval or empty answers and fix the pipeline.

---

## 6. How to interpret results

- **context_relevance high, faithfulness/answer_relevance low**
  Retrieval is good; the model may be ignoring context or answering vaguely. Look at prompts, model choice, or number of chunks.

- **context_relevance low**
  Retrieval or indexing issue: wrong/empty chunks, bad chunking, or wrong `pdf_id`. Use `n_source_documents` and fix embedding/indexing.

- **faithfulness low, answer_relevance high**
  The model is answering the question but not from the context (hallucination or off-context knowledge). Tighten the system prompt to “only use the context.”

- **All metrics high**
  Pipeline is doing what we want for that set of questions. Use evals again after changes to guard against regressions.

- **Latency high**
  Inspect retrieval (index size, k) and model (size, API vs local) and optimize where it hurts.

---

## 7. When to use this eval strategy

- **After changing** retrieval (chunk size, embedding model, k, or index).
- **After changing** the model or the QA prompt.
- **On a sample of real or synthetic questions** to get a baseline (mean/std per metric).
- **In CI or before release** to catch regressions (e.g. “mean faithfulness must stay above 0.7”).

We are **not** (in this setup):

- Using human-labeled “correct” answers (no exact match accuracy).
- Measuring retrieval recall with a golden set of relevant chunks (that would be a separate, label-based eval).

---

## 8. Streaming: token-level

The app uses **token-level streaming** for the RAG answer:

- The API supports `?stream=true` and returns **Server-Sent Events (SSE)**.
- The pipeline is **StreamingConversationalRAGChain** (`app/chat/chains/conversational_rag.py`): **invoke()** = full pipeline; **stream()** = condense + retrieve, then `qa_chain.stream(...)` yielding `{"answer": token}` per token.
- The frontend appends each chunk so the answer appears **word-by-word** as the LLM generates it.

The pipeline itself is built as one **RunnableLambda** that runs **condense → retrieve → QA** and then returns `{"answer": ..., "source_documents": ...}`. So when you call `chat.stream()`:

- The **Lambda runs to completion** (condense + retrieve + full LLM reply).
- The “stream” the client sees is typically **the full answer (or a single chunk)** after that work is done, not **token-by-token** as the LLM generates.

So:

- **Streaming in the app:** Yes – SSE, streaming response, frontend can show progress.
- **Token-level streaming:** Probably not – the answer usually appears in one go after the delay, unless the LangChain version/wrapper streams the inner LLM. If the UI shows words appearing one-by-one, then you do have token-level; if the answer pops in after a few seconds, it’s “streaming the completed result.”

To get **guaranteed token-level streaming** you’d restructure so the **QA step** is streamed (e.g. run condense + retrieve once, then stream `qa_chain.stream(...)` and yield those tokens over SSE), instead of wrapping the whole pipeline in a single Lambda.

---

## 9. Summary table

| Metric | What it measures | How we get it |
|--------|-------------------|----------------|
| **context_relevance** | Do retrieved chunks match the question? | Embed question + chunks; average cosine similarity. |
| **faithfulness** | Is the answer grounded in the context? | LLM judge (context + answer → 0–1). |
| **answer_relevance** | Does the answer address the question? | LLM judge (question + answer → 0–1). |
| **retrieval_latency_sec** | Time in retrieval (estimate) | Half of total (unless instrumented). |
| **generation_latency_sec** | Time in generation (estimate) | Half of total. |
| **total_latency_sec** | End-to-end RAG time | Wall-clock time of the RAG call. |

Together, this eval strategy gives you a **repeatable, automated** view of retrieval quality, answer quality, and speed so you can improve and monitor your RAG pipeline in depth.
