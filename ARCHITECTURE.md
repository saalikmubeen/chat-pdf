# System Architecture

## Overview

This document provides a detailed technical overview of the RAG system architecture, design decisions, and component interactions.

## High-Level Architecture

The system follows a **layered architecture** pattern with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                      │
│                    (Svelte + TypeScript)                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│                     (Flask REST API)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Business Logic Layer                    │
│              (RAG Chains, LLM Orchestration)                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Data Access Layer                       │
│         (Vector Stores, SQL Database, Cache)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. RAG Chain Architecture

The conversational RAG chain is built using **LangChain Expression Language (LCEL)** and follows this pipeline:

```python
Input: {question: str, chat_history: List[BaseMessage]}
  ↓
[1] Question Condensation
  ↓ (standalone_question)
[2] Document Retrieval
  ↓ (relevant_documents)
[3] Context Formatting
  ↓ (formatted_context)
[4] Answer Generation
  ↓
Output: {answer: str, source_documents: List[Document]}
```

#### Step 1: Question Condensation
**Purpose**: Convert follow-up questions into standalone queries using conversation history.

**Implementation**:
```python
condense_chain = CONDENSE_QUESTION_PROMPT | condense_llm | StrOutputParser()
```

**Why**: Follow-up questions often contain pronouns or references to previous context. For example:
- User: "What is RAG?"
- Assistant: "RAG stands for Retrieval-Augmented Generation..."
- User: "How does it work?" ← This needs context

The condenser transforms "How does it work?" → "How does Retrieval-Augmented Generation work?"

#### Step 2: Document Retrieval
**Purpose**: Find the most relevant document chunks using semantic search.

**Implementation**:
```python
retriever.invoke(standalone_question)
```

**Vector Search Process**:
1. Embed the standalone question using the same embedding model as documents
2. Perform similarity search in vector store (cosine similarity)
3. Return top-k most relevant chunks (default k=4)

#### Step 3: Context Formatting
**Purpose**: Combine retrieved documents into a single context string.

**Implementation**:
```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

#### Step 4: Answer Generation
**Purpose**: Generate a grounded answer using retrieved context.

**Implementation**:
```python
qa_chain = QA_PROMPT | llm | StrOutputParser()
```

**Prompt Structure**:
```
Use the following pieces of context to answer the question.
If you don't know the answer, say so - don't make up information.

{context}

Question: {question}

Helpful Answer:
```

---

### 2. Memory Management

**SQL-Backed Conversation History**:
- Uses `SqlMessageHistory` for persistent storage
- Stores messages in the `message` table with `conversation_id` foreign key
- Automatically loads history for each conversation session

**Why SQL over In-Memory**:
- ✅ Persistence across server restarts
- ✅ Multi-user support
- ✅ Conversation history accessible for analytics
- ✅ Easy to implement pagination and search

**Integration with LangChain**:
```python
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```

---

### 3. Vector Store Architecture

**Dual Vector Store Support**:

| Feature | ChromaDB | Pinecone |
|---------|----------|----------|
| **Deployment** | Local, embedded | Cloud-hosted |
| **Use Case** | Development, small-scale | Production, large-scale |
| **Persistence** | Disk-based | Managed service |
| **Scalability** | Limited | Highly scalable |
| **Cost** | Free | Paid tiers |

**Document Chunking Strategy**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Characters per chunk
    chunk_overlap=200,      # Overlap to preserve context
    separators=["\n\n", "\n", " ", ""]
)
```

**Why These Parameters**:
- **chunk_size=1000**: Balances context vs. precision. Smaller chunks = more precise retrieval but less context. Larger chunks = more context but noisier retrieval.
- **chunk_overlap=200**: Ensures sentences/paragraphs aren't split awkwardly across chunks.

---

### 4. Async Task Processing

**Celery Worker Architecture**:

```
User uploads PDF
       ↓
Flask API creates PDF record
       ↓
Task queued in Redis
       ↓
Celery worker picks up task
       ↓
[1] Extract text from PDF
[2] Split into chunks
[3] Generate embeddings
[4] Store in vector DB
       ↓
Update PDF status → "ready"
```

**Why Celery**:
- PDF processing is CPU-intensive (text extraction, embedding generation)
- Keeps API responsive (non-blocking)
- Enables horizontal scaling (add more workers)
- Retry logic for failed tasks

**Task Example**:
```python
@celery.task
def process_pdf(pdf_id):
    pdf = Pdf.query.get(pdf_id)
    # Extract, chunk, embed, store
    pdf.status = "ready"
    db.session.commit()
```

---

### 5. Observability with Langfuse

**Tracing Hierarchy**:
```
Trace (conversation-level)
  ├── Generation (question condensation)
  ├── Retrieval (vector search)
  └── Generation (answer generation)
      ├── Input: {question, context}
      ├── Output: {answer}
      ├── Metadata: {model, tokens, latency}
```

**What Gets Traced**:
- ✅ User input
- ✅ Condensed question
- ✅ Retrieved documents (with scores)
- ✅ LLM prompts (full templates)
- ✅ LLM responses
- ✅ Token counts
- ✅ Latencies (per step)
- ✅ Errors and exceptions

**Integration Points**:
```python
# Streaming responses
trace = langfuse.trace(
    name="rag_conversation",
    session_id=conversation_id,
    user_id=user_id,
    input={"question": question}
)

generation = langfuse.generation(
    name="answer_generation",
    trace_id=trace.id,
    input={"question": question, "context": context}
)

# After streaming completes
generation.end(output={"answer": full_response})
trace.update(output={"answer": full_response})
```

---

### 6. Evaluation Framework

**Metric Calculation Pipeline**:

```python
# 1. Context Relevance (Embedding Similarity)
question_embedding = embeddings.embed_query(question)
context_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]
similarities = [cosine_similarity(question_embedding, ctx_emb) for ctx_emb in context_embeddings]
context_relevance = mean(similarities)

# 2. Faithfulness (LLM-as-Judge)
faithfulness_prompt = f"""
Given the context and answer, is the answer fully supported by the context?
Context: {context}
Answer: {answer}
Score (0-1):
"""
faithfulness_score = judge_llm.invoke(faithfulness_prompt)

# 3. Answer Relevance (LLM-as-Judge)
relevance_prompt = f"""
Does the answer address the question?
Question: {question}
Answer: {answer}
Score (0-1):
"""
answer_relevance = judge_llm.invoke(relevance_prompt)
```

**Why These Metrics**:
- **Context Relevance**: Measures retrieval quality (garbage in = garbage out)
- **Faithfulness**: Prevents hallucinations (answer must be grounded)
- **Answer Relevance**: Ensures the answer actually addresses the question

---

## Database Schema

```sql
-- Users
CREATE TABLE user (
    id INTEGER PRIMARY KEY,
    email VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- PDFs
CREATE TABLE pdf (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES user(id),
    filename VARCHAR NOT NULL,
    status VARCHAR DEFAULT 'processing',  -- processing, ready, failed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversations
CREATE TABLE conversation (
    id VARCHAR PRIMARY KEY,  -- UUID
    user_id INTEGER REFERENCES user(id),
    pdf_id INTEGER REFERENCES pdf(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages
CREATE TABLE message (
    id INTEGER PRIMARY KEY,
    conversation_id VARCHAR REFERENCES conversation(id),
    role VARCHAR NOT NULL,  -- 'human' or 'ai'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## API Design

### RESTful Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/auth/register` | User registration |
| `POST` | `/api/auth/login` | User login |
| `POST` | `/api/pdfs` | Upload PDF |
| `GET` | `/api/pdfs` | List user's PDFs |
| `POST` | `/api/conversations?pdf_id=<id>` | Create conversation |
| `GET` | `/api/conversations?pdf_id=<id>` | List conversations for PDF |
| `POST` | `/api/conversations/<id>/messages` | Send message (chat) |
| `POST` | `/api/eval/run` | Run evaluation |
| `GET` | `/api/eval/metrics` | Get metric descriptions |

### Streaming Response Format

**Server-Sent Events (SSE)**:
```python
def generate():
    for chunk in chain.stream({"question": question}, config=config):
        yield chunk.get("answer", "")

return Response(
    stream_with_context(generate()),
    mimetype="text/event-stream"
)
```

**Client Consumption**:
```typescript
const eventSource = new EventSource('/api/conversations/123/messages?stream=true');
eventSource.onmessage = (event) => {
    appendToAnswer(event.data);
};
```


---

## Security

### Authentication
- Session-based auth with secure cookies
- Password hashing with bcrypt

### Data Isolation
- Row-level security: Users can only access their own PDFs/conversations
- Implemented via SQLAlchemy filters: `Pdf.query.filter_by(user_id=g.user.id)`

---

## Deployment Architecture (Recommended)

```
┌─────────────────┐
│   Load Balancer │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼───┐
│ API  │  │ API  │  (Multiple Flask instances)
│ Node │  │ Node │
└───┬──┘  └──┬───┘
    │         │
    └────┬────┘
         │
┌────────▼────────┐
│   PostgreSQL    │  (Managed database)
└─────────────────┘

┌─────────────────┐
│  Redis Cluster  │  (Managed cache)
└─────────────────┘

┌─────────────────┐
│ Celery Workers  │  (Auto-scaling worker pool)
└─────────────────┘

┌─────────────────┐
│    Pinecone     │  (Managed vector DB)
└─────────────────┘
```


---

## Future Enhancements

### Hybrid Search
Combine semantic search with keyword search (BM25) for better retrieval:
```python
semantic_results = vector_store.similarity_search(query, k=10)
keyword_results = bm25_search(query, k=10)
reranked_results = rerank(semantic_results + keyword_results, query)
```

### Re-ranking
Add a cross-encoder re-ranker after retrieval:
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, doc.page_content) for doc in docs])
reranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
```

### Agentic RAG
Enable self-correction and multi-step reasoning:
```python
# If answer quality is low, retry with different retrieval strategy
if faithfulness_score < 0.7:
    # Try with more chunks or different query reformulation
    retry_with_expanded_context()
```

---

**Last Updated**: February 2026
