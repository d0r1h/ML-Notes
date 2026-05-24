---
description: >-
  "RAG" can refer to Retrieval-Augmented Generation, a technique that improves
  AI language models by providing them with external, up-to-date information
  before they generate a response.
---

# RAG (Retrieval Augmented Generation)

**Code Link \[**[**https://github.com/d0r1h/Learn-AI/tree/main/Agentic\_AI/RAG**](https://github.com/d0r1h/Learn-AI/tree/main/Agentic_AI/RAG)**]**&#x20;

Why we need RAG, what is the motivation ? Connecting LLMs to external data.

1. Currently all the LLMs are trained on the publicly available data, but what if we need to work with private or internal data (for organization) then what ?&#x20;
2. We can achieve that by treating LLMs as an operating system and connecting internal data sources through tools.

<img src="../.gitbook/assets/unknown (20).png" alt="" width="375">

**Core of any RAG system at high level is 3 things :**

1. **Indexing** :- Storing data (building database - in context of LLM we use vector databases)
2. **Retrieval** :- Extracting data / document relevant to user query&#x20;
3. **Generation** :- LLM generating response based on the question asked and data

<img src="../.gitbook/assets/unknown (21).png" alt="Any RAG system architecture" height="334" width="708">

At high level rag has only three components as we saw above, but within those three components there is a lot that goes to make the system efficient which makes any simple rag system an Advance RAG system and following are the landscape of the same.

<img src="../.gitbook/assets/unknown (22).png" alt="Advance RAG system Landscape" height="680" width="916">

User Query | Query Transformation / Translation | Routing | Query constructions | Indexing | Retrieval Generation | Answer

### Indexing&#x20;

























**RAG System Pipeline**

User Query → Embedding → Vector Search → Top-k Retrieval → Re-ranking (optional) → Prompt Construction → LLM Generation

**Chunking** :-&#x20;

* Fixed-size chunking
* Recursive chunking
* Semantic chunking
* Sliding window
* Structure-aware chunking (headers, sections)

Why 512 tokens might outperform 1000 tokens

Why overlap helps recall

When overlap hurts performance

How chunk size affects embedding quality

**Embedding**&#x20;

Why cosine similarity works

Why dot product differs

Why normalized embeddings matter

Dense vs sparse embeddings

When hybrid search beats dense-only

Domain-specific embedding finetuning

Embedding drift

How to evaluate embedding quality

**Retrieval Optimization**

Top-k tuning

MMR (Max Marginal Relevance)

Re-ranking with cross-encoders

Hybrid search (BM25 + dense)

Metadata filtering

Tradeoff example:

High k → better recall, worse latency

Low k → faster but risk missing context

**Prompt Construction**

Context ordering strategies

“Lost in the middle” problem

How to structure retrieved chunks

Delimiters vs no delimiters

Citation prompting

Compression prompting

Map-reduce RAG

\
<br>

**Why is your RAG hallucinating?**

* Poor chunking
* Retrieval miss
* Embedding mismatch
* Prompt injection
* Context truncation
* Irrelevant top-k results

**Scaling RAG**

Performance problems:

* Embedding bottleneck
* Vector DB latency
* Large context cost
* Cold start time

**Solutions:**

* Embedding cache
* Response cache
* Async batching
* Quantized inference
* Approximate NN search (HNSW, IVF)

**Evaluation**

* Recall@k
* Precision@k
* MRR
* Faithfulness
* Answer relevance
* Context relevance
* LLM-as-judge evaluation



Project / Reference / Blogs :-&#x20;

1. Building GPT (bot) for Bank :- [https://medium.com/@jyotinigam2370/customer-support-chatbot-using-rag-2934acfa9ea2](https://medium.com/@jyotinigam2370/customer-support-chatbot-using-rag-2934acfa9ea2)
2.  **RAG Idea : Design a chatbot over internal documents**

    Ingestion pipeline | Chunking strategy| Embedding model | Vector DB (FAISS / Pinecone / Weaviate) | Retriever (top-k, MMR) | Re-ranking | Prompt construction  | LLM inference  | Evaluation&#x20;

    Why chunk size matters | Embedding drift | Hybrid search (BM25 + vector) | Hallucination mitigation
3. [https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag](https://blog.dailydoseofds.com/p/5-chunking-strategies-for-rag)&#x20;
4. [https://www.pinecone.io/learn/chunking-strategies/](https://www.pinecone.io/learn/chunking-strategies/)&#x20;
