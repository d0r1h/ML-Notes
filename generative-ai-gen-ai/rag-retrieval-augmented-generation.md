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

## **Core of any RAG system at high level is 3 things :**

1. **Indexing** :- Storing data (building database - in context of LLM we use vector databases)
2. **Retrieval** :- Extracting data / document relevant to user query&#x20;
3. **Generation** :- LLM generating response based on the question asked and data

<img src="../.gitbook/assets/unknown (21).png" alt="Any RAG system architecture" height="334" width="708">

At high level rag has only three components as we saw above, but within those three components there is a lot that goes to make the system efficient which makes any simple rag system an Advance RAG system and following are the landscape of the same.

<img src="../.gitbook/assets/unknown (22).png" alt="Advance RAG system Landscape" height="680" width="916">

User Query | Query Transformation / Translation | Routing | Query constructions | Indexing | Retrieval Generation | Answer

### Indexing&#x20;

* To get the answers of user queries from the documents we need to convert them into numbers so that LLM can look into it and find the relevant answers from that.
* Because machine learning models understand numbers (not text, image, or audio) and we need to store them such that it’s easy to find relevant documents which are most similar to user query and this can be done multiple ways and cosine similarity is one of them.&#x20;

<img src="../.gitbook/assets/unknown (23).png" alt="" height="310" width="935">

* We can convert documents and user query into numbers though Embedding and then match both based on similarity on those embedding.&#x20;
* First we need to split the document because of the limited context window on embedding models (LLM) and then compress it into vectors which captures the semantic meaning of these documents.

<img src="../.gitbook/assets/unknown (24).png" alt="" height="338" width="1008">

### Retrieval <a href="#docs-internal-guid-cee4866d-7fff-cf7a-a146-9cb93bbcda56" id="docs-internal-guid-cee4866d-7fff-cf7a-a146-9cb93bbcda56"></a>

* Retrieval is powered via similarity search.&#x20;

![](<../.gitbook/assets/unknown (25).png>)

### Generation

* In particular, this involves retrieving documents but then passing them into the LLM context window so that the LLM can reason about them.&#x20;

<img src="../.gitbook/assets/unknown (26).png" alt="" height="613" width="844">

* This uses the idea of a prompt, which is a template that includes placeholders that we can use with our particular retrieved docs and question.  …

<img src="../.gitbook/assets/unknown (27).png" alt="" height="399" width="793">





## Query Translation  <a href="#docs-internal-guid-cee4866d-7fff-cf7a-a146-9cb93bbcda56" id="docs-internal-guid-cee4866d-7fff-cf7a-a146-9cb93bbcda56"></a>

It’s the first stage of an advanced RAG Pipeline and the goal of query translation is to take an input user question and to translate in some way in order to improve retrieval and we need this because user queries can be ambiguous and if the query is poorly written we won't retrieve the proper documents from our index because we're typically doing semantic similarity search between the query and our documents.

<img src="../.gitbook/assets/unknown (28).png" alt="" height="302" width="633">

There are few different approaches to tackle this problem :-&#x20;

1. Query Rewriting&#x20;
2. Decomposition&#x20;
3. Step-back prompting &#x20;
4. HyDE

### **Query** Rewriting&#x20;

Taking a query and re-framing it like writing it from a different perspective using approaches like multi-query or rag Fusion.

#### **Multi Query**&#x20;

In this we're taking a user query and breaking it down into a few differently worded questions from different perspectives. And we do this because it is possible that the way a question is initially worded once embedded, it may not be well aligned or in close proximity in this High dimensional embedding space to a document (vector store) that we want to retrieve. So by kind of rewriting it in a few different ways you actually increase the likelihood of actually retrieving the document that you really want to.

<img src="../.gitbook/assets/unknown (29).png" alt="" height="378" width="902">

<img src="../.gitbook/assets/unknown (30).png" alt="" height="251" width="794">

#### **RAG Fusion**&#x20;

[Paper Link](https://arxiv.org/pdf/2402.03367) | [Code Link](https://github.com/Raudaschl/rag-fusion/blob/master/main.py)

RAG-fusion is an approach that re-writes a question from multiple perspectives (multi-query), performs retrieval on each re-written question, and performs reciprocal rank fusion on the results from each retrieval, giving a consolidated ranking.

1. Query Generation: The system starts by generating multiple queries from a user's initial query&#x20;
2. Vector Search: Conducts vector-based searches on each of the generated queries to retrieve relevant documents from a predefined set.
3. Reciprocal Rank Fusion: Applies the Reciprocal Rank Fusion algorithm to re-rank the documents based on their relevance across multiple queries.
4. Output Generation: Produces a final output consisting of the re-ranked list of documents.

### Decomposition

[Paper Link](https://arxiv.org/pdf/2205.10625) | [Paper Link](https://arxiv.org/pdf/2212.10509)

Query decomposition is a strategy to improve question-answering by breaking down a question into sub-questions. These can either be solved sequentially or independently answered followed by consolidation into a final answer.&#x20;

In this we combine two approaches :- Least-To-Most (Decompose problem into sub-problems, solve sequentially) and IR-COT (Interleave retrieval with Chain of Thoughts)

<img src="../.gitbook/assets/unknown (32).png" alt="" height="237" width="697">

### **Step Back Prompting**&#x20;

[Paper Link](https://arxiv.org/pdf/2310.06117)

Step-back prompting is an approach to improve retrieval that builds on chain-of-thought reasoning. From a question, it generates a step-back (higher level, more abstract) question that can serve as a precondition to correctly answering the original question. This is especially useful in cases where background knowledge or more fundamental understanding is helpful to answer a specific question.

<img src="../.gitbook/assets/unknown (33).png" alt="Example from TimeQA where the high-level concept of education history is a result of the abstraction. Left: PaLM-2L fails to answer the original question. Chain-of-Thought prompting ran into errors during intermediate reasoning steps (highlighted as red). Right: PaLM-2L successfully answers the question via STEP-BACK PROMPTING" height="279" width="955">

### HyDE

[Paper Link ](https://arxiv.org/pdf/2212.10496)

HyDE (Hypothetical Document Embedding) is an approach to improve retrieval that generates hypothetical documents that could be used to answer the user input question. These documents, drawn from the LLMs knowledge, are embedded and used to retrieve documents from an index. The idea is that hypothetical documents may be better aligned with the indexed documents than the raw user question.

<img src="../.gitbook/assets/unknown (34).png" alt="" width="375">

![](<../.gitbook/assets/unknown (35).png>)

## Routing

[Paper Link ](https://arxiv.org/pdf/2505.23052)

Query routing is a strategy that leverages the user's query to determine, from a set of predetermined options, the next course of action. Query routing helps in deciding where to direct that question to. There can be multiple databases and vector stores for your data and the answer could be in any of them. This is where routing comes in. Based on the user query and predefined choices, the LLMs can decide

### **Logical Routing**&#x20;

In this case, we let the LLM decide which part of our knowledge base to refer to, using predefined routes. This is useful for creating non-deterministic chains where the output of one step leads to the next.

<img src="../.gitbook/assets/unknown (36).png" alt="" height="354" width="765">

### **Semantic Routing**&#x20;

<img src="../.gitbook/assets/unknown (39).png" alt="" height="341" width="796">

## Query Structuring (Construction) <a href="#docs-internal-guid-e1b783e5-7fff-72da-d46b-d678626f18e5" id="docs-internal-guid-e1b783e5-7fff-72da-d46b-d678626f18e5"></a>

[Blog Link](https://www.langchain.com/blog/query-construction)

With typical retrieval augmented generation (RAG), a user query is converted into a vector representation. This vector is then compared to vector representations of the source documents to find the most similar ones. This works fairly well for unstructured data, but what about structured data? Most data in the world has some structure. Much of this data lives in relational (e.g., SQL) or graph databases. And even unstructured data often associated structured metadata (e.g., things like the author, genre, data published, etc).

For example, consider the query about movies about aliens in the year 1980. There is a portion (aliens) that we may want to look up semantically, but also a component ("year == 1980") that we want to look up in an exact way.

Query construction is taking a natural language query and converting it into the query language of the database you are interacting with.

<img src="../.gitbook/assets/unknown (40).png" alt="" height="229" width="528">



## Indexing



### **Multi-Representation Indexing**

[Paper Link](https://arxiv.org/pdf/2312.06648) | [Multi-modal RAG on slide decks](https://blog.langchain.com/multi-modal-rag-template/) |  [Multi-Vector Retriever for RAG on tables, text, and images](https://blog.langchain.com/semi-structured-multi-modal-rag/)

Many RAG approaches focus on splitting documents into chunks and returning some number upon retrieval for the LLM. But chunk size and chunk number can be brittle parameters that many users find difficult to set; both can significantly affect results if they do not contain all context to answer a question and to overcome that we can use something like Proposition indexing, which uses an LLM to produce document summaries ("propositions") that are optimized for retrieval.&#x20;

1. multi-vector retriever embeds summaries, but returns full documents to the LLM.
2. parent-doc retriever embeds chunks but returns full documents to the LLM.&#x20;

Idea is to get best of both worlds: use smaller / concise representations (summaries or chunks) to retrieve, but link them to full documents / context for generation.The approach is very general, and can be applied to tables or images: in both cases, index a summary but return the raw table or image for reasoning. This gets around challenges w/ directly embedding tables or images (multi-modal embedding), using a summary as a representation for text-based similarity search.

<img src="../.gitbook/assets/unknown (41).png" alt="" height="316" width="671">



### Raptor

[Paper Link](https://arxiv.org/pdf/2401.18059) | [Building long context RAG with RAPTOR from scratch](https://www.youtube.com/watch?v=jbGchdTL7d0)

RAG systems need to handle "lower-level" questions that reference specific facts found in a single document or "higher-level" questions that distill ideas that span many documents. Handling both types of questions can be a challenge with typical kNN retrieval where only a finite number of doc chunks are retrieved.&#x20;

Raptor addresses this by creating document summaries that capture higher-level concepts. It embeds and clusters documents, and then summarizes each cluster. It does this recursively, producing a tree of summaries with increasingly high-level concepts. The summaries and starting docs are indexed together, giving coverage across user questions.&#x20;

<img src="../.gitbook/assets/unknown (43).png" alt="" height="389" width="507">



### ColBERT

[Paper Link](https://arxiv.org/pdf/2004.12832) | [Blog1](https://hackernoon.com/how-colbert-helps-developers-overcome-the-limits-of-rag) | [Blog2](https://docs.langchain.com/oss/python/integrations/retrievers/ragatouille) | [Blog](https://developer.ibm.com/articles/how-colbert-works/) | [Blog4](https://til.simonwillison.net/llms/colbert-ragatouille) |

Embedding models compress text into fixed-length (vector) representations that capture the semantic content of the document. This compression is very useful for efficient search / retrieval, but puts a heavy burden on that single vector representation to capture all the semantic nuance / detail of the doc. In some cases, irrelevant / redundant content can dilute the semantic usefulness of the embedding.&#x20;

ColBERT is a neat approach to address this with higher granularity embeddings:&#x20;

* Produce a contextually influenced embedding for each token in the document and query.&#x20;
* Score similarity between each query token and all document tokens.&#x20;
* Take the max.&#x20;
* Do this for all query tokens.&#x20;
* Take the sum of the max scores (in step 3) for all query tokens to get the similarity score.&#x20;



## Retrieval



### Ranking

[Cohere Rerank](https://cohere.com/blog/rerank)

1. Re-Rank&#x20;
2. RankGPT
3. RAG-Fusion

<img src="../.gitbook/assets/unknown (44).png" alt="" height="318" width="877">

### **CRAG (Corrective-RAG )**&#x20;

[Paper Link](https://arxiv.org/pdf/2401.15884)  | [Building Corrective RAG from scratch with open-source, local LLMs](https://www.youtube.com/watch?v=E2shqsYwxck)

Corrective-RAG (CRAG) is a strategy for RAG that incorporates self-reflection / self-grading on retrieved documents. This is a Refinement process. &#x20;

* If at least one document exceeds the threshold for relevance, then it proceeds to generation
* Before generation, it performs knowledge refinement
* This partitions the document into "knowledge strips"
* It grades each strip, and filters our irrelevant ones
* If all documents fall below the relevance threshold or if the grader is unsure, then the framework seeks an additional data source.
* It will use web search to supplement retrieval

## Generation

**Retrieval (Self-RAG)**

[Code](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb) | [Paper](https://arxiv.org/pdf/2310.11511)







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
