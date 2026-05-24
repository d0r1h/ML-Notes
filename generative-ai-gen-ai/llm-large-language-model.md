---
description: Transformer Architecture
---

# LLM (Large Language Model)

Large Language Model are based on the famous Transformer architecture and knowing in/out of that is very important.

### **Transformer (LLM) Training Pipeline**

The general process of creating an LLM includes pre training and fine tuning. The term "pre" in "pre training" refers to the initial phase where a model like an LLM is trained on a

large, diverse dataset to develop a broad understanding of language. This pre trained model then serves as a foundational resource that can be further refined through fine tuning, a

process where the model is specifically trained on a narrower dataset that is more specific to particular tasks or domains.

`Pre-training → Fine-tuning → RLHF / Alignment → Deployment`

![](<../.gitbook/assets/unknown (7).png>)

#### **Pre Training LLM**

Through Pre training LLM learns general language understanding from massive unlabeled text data using self-supervised learning. During pre-training LLMs are fed with no task-specific labels; it's just raw internet-scale text.&#x20;

For example :- The capital of France is \_\_\_, model learns to predict target Paris.&#x20;

There are two main pre-training approaches :- <br>

1. Causal Language Modeling (Predicting next word/token) → Used in GPT style models, this is called autoregressive modeling.&#x20;
2. Masked Language Modeling → Used in Bert style model for example The capital of \[MASK] is Paris.&#x20;

Pre-training Pipeline <br>

* Data Collection & Clearing&#x20;
* Tokenization \[BPE, WordPiece, SentencePiece] [https://www.youtube.com/watch?v=zduSFxRajkE\&t=2098s](https://www.youtube.com/watch?v=zduSFxRajkE\&t=2098s)&#x20;
* Transformer training → Model learns the contextual representations. \[ Multi-head self-attention |  Feed-forward layers | Layer norm | Residual connections]
* Loss Function → Cross Entropy Loss \[We minimize prediction error over billions of tokens.]

During Pre training of the model, it not just learns the grammar but also Syntax, Semantics, World knowledge, Reasoning patterns, Code structure, Multi-lingual alignment and this all from predicting next tokens.&#x20;

“LLM pre-training is a self-supervised learning phase where a transformer model is trained on massive unlabeled text corpora to predict next or masked tokens using cross-entropy loss. It enables the model to learn general language representations and world knowledge before downstream fine-tuning.”

Now we are saying that models learn all of this just by predicting tokens, but that is not happening randomly right because in order to minimize loss across trillions of tokens, the model must extract statistical structure from language.

For example :- “The capital of France is \_\_ ”. The only way to consistently predict “Paris” is to encode:

* What “capital” means
* That France is a country
* That countries have capitals
* That Paris is France’s capital

That’s structured knowledge.

During training Inside a transformer:

* Early layers learn syntax (parts of speech, sentence structure)
* Middle layers learn semantics (who did what to whom
* Deeper layers learn abstract relationships and reasoning

And this works because language has structure and if you compress language optimally (minimize prediction error), you must discover latent structure. To predict words correctly across contexts, the model must cluster words with similar meanings into similar vector spaces.&#x20;

Prediction pressure → structure discovery → representation learning.

Next-token prediction forces the model to capture the statistical structure of language. Because language reflects syntax, semantics, and world knowledge, minimizing prediction loss requires the model to form hierarchical internal representations of grammar, meaning, and reasoning patterns. As scale increases, these representations become increasingly abstract and generalize.

#### Post Training / Fine Tuning

After pretraining (next-token prediction on massive internet data), the model knows the language, facts, and patterns but it doesn’t know how to behave properly. Post-training (sometimes referred to as “alignment”) is a key component of modern LLMs, and it is the way to “teach” models how to answer in a way that humans like, and how to reason.

1. Supervised Fine Tuning (SFT)&#x20;
2. RLHF (Reinforcement Learning from Human Feedback)
3. Direct Preference Optimization (DPO)
4. Constitutional AI (Used by Anthropic)

In supervised fine tuning we create examples of question and answers and then train models on that curated dataset.

### Transformer Architecture

Most modern LLMs rely on the transformer architecture, which is a deep neural network architecture introduced in the 2017 paper Attention Is All You Need. To understand LLMs we briefly have to go over the original transformer, which was originally developed for machine translation, translating English texts to German and French.

![](<../.gitbook/assets/unknown (8).png>)

**Encoder** → Encode input tokens into vectors, it processes the input text and produces an embedding representation (a numerical representation that captures many different factors in different dimensions) of the text.

**Decoder** → Generate or produce the output, the decoder module takes these encoded vectors and generates the output text from them.

**Self Attention** → Allows the model to weigh the importance of different word/tokens relative to each other, it enables the model to learn long term dependency. Both the encoder and decoder consist of many layers connected by a self-attention mechanism.

Traditional models (RNNs, LSTMs) process tokens sequentially but in Transformer instead of processing sequentially, let every word directly look at every other word and decide how important they are; this is the core idea behind Self Attention.&#x20;

For each token self attention asks :-&#x20;

* Which other tokens are important to me?
* How much should I focus on them?
* How do I combine their information?

### Building a Large Language Model (LLM)

<img src="../.gitbook/assets/unknown (9).png" alt="" height="324" width="685">























References :-&#x20;

1. [Building LLMs from scratch ](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)&#x20;
2. [Build a Large Language Model (From Scratch) - YouTube](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11)&#x20;
3. [Stanford CS336 Language Modeling from Scratch I 2025 - YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)
4. [Create a Large Language Model from Scratch with Python – Tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8)
5. Book : Build a Large Language Model (From Scratch) by Sebastian Raschka
6. [Stanford CS336](https://stanford-cs336.github.io/spring2025/)&#x20;
7. [LLM Course](https://huggingface.co/learn/llm-course/e)&#x20;
8. [LLM Transformer Model Visually Explained](https://poloclub.github.io/transformer-explainer/)
9. [https://dev.to/jps27cse/how-transformers-work-inside-an-llm-step-by-step-1k3g](https://dev.to/jps27cse/how-transformers-work-inside-an-llm-step-by-step-1k3g)
