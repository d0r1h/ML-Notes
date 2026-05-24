---
description: Mastering Gen AI
---

# Generative AI (Gen AI)

![](<../.gitbook/assets/unknown (5).png>)

* **Deep Understanding of Transformer Architecture** \
  Why attention > RNNs | Self-attention | Q, K, V intuition | Multi-head attention | Positional encoding | Causal masking | Cross attention
* **Large Language Model Training Concepts**                                                                                          Pretraining vs Fine-tuning | SFT | RLHF | DPO | Instruction tuning | LoRA / QLoRA | PEFT | Quantization (fp16, bf16, int8, fp8) |  Offloading | vLLM | Paged attention | KV cache
* **Tokenization.**                                                                                                                                                                      BPE | SentencePiece | Why tokenization matters | Context window impact | Token explosion problems
* **Inference Optimization**                                                                                                                                                                                Why vLLM > naive HF inference | Continuous batching | KV cache | Paged attention | Throughput vs latency tradeoff | GPU memory planning | Temperature | Top-k & Top-p | Beam search | Sampling strategies | Latency vs throughput tradeoff | GPU vs CPU inference



**GenAI System Design**

Multi-agent systems | Tool use (function calling) | ReAct pattern | Memory management | Long context handling | Guardrails

Retrieval layer | Tool layer | Safety layer | Monitoring | Prompt injection defense



**Notes :-**&#x20;

Speculative decoding | Mixture of Experts | Flash Attention | KV cache compression | Context length extrapolation (RoPE scaling) | Evaluation: BLEU is bad for LLMs | LLM as judge (pros & pitfalls)

* Why does RAG reduce hallucination?
* When would you fine-tune vs use RAG?
* How does LoRA work mathematically?
* What causes hallucination technically?
* How would you evaluate a GenAI system ?
* What is temperature doing mathematically?

How RAG & Agent LLM model works | RAG vs Agent | Shared memory between Agents |  Embedding models |  how to access a very big RAG

How do you deal with PDF's containing images in RAG |  What are the prompt engineering techniques |  Types of chucking |&#x20;

