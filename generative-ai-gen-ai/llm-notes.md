---
description: Notes & Questions related to Generative AI and LLMs
---

# LLM - Notes

* **What problem does the Transformer solve compared to RNNs/LSTMs?**

Transformers remove sequential recurrence and replace it with self-attention, enabling full parallelization during training.

RNNs:

Process tokens sequentially → slow

Struggle with long-range dependencies (vanishing gradients)

Transformers:

Compute attention across all tokens simultaneously

Capture long-range dependencies directly

Scale better with data and compute

The key breakthrough: attention replaces recurrence.



* Explain self-attention mathematically
* Why use multi-head attention instead of one big attention?
* What happens if we reduce the number of heads?
* Why do we need positional encoding?&#x20;
* Why sinusoidal positional encoding?
* Why do we use causal masking in LLMs?
* What’s the difference between encoder-only, decoder-only, encoder-decoder?
* What loss is used to train LLMs?
* What are scaling laws in LLMs?
* What limits Transformer context length?
* Why is LayerNorm used before attention in modern LLMs?
* Why are residual connections critical?
* If attention is so powerful, why do we still need MLP layers?
* What exactly is stored in the KV cache during inference ?
