---
description: Distributed training in PyTorch
---

# Distributed Training

Distributed training in PyTorch means training a model using multiple GPUs (or machines) in parallel, where each GPU runs its own process on different data and synchronizes gradients automatically (usually via DDP).

When to use distributed training :&#x20;

1. Your **model or batch doesn’t fit on one GPU**
2. You want **faster training** using multiple GPUs / nodes
3. You’re training **large transformers / diffusion / LLM**

In PyTorch, **1 process per GPU and** Each process has its own model copy, works on different data and syncs gradients with others

PyTorch API :&#x20;

`nn.parallel.DistributedDataParallel (DDP)`&#x20;

`torch.distributed.fsdp`

**How does DistributedDataParallel (DDP) works :**-&#x20;

1. Each GPU runs one process
2. Data is sharded using `DistributedSampler`
3. Gradients are all-reduced automatically
4. Model stays fully replicated

When Model does not fit in GPU memory such as (LLMs, ViTs, Diffusion models) then we can move to FSDP (Fully Sharded Data Parallel) from DDP.&#x20;

[https://chatgpt.com/c/69520571-cd7c-8323-ade4-081645a7d7d6](https://chatgpt.com/c/69520571-cd7c-8323-ade4-081645a7d7d6)















References :-&#x20;

* [https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51](https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51)&#x20;
* [https://www.youtube.com/watch?v=toUSzwR0EV8](https://www.youtube.com/watch?v=toUSzwR0EV8)&#x20;
* [https://www.youtube.com/watch?v=S1WQuFW\_hHU](https://www.youtube.com/watch?v=S1WQuFW_hHU)
* [https://www.youtube.com/watch?v=3XUG7cjte2U](https://www.youtube.com/watch?v=3XUG7cjte2U)
* [https://www.youtube.com/watch?v=azLCUayJJoQ](https://www.youtube.com/watch?v=azLCUayJJoQ)&#x20;
* [https://www.youtube.com/watch?v=pHqcHzxx6I8](https://www.youtube.com/watch?v=pHqcHzxx6I8)
* [https://www.youtube.com/watch?v=9TwTKt50ZG8](https://www.youtube.com/watch?v=9TwTKt50ZG8)
* [https://www.youtube.com/watch?v=3XUG7cjte2U](https://www.youtube.com/watch?v=3XUG7cjte2U)&#x20;
* [https://www.youtube.com/watch?v=r3DwMv84H9I](https://www.youtube.com/watch?v=r3DwMv84H9I)
* [https://www.youtube.com/watch?v=hc0u4avAkuM](https://www.youtube.com/watch?v=hc0u4avAkuM)

