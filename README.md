# Large Language Models

A hands-on learning repository for understanding how large language models are
built, trained, evaluated, quantized, and eventually served. It currently
combines course assignments, book companion code, practical quantization
scripts, and notes on AI engineering.

## Repository contents

| Area | What is available | Status |
| --- | --- | --- |
| [`build-models/brm-book`](./build-models/brm-book) | Chapters 2–8 and appendices C–G from *Build a Reasoning Model (From Scratch)* | Available |
| [`build-models/cs336`](./build-models/cs336) | Stanford CS336 Spring 2025 assignments 1–5 | Available |
| [`quantize-models`](./quantize-models) | Standalone scripts for GGUF, BitsAndBytes, Dynamic 4-bit, AWQ, GPTQ, TorchAO FP8, merged 16-bit, and Ollama exports | Available |
| [`ai-engineering`](./ai-engineering) | Notes on context engineering, RAG, and related agent-system concepts | In progress |
| [`post-training-models`](./post-training-models) | Future post-training material | In progress |
| [`serve-models`](./serve-models) | Future model-serving material | In progress |
| [`host-models`](./host-models) | Future model-hosting material | In progress |



## Attribution and license

- The reasoning-model materials are derived from Sebastian Raschka's
  [`rasbt/reasoning-from-scratch`](https://github.com/rasbt/reasoning-from-scratch)
  and retain their upstream attribution and license information.
- The CS336 assignment implementations are sourced from
  [`Louisym/Stanford-CS336-spring25`](https://github.com/Louisym/Stanford-CS336-spring25).
  Refer to the assignment directories for any component-specific terms.
- Original material in this repository is covered by the root
  [Apache License 2.0](./LICENSE), except where an included component states
  otherwise.
