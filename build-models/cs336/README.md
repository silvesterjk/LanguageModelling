# CS336: Language Modeling from Scratch

This repository contains my implementations for all assignments from Stanford's CS336: Language Modeling from Scratch course (Spring 2025). The course covers the fundamentals of building, training, and deploying large language models.

## Course Overview

CS336 provides a comprehensive, hands-on approach to understanding language models by implementing them from the ground up. Topics include model architecture, systems optimization, scaling laws, data processing, and alignment techniques.

## Assignments

### [Assignment 1: Basics](./assignment1-basics)
**Core Topics:** Tokenization, Transformer Architecture, Language Model Training

Implementation of fundamental language modeling components:
- Byte Pair Encoding (BPE) tokenization
- Multi-head self-attention mechanisms
- Transformer decoder architecture
- Basic training loops and optimization
- Training on TinyStories and OpenWebText datasets

> **Note:** CPU parallel tokenizer training is not implemented in this assignment.

### [Assignment 2: Systems](./assignment2-systems)
**Core Topics:** Performance Optimization, Distributed Training, Systems Engineering

Deep dive into efficient LM training systems:
- Flash Attention and kernel-level optimizations
- Mixed precision training (fp16/bf16)
- Memory profiling and optimization
- Distributed data parallelism
- Model parallelism techniques

> **Note:** Distributed parallel training components are not implemented in this assignment.

### [Assignment 3: Scaling](./assignment3-scaling)
**Core Topics:** Scaling Laws, Compute-Optimal Training, Model Scaling

Exploration of scaling behaviors in language models:
- Empirical investigation of scaling laws
- Compute-optimal training strategies
- Model size vs. training data trade-offs
- Performance prediction across scales

> **Note:** Due to the lack of official API access, this assignment could not be fully tested. It is recommended to use custom synthetic datasets or publicly available datasets for verification.

### [Assignment 4: Data](./assignment4-data)
**Core Topics:** Data Curation, Quality Filtering, Dataset Construction

Building high-quality training datasets:
- Web scraping and data collection
- Content filtering and deduplication
- Quality classification models
- Data pipeline construction
- Training on curated datasets

> **Note:** This assignment was completed using smaller datasets. Large-scale dataset training was not performed.

### [Assignment 5: Alignment](./assignment5-alignment)
**Core Topics:** RLHF, Preference Learning, Model Safety

Aligning language models with human preferences:
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)
- Reward modeling
- Safety alignment techniques
- Instruction tuning

> **Note:** The supplemental assignment on safety RLHF has not been implemented yet. Stay tuned for future updates!

## Repository Structure

```
.
├── assignment1-basics/      # Fundamental LM implementation
├── assignment2-systems/     # Systems optimization
├── assignment3-scaling/     # Scaling law investigations
├── assignment4-data/        # Data processing pipeline
├── assignment5-alignment/   # Alignment techniques
└── README.md               # This file
```

Each assignment directory contains:
- Complete source code implementation
- Unit tests
- Assignment handout (PDF)
- Assignment-specific README with setup instructions

## Technical Stack

- **Language:** Python 3.11+
- **Deep Learning:** PyTorch
- **Dependency Management:** [uv](https://github.com/astral-sh/uv)
- **Testing:** pytest
- **Key Libraries:**
  - `torch` - Deep learning framework
  - `flash-attn` - Optimized attention implementation
  - `einops` - Tensor operations
  - `wandb` - Experiment tracking

## Setup

### Prerequisites

1. Install `uv` for dependency management:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via pip
pip install uv

# or via Homebrew
brew install uv
```

2. Python 3.11 or higher

### Running Individual Assignments

Each assignment is self-contained. Navigate to the assignment directory and use `uv run`:

```bash
cd assignment1-basics
uv run pytest                    # Run tests
uv run python scripts/train.py   # Run training script
```

Dependencies are automatically installed based on each assignment's `pyproject.toml`.

## Key Learning Outcomes

Through these assignments, I gained practical experience in:

1. **Model Architecture:** Implementing Transformers from scratch, understanding attention mechanisms and layer normalization
2. **Training Infrastructure:** Building robust training loops, implementing gradient accumulation and mixed precision training
3. **Systems Optimization:** Profiling memory usage, optimizing compute kernels, implementing distributed training
4. **Scaling Principles:** Understanding how model performance scales with compute, data, and parameters
5. **Data Engineering:** Constructing high-quality datasets through filtering, deduplication, and quality assessment
6. **Alignment Techniques:** Implementing RLHF and preference optimization to align models with human values

## Course Information

- **Course:** CS336 - Language Modeling from Scratch
- **Institution:** Stanford University
- **Term:** Spring 2025
- **Course Website:** [https://stanford-cs336.github.io/spring2025/](https://stanford-cs336.github.io/spring2025/)

## Notes

- Large files (datasets, model checkpoints, trained models) have been excluded from this repository
- Each assignment includes comprehensive unit tests to verify implementations
- For detailed assignment requirements, refer to the PDF handouts in each assignment directory

## License

This code is provided for educational purposes. Please refer to individual assignment directories for specific licensing information.

---

