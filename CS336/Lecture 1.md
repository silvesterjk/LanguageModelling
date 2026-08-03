```
Overview, Architecture, & Tokenization
```

### 1. Course Philosophy & Fundamental Trade-offs

- **Building "From Scratch"**: Prompts and fine-tuning operate above leaky abstractions. Fundamental language model research requires hands-on engineering across the full technology stack.
    
- **Academic vs. Frontier Models**:
    
    - Frontier models (GPT-4 class) cost $100M–$1B+ and omit engineering details due to competitive/safety concerns.
        
    - Small-scale models differ structurally: compute spend on MLP layers grows from ~44% at small scale to ~80% at 175B scale.
        
    - Certain capabilities (like in-context learning) emerge only beyond critical scale thresholds.
        
- **Transferable Knowledge Domains**:
    
    - **Mechanics**: How architectures, model parallelism, and GPU execution function.
        
    - **Mindset**: Hardware efficiency, profiling, and benchmarking.
        
    - **Intuitions**: Empirical data and modeling decisions (scale-dependent and largely experimental).
        
- **The Bitter Lesson & Efficiency**:
    
    - Scaling is driven by algorithms that scale efficiently, not just hardware additions alone.
        
    - General formula: \text{Model Performance} = \text{Efficiency} \times \text{Resources}.
        
    - Algorithmic efficiency gains contributed a 44x improvement in computer vision compute efficiency between 2012 and 2019.
        

### 2. Historical Evolution of LMs

- **Foundations**: Shannon entropy (1950s), n-gram statistical models for translation and speech.
    
- **Neural Architecture Trajectory**:
    
    - LSTMs (1990s) \to Feed-Forward Neural LMs (Bengio 2003) \to Seq2Seq, Adam, Attention, and Transformers (2017).
        
    - Mixture-of-Experts (MoE) and distributed model parallelism.
        
- **Paradigm Shifts**:
    
    - Fine-tuning era (ELMo, BERT, T5) \to Pre-training & In-Context Learning (GPT-1 to GPT-3).
        
    - Open Weight Ecosystem: Early replications (EleutherAI, OPT-175B, BLOOM) \to High-performance open models (LLaMA, Qwen, DeepSeek) \to Fully open artifacts (OLMo, Marin).
        
    - Shift towards autonomous reasoning agents requiring extended context windows and tool orchestration.
        

### 3. Compute vs. Memory (Hardware Profiling)

- **Separation of Compute and Memory**:
    
    - GPU memory (High Bandwidth Memory / HBM) is distinct from compute cores (Tensor Cores).
        
    - Data must constantly move from HBM to compute units and back, making data transfer the primary operational bottleneck.
        
- **Hardware Profile (e.g., NVIDIA B200 Spec)**:
    
    - **BF16 Compute Performance**: ~2.25 PFLOPS.
        
    - **HBM Bandwidth**: ~8.0 TB/s.
        
- **Roofline Analysis**:
    
    - Analytical framework used to determine if a kernel/layer is **compute-bound** (limited by TFLOPS) or **memory-bandwidth bound** (limited by memory access speeds).
        
    - Most standard Transformer operations (especially token-by-token decoding) are heavily memory-bound.
        
- **Optimization Strategies**:
    
    - **Operator Fusion**: Combines sequential operations into a single kernel to avoid writing intermediate activations back to HBM.
        
    - **Tiling**: Loads sub-matrices into fast SRAM to perform local calculations before writing back to HBM (foundational principle of FlashAttention).
        

### 4. Course Structure & Syllabus Outline

1. **Assignment 1 (Basics)**: Implement BPE Tokenizer, Transformer, Training Loop, and FLOP/Memory Resource Accounting.
    
2. **Assignment 2 (Systems)**: Custom Triton Kernels, Operator Fusion, Model/Data Parallelism, and Inference Systems.
    
3. **Assignment 3 (Scaling Laws)**: Fitting Chinchilla Scaling Laws, Hyperparameter Transfer, and Budget-Constrained Optimization.
    
4. **Assignment 4 (Data Pipeline)**: Web Crawl Extraction, Quality Filtering, MinHash Deduplication, and Evaluation.
    
5. **Assignment 5 (Alignment & RL)**: Preference Optimization (DPO, GRPO) and Multi-Node RL Rollout Infrastructure.
    

### 5. Deep-Dive: Tokenization

- **Definition & Core Requirement**: Maps Unicode strings to integer token IDs and vice versa. Requires exact lossless round-trip recovery (decode(encode(text)) == text).
    
- **Key Efficiency Metric**:
    
    $$\text{Compression Ratio} = \frac{\text{Number of Raw Bytes}}{\text{Number of Generated Tokens}}$$
    
    A higher compression ratio shortens the sequence length N, directly reducing the O(N^2) compute and memory cost of self-attention.
    

#### Trade-Offs Across Approaches

- **Character-Level**: Vocab size ~150k. Inefficient vocabulary allocation and low compression ratio.
    
- **Byte-Level**: Fixed vocab size of 256 (bytes 0–255). Zero Out-Of-Vocabulary (OOV) issues, but compression ratio is 1.0 (leading to long sequence lengths and high compute overhead).
    
- Word-Level: High compression, but unbounded vocabulary size and severe failure on Out-Of-Vocabulary () words.
    
- **Byte-Pair Encoding (BPE)**: Subword-level, data-driven algorithm starting with 256 base byte tokens. Merges common subwords into single tokens while rare words decompose into smaller known sub-units, eliminating unknown tokens.
    

#### BPE Algorithm

1. Represent raw text as a sequence of byte integers (0..255).
    
2. Count frequencies of all adjacent pairs.
    
3. Merge the most frequent pair (T_i, T_j) and assign it a new ID (256, 257, \dots).
    
4. Substitute all occurrences of (T_i, T_j) in the sequence with the new token ID.
    
5. Repeat until the target vocabulary size or merge limit is reached.
    

#### Implementation Factors

- **Pre-tokenization**: Uses regex rules to prevent invalid merges across distinct text categories (e.g., merging punctuation with digits or letters).
    
- **Execution Efficiency**: Iterating over all vocabulary merges in pure Python is too slow for production; optimized implementations use fast lookup structures or compiled languages like Rust/C++.
    

### 6. Summary: Efficiency as the Core Design Constraint

- **Tokenization**: Controls sequence length to minimize O(N^2) attention compute.
    
- **Architecture Design**: Uses techniques like MoE, GQA, and Rotary Embeddings to maximize parameter capacity while bounding FLOPs and KV-cache memory.
    
- **Data Processing**: Prevents wasting gradient updates and memory bandwidth on low-quality/duplicate tokens.
    
- **Scaling Laws**: Prevents wasteful full-scale GPU runs by establishing predictable hyperparameter trajectories at smaller scales.