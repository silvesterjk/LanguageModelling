https://youtu.be/VEmrEo0ZDw8?list=PLPTV0NXA_ZSgMaz0Mu-SjCPZNUjz6-6tN

### 1. Architectural Foundation & Core Hyperparameters

GPT-3 (175B) follows the autoregressive, decoder-only Transformer architecture introduced in OpenAI's *"Language Models are Few-Shot Learners"* (Brown et al., 2020). The parameter count is determined by a small set of structural hyperparameters:

- **Number of Layers (`L`)**: 96 transformer decoder blocks.
- **Hidden / Model Dimension (`d_model`)**: 12,288.
- **Number of Attention Heads (`n_heads`)**: 96.
- **Dimension per Head (`d_head`)**: 
  `d_head = d_model / n_heads = 12,288 / 96 = 128`
- **Feed-Forward Inner Dimension (`d_ff`)**: Standard 4× expansion:
  `d_ff = 4 × d_model = 4 × 12,288 = 49,152`
- **Vocabulary Size (`V`)**: 50,257 tokens (Byte-Pair Encoding / BPE).
- **Context Window (`n_ctx`)**: 2,048 tokens.

---

### 2. Input Embedding Block

Before tokens enter the transformer layers, they are converted into dense vector representations combining token identity and sequence position.

1. **Token Embeddings (`W_e`)**:
   - Maps each discrete token ID in the vocabulary to a `d_model`-dimensional continuous vector.
   - Calculation: `V × d_model = 50,257 × 12,288 = 617,557,984` (~617.56M)

2. **Positional Embeddings (`W_pos`)**:
   - Learns explicit 1D absolute position embeddings up to the maximum context window size.
   - Calculation: `n_ctx × d_model = 2,048 × 12,288 = 25,165,824` (~25.17M)

- **Total Input Parameters**:
  `617,557,984 + 25,165,824 = 642,723,808` (~0.643B)

---

### 3. Anatomy of a Single Transformer Block

Every one of the 96 decoder blocks consists of three primary components:
1. Multi-Head Self-Attention (MHA)
2. Position-wise Feed-Forward Network (FFN / MLP)
3. Pre-Layer Normalization (Pre-LN)

```
Input x
  │
  ├────────────────────────┐ (Residual Connection)
  ▼                        │
LayerNorm (LN1)            │
  ▼                        │
Multi-Head Attention (MHA) │
  ▼                        │
  + ◄──────────────────────┘
  │
  ├────────────────────────┐ (Residual Connection)
  ▼                        │
LayerNorm (LN2)            │
  ▼                        │
Feed-Forward Net (FFN)     │
  ▼                        │
  + ◄──────────────────────┘
  │
Output to Next Block
```

#### A. Multi-Head Self-Attention (MHA)
The attention mechanism projects inputs into Query (`Q`), Key (`K`), and Value (`V`) representations and linearly recombines them through an output projection (`O`):

- **Q, K, V Projections**:
  - Each maps `d_model → d_model`.
  - Weights (3 matrices): `3 × (d_model × d_model) = 3 × (12,288 × 12,288) = 452,984,832`
  - Biases (3 vectors): `3 × d_model = 3 × 12,288 = 36,864`
  - Subtotal: `452,984,832 + 36,864 = 453,021,696`

- **Output Projection (`O`)**:
  - Maps concatenated multi-head outputs back into `d_model`.
  - Weights: `d_model × d_model = 12,288 × 12,288 = 150,994,944`
  - Bias: `d_model = 12,288`
  - Subtotal: `150,994,944 + 12,288 = 151,007,232`

- **Total Attention Parameters per Layer**:
  `4 × d_model² + 4 × d_model = 603,979,776 + 49,152 = 604,028,928` (~0.604B)

#### B. Position-wise Feed-Forward Network (FFN)
A two-layer MLP with a GeLU non-linearity that expands the representation by 4× and projects it back:

- **First Linear Layer (`W_1`, Expansion)**:
  - Projects `d_model → 4 × d_model` (12,288 → 49,152).
  - Weights: `d_model × 4 × d_model = 12,288 × 49,152 = 603,979,776`
  - Biases: `4 × d_model = 49,152`
  - Subtotal: `603,979,776 + 49,152 = 604,028,928`

- **Second Linear Layer (`W_2`, Projection)**:
  - Projects `4 × d_model → d_model` (49,152 → 12,288).
  - Weights: `4 × d_model × d_model = 49,152 × 12,288 = 603,979,776`
  - Biases: `d_model = 12,288`
  - Subtotal: `603,979,776 + 12,288 = 603,992,064`

- **Total FFN Parameters per Layer**:
  `8 × d_model² + 5 × d_model = 1,207,959,552 + 61,440 = 1,208,020,992` (~1.208B)

#### C. Layer Normalization (Pre-LN)
Two LayerNorm operations per transformer block (pre-attention and pre-FFN). Each LayerNorm consists of a learnable scale (gamma) and shift (beta) vector:
- `2 × (2 × d_model) = 4 × 12,288 = 49,152`

---

#### D. Subtotal for a Single Transformer Layer
```
Params_layer = Params_MHA + Params_FFN + Params_LN
             = (4 × d_model² + 4 × d_model) + (8 × d_model² + 5 × d_model) + 4 × d_model
             = 12 × d_model² + 13 × d_model
             = 604,028,928 + 1,208,020,992 + 49,152
             = 1,812,099,072 (~1.812B parameters / layer)
```

---

### 4. Stacking the Layers (The Bulk of Parameters)

Multiplying the single-layer cost across all 96 decoder layers:

`96 × 1,812,099,072 = 173,961,510,912` (~173.96B)

> The 96 transformer blocks constitute **~99.3%** of the entire model's parameter budget.

---

### 5. Output Block & Total Parameter Accounting

1. **Final Layer Normalization**:
   - Normalizes representations from the 96th layer before generating final vocabulary logits.
   - Calculation: `2 × d_model = 2 × 12,288 = 24,576`

2. **Language Modeling Head (Unembedding Matrix)**:
   - Projects hidden states back into vocabulary token logits (`d_model → V`).
   - Matrix size: `d_model × V = 12,288 × 50,257 = 617,557,984` (~617.56M)
   - **Weight Tying vs. Untied**:
     - **Weight-Tied** (shares weights with the input token embedding matrix `W_e`): adds 0 extra parameters.
     - **Untied Head** (standard in GPT-3 175B configuration): adds 617,557,984 parameters.

#### Grand Total Summary Table

| Component | Calculation Formula | Exact Parameter Count | Approximate Count |
| :--- | :--- | :--- | :--- |
| **Token Embeddings (`W_e`)** | `V × d_model` | 617,557,984 | ~0.618 B |
| **Positional Embeddings (`W_pos`)** | `n_ctx × d_model` | 25,165,824 | ~0.025 B |
| **Transformer Blocks (96×)** | `96 × (12 × d_model² + 13 × d_model)` | 173,961,510,912 | ~173.962 B |
| **Final LayerNorm** | `2 × d_model` | 24,576 | ~0.00002 B |
| **Output LM Head (Untied)** | `d_model × V` | 617,557,984 | ~0.618 B |
| **Grand Total (Untied)** | | **175,221,817,280** | **~175.22 Billion** |
| *(Alternative: Tied Head)* | | *174,604,259,296* | *~174.60 Billion* |

---

### 6. Architectural Insights & Vision Transfer (ViT)

- **The 2:1 FFN to Attention Parameter Ratio**:
  - The FFN contains `8 × d_model²` weights, whereas Attention contains `4 × d_model²` weights.
  - In every transformer block, approximately **66.7% (2/3)** of parameters reside in the feed-forward layers, while **33.3% (1/3)** reside in the attention heads.
  - **Intuition**: Attention acts as a dynamic routing mechanism (determining token-to-token relationships), while FFN layers act as associative memory banks (storing factual and semantic world knowledge).

- **Quadratic vs. Linear Scaling**:
  - Parameters scale **quadratically** with hidden dimension (`O(d_model²)`).
  - Sequence length (`n_ctx`) and vocabulary size (`V`) only scale the parameter count **linearly** (`O(n_ctx × d_model)`, `O(V × d_model)`).

- **Transfer to Vision Transformers (ViT)**:
  - The internal block formula (`12 × d_model² + 13 × d_model`) transfers identically to Vision Transformers (e.g., ViT-Base, ViT-Large, ViT-Huge) and Small Language Models (SLMs).
  - The only differences lie in the input/output boundaries:
    - Text token embeddings (`V × d_model`) are replaced by linear patch projections (`(patch_size² × C) × d_model`).
    - The output LM head is replaced by an MLP classification head projecting from `d_model` to the target class count.
