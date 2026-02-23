# CS336 Lecture 2: PyTorch Primitives & Resource Accounting

Based on: [CS336 Spring 2025 Lecture 02](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json)

## Overview
This lecture covers the primitives needed to train a model, going bottom-up from tensors to models to optimizers to the training loop. A key focus is on **efficiency**, specifically accounting for:
*   **Memory** (GB)
*   **Compute** (FLOPs)

## 1. Motivating Questions (Napkin Math)

Before coding, it is crucial to estimate resource requirements.

### Training Time Estimation
**Question:** How long to train a 70B model on 15T tokens on 1024 H100s?

```python
def motivating_questions():
    # Constants
    h100_flop_per_sec = 1979e12 / 2  # Approx 989 TFLOPS (dense)
    mfu = 0.5  # Model Flops Utilization (typical good value)

    # Formula: 6 * num_params * num_tokens
    # 6 FLOPs per token per parameter (2 forward, 4 backward)
    total_flops = 6 * 70e9 * 15e12

    # FLOPs per day for the cluster
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24

    days = total_flops / flops_per_day
    print(f"Estimated days: {days}")
```

### Max Model Size Estimation
**Question:** Largest model trainable on 8 H100s using AdamW?

```python
    h100_bytes = 80e9  # 80GB VRAM
    num_gpus = 8

    # Naive float32 accounting:
    # 4 bytes (param) + 4 bytes (grad) + 8 bytes (optimizer state: m, v) = 16 bytes/param
    bytes_per_parameter = 16 

    num_parameters = (h100_bytes * num_gpus) / bytes_per_parameter
    
    # Caveat 1: we are naively using float32. 
    # We could use bf16 for params/grads (2+2) and keep extra float32 copy of params (4).
    # This is ZeRO optimization.
    
    # Caveat 2: activations are not accounted for (depends on batch size and sequence length).
```

## 2. Memory Accounting

### Tensor Basics
Tensors are the basic building block for storing everything: parameters, gradients, optimizer state, data, activations.

```python
import torch
import torch.nn as nn

def tensors_basics():
    # From data
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])

    # Common initializations
    x = torch.zeros(4, 8)
    x = torch.ones(4, 8)
    x = torch.randn(4, 8)  # Normal distribution

    # Allocation without initialization (efficient if you overwrite immediately)
    x = torch.empty(4, 8)

    # Custom initialization
    nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)
```

### Tensor Memory Usage
Memory is determined by the number of values and the data type.

#### float32 (Default)
*   4 bytes per element.
*   Standard for scientific computing, but often overkill for DL.

```python
def tensors_memory():
    x = torch.zeros(4, 8)
    assert x.dtype == torch.float32
    assert x.element_size() == 4
    # Memory = numel * element_size = 32 * 4 = 128 bytes
```

#### float16 (fp16)
*   2 bytes per element.
*   Limited dynamic range; can cause underflow (small numbers become 0).

```python
    x = torch.zeros(4, 8, dtype=torch.float16)
    assert x.element_size() == 2
    
    # Underflow example
    x = torch.tensor([1e-8], dtype=torch.float16)
    assert x == 0
```

#### bfloat16 (bf16)
*   2 bytes per element.
*   Same dynamic range as float32 (8-bit exponent), but lower precision (7-bit mantissa).
*   Preferred for DL training (Google Brain, 2018).

```python
    x = torch.tensor([1e-8], dtype=torch.bfloat16)
    assert x != 0  # No underflow!
```

#### fp8
*   1 byte per element.
*   Supported on H100s (E4M3 and E5M2 variants).

## 3. Compute Accounting

### Tensors on GPUs
By default, tensors are on CPU. For massive parallelism, move them to GPU.

```python
def tensors_on_gpus():
    x = torch.zeros(32, 32) # CPU
    
    if torch.cuda.is_available():
        # Move to GPU
        y = x.to("cuda:0")
        
        # Create directly on GPU
        z = torch.zeros(32, 32, device="cuda:0")
```

### Tensor Operations & Storage
PyTorch tensors are pointers into allocated memory with metadata (strides) describing how to access elements.

```python
def tensor_storage():
    x = torch.tensor([
        [0., 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ])
    # Stride: steps to go to next element in each dimension
    assert x.stride(0) == 4 # Next row
    assert x.stride(1) == 1 # Next column
```

### Slicing & Views
Many operations provide a **view** without copying memory. This is efficient but means mutations affect the original tensor.

```python
def tensor_slicing():
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])
    
    # View
    y = x.view(3, 2)
    assert same_storage(x, y) # Helper to check data pointer
    
    # Transpose (also a view)
    y = x.transpose(1, 0)
    
    # Non-contiguous views cannot be reshaped directly
    # Need .contiguous() which forces a copy
    try:
        y.view(2, 3)
    except RuntimeError:
        y = y.contiguous().view(2, 3)
```

### Einops
`einops` provides readable tensor manipulation, replacing confusing `view`, `transpose`, and `permute` calls.

```python
from einops import rearrange, reduce, einsum
from jaxtyping import Float

def einops_examples():
    x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
    y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)
    
    # Einsum: Generalized matrix multiplication
    # Old: x @ y.transpose(-2, -1)
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
    
    # Reduce: Aggregation
    # Old: x.mean(dim=-1)
    z = reduce(x, "... hidden -> ...", "sum")
    
    # Rearrange: Reshaping/Transposing
    # Example: Splitting heads
    x_flat: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
    x_heads = rearrange(x_flat, "... (heads hidden1) -> ... heads hidden1", heads=2)
```

### FLOPs (Floating Point Operations)
*   **FLOPs**: Total operations (measure of work).
*   **FLOP/s**: Operations per second (measure of speed).
*   **MFU (Model FLOPs Utilization)**: Actual FLOP/s / Peak FLOP/s.

Matrix multiplication dominates compute cost.
For $m \times n$ and $n \times p$ matrix multiplication: $2mnp$ FLOPs.

```python
def tensor_operations_flops():
    B, D, K = 1024, 256, 64
    x = torch.ones(B, D)
    w = torch.randn(D, K)
    
    # FLOPs = 2 * B * D * K
    # 1 mult + 1 add per element
    actual_num_flops = 2 * B * D * K
```

### Gradients & Backward Pass
The backward pass computes gradients via the chain rule.
**Cost Rule of Thumb**:
*   Forward: $2N$ FLOPs (where $N$ is parameter count * tokens).
*   Backward: $4N$ FLOPs.
*   Total: $6N$ FLOPs.

```python
def gradients_basics():
    x = torch.tensor([1., 2, 3])
    w = torch.tensor([1., 1, 1], requires_grad=True)
    
    # Forward
    pred_y = x @ w
    loss = 0.5 * (pred_y - 5).pow(2)
    
    # Backward
    loss.backward()
    
    # Gradients populated in .grad attribute
    print(w.grad)
```

## 4. Models (nn.Module)

PyTorch models subclass `nn.Module`. Parameters should be initialized carefully to avoid exploding/vanishing gradients (e.g., Xavier/Kaiming init).

```python
class Linear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Xavier initialization: scale by 1/sqrt(input_dim)
        self.weight = nn.Parameter(
            torch.randn(input_dim, output_dim) / np.sqrt(input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight

class Cruncher(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(dim, dim) for _ in range(num_layers)
        ])
        self.final = Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return x.squeeze(-1)
```

## 5. Training Loop & Best Practices

### Randomness
Set seeds for reproducibility.

```python
import random
import numpy as np

def set_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
```

### Data Loading
Use `np.memmap` for large datasets to avoid loading everything into RAM. Pin memory for faster CPU->GPU transfer.

```python
def get_batch(data: np.array, batch_size: int, sequence_length: int, device: str):
    # Sample random positions
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))
    
    # Index into data
    x = torch.tensor([data[start:start + sequence_length] for start in start_indices])
    
    # Pin memory for async transfer
    if torch.cuda.is_available():
        x = x.pin_memory()
        
    return x.to(device, non_blocking=True)
```

### Optimizer
Custom implementation of SGD and AdaGrad to understand internals.

```python
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None: continue
                p.data -= lr * p.grad.data

class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                grad = p.grad.data
                
                # Accumulate squared gradients
                g2 = state.get("g2", torch.zeros_like(grad))
                g2 += torch.square(grad)
                state["g2"] = g2
                
                # Update with adaptive learning rate
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```

### Training Loop
Standard boilerplate: Forward -> Loss -> Backward -> Optimizer Step.

```python
import torch.nn.functional as F

def train_loop(model, optimizer, num_steps=100):
    model.train()
    for t in range(num_steps):
        x, y = get_batch(B=32) # Pseudocode
        
        # Forward
        pred_y = model(x)
        loss = F.mse_loss(pred_y, y)
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

### Checkpointing
Save model and optimizer state to resume training later.

```python
def save_checkpoint(model, optimizer, path="checkpoint.pt"):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path="checkpoint.pt"):
    return torch.load(path)
```

### Mixed Precision Training
Use `float32` for parameters/gradients (stability) and `bfloat16`/`fp8` for activations (speed/memory).

```python
# PyTorch AMP (Automatic Mixed Precision)
def mixed_precision_step(model, x, y, optimizer):
    # Autocast runs forward pass in lower precision where safe
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred_y = model(x)
        loss = F.mse_loss(pred_y, y)
    
    # Backward/Optimizer remains standard (gradients auto-cast back if needed)
    # Note: For float16, GradScaler is needed. For bfloat16, usually not.
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Helper Functions

```python
def same_storage(x, y):
    return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()

def get_num_parameters(model):
    return sum(param.numel() for param in model.parameters())

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
