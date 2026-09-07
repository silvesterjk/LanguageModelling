# Model Quantization with Unsloth

This directory contains production-ready, standalone quantization scripts utilizing [Unsloth](https://github.com/unslothai/unsloth) and its surrounding ecosystem. Each script implements a specific quantization method with full CLI argument parsing, Hugging Face Hub integration, and error handling.

---

## Quantization Methods Quick Reference

| Script | Method | Target Precision | Use Case / Target Runtime | Accuracy Retention |
| :--- | :--- | :--- | :--- | :--- |
| [`quantize_gguf_q4km.py`](./quantize_gguf_q4km.py) | GGUF Q4_K_M | 4-bit (medium) | `llama.cpp`, Ollama, LM Studio, vLLM | ⭐⭐⭐⭐ (Very High) |
| [`quantize_gguf_q5km.py`](./quantize_gguf_q5km.py) | GGUF Q5_K_M | 5-bit (medium) | `llama.cpp`, Ollama, High-quality local inference | ⭐⭐⭐⭐⭐ (Near FP16) |
| [`quantize_gguf_q8_0.py`](./quantize_gguf_q8_0.py) | GGUF Q8_0 | 8-bit | `llama.cpp`, Ollama, Lossless CPU/GPU inference | ⭐⭐⭐⭐⭐ (Lossless) |
| [`quantize_gguf_imatrix.py`](./quantize_gguf_imatrix.py) | GGUF IQ + imatrix | 1.5 - 4-bit | Extreme compression, Mobile/edge devices | ⭐⭐⭐⭐ (Optimized) |
| [`quantize_bnb_4bit.py`](./quantize_bnb_4bit.py) | BitsAndBytes NF4 | 4-bit NormalFloat | QLoRA fine-tuning, Transformers, Unsloth | ⭐⭐⭐⭐ (Very High) |
| [`quantize_bnb_8bit.py`](./quantize_bnb_8bit.py) | BitsAndBytes 8-bit | 8-bit (LLM.int8()) | Outlier-safe GPU inference, Transformers | ⭐⭐⭐⭐⭐ (Near Lossless) |
| [`quantize_dynamic_4bit.py`](./quantize_dynamic_4bit.py) | Unsloth Dynamic 4-bit | Mixed 4/16-bit | Vision models, Small LLMs (<7B), Minimal loss | ⭐⭐⭐⭐⭐ (Near Lossless) |
| [`quantize_merged_16bit.py`](./quantize_merged_16bit.py) | Merged 16-bit | FP16 / BF16 | Dequantization for vLLM, SGLang, AWQ/GPTQ base | ⭐⭐⭐⭐⭐ (Full Precision) |
| [`quantize_awq.py`](./quantize_awq.py) | AutoAWQ (W4A16) | 4-bit Activation-aware | High-throughput serving with vLLM & SGLang | ⭐⭐⭐⭐⭐ (Very High) |
| [`quantize_gptq.py`](./quantize_gptq.py) | AutoGPTQ | 4-bit / 8-bit Hessian | ExLlamaV2, vLLM GPU inference | ⭐⭐⭐⭐ (High) |
| [`quantize_fp8_torchao.py`](./quantize_fp8_torchao.py) | TorchAO FP8 | 8-bit Float8 (e4m3fn) | NVIDIA Ada/Hopper/Blackwell (RTX 4090, H100) | ⭐⭐⭐⭐⭐ (Near Lossless) |
| [`quantize_ollama.py`](./quantize_ollama.py) | Ollama Export | 4-bit / 5-bit / 8-bit | Instant local serving with `ollama run` | ⭐⭐⭐⭐ (Configurable) |

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** CUDA-enabled GPU (NVIDIA Turing, Ampere, Ada Lovelace, Hopper, or Blackwell) is strongly recommended for fast export and quantization.

---

## Detailed Usage per Method

### 1. GGUF Q4_K_M (Recommended General Quant)
Quantizes the model using 4-bit medium precision, keeping half of the attention and feed-forward weights at higher bit-depth (`Q6_K`).
```bash
python quantize_gguf_q4km.py \
    --model_name_or_path "unsloth/llama-3-8b-Instruct" \
    --output_dir "./models/llama3-8b-q4km" \
    --max_seq_length 2048 \
    --push_to_hub \
    --hub_repo_id "your-username/llama-3-8b-Q4_K_M-GGUF"
```

### 2. GGUF Q5_K_M (High Quality 5-bit)
Higher quality with lower perplexity loss than Q4_K_M while remaining significantly smaller than 8-bit or 16-bit.
```bash
python quantize_gguf_q5km.py \
    --model_name_or_path "unsloth/mistral-7b-instruct-v0.3" \
    --output_dir "./models/mistral-7b-q5km"
```

### 3. GGUF Q8_0 (Near-Lossless 8-bit)
Converts model to symmetrical 8-bit GGUF with near-instant conversion and virtually zero degradation.
```bash
python quantize_gguf_q8_0.py \
    --model_name_or_path "unsloth/Qwen2.5-7B-Instruct" \
    --output_dir "./models/qwen-7b-q8_0"
```

### 4. GGUF Dynamic Quantization with Importance Matrix (imatrix)
Applies dynamic IQ quantization using an importance matrix computed from calibration text.
```bash
python quantize_gguf_imatrix.py \
    --model_name_or_path "unsloth/llama-3-8b-Instruct" \
    --output_dir "./models/llama3-8b-iq4nl" \
    --quantization_method "iq4_nl" \
    --imatrix_file "./calibration.imatrix"
```

### 5. BitsAndBytes 4-bit (NF4 Merged Checkpoint)
Merges LoRA adapters directly into the 4-bit base weights without dequantization drift.
```bash
python quantize_bnb_4bit.py \
    --model_name_or_path "unsloth/llama-3-8b" \
    --output_dir "./models/llama3-8b-bnb-4bit" \
    --save_method "merged_4bit"
```

### 6. BitsAndBytes 8-bit (LLM.int8())
Loads and runs model with 8-bit outlier vector decomposition.
```bash
python quantize_bnb_8bit.py \
    --model_name_or_path "unsloth/Qwen2.5-7B" \
    --output_dir "./models/qwen-7b-bnb-8bit"
```

### 7. Unsloth Dynamic 4-bit (Selective Layer Precision)
Inspects and exports models using Unsloth's selective dynamic 4-bit precision, preserving sensitive layers in 16-bit.
```bash
python quantize_dynamic_4bit.py \
    --model_name_or_path "unsloth/Llama-3.2-1B-Instruct-bnb-4bit" \
    --output_dir "./models/llama-3.2-dynamic-4bit"
```

### 8. Merged 16-bit Dequantization (FP16 / BF16 Safetensors)
Dequantizes 4-bit weights and fuses LoRA adapters into standard Hugging Face 16-bit safetensors. Essential for exporting to vLLM, SGLang, or as an input to AWQ/GPTQ.
```bash
python quantize_merged_16bit.py \
    --model_name_or_path "path/to/fine-tuned-lora" \
    --output_dir "./models/merged-16bit-hf" \
    --max_shard_size "5GB"
```

### 9. AutoAWQ 4-bit Quantization
Converts an Unsloth model into 4-bit Activation-aware Weight Quantization (AWQ) for ultra-fast vLLM serving.
```bash
python quantize_awq.py \
    --model_name_or_path "unsloth/llama-3-8b-Instruct" \
    --output_dir "./models/llama3-8b-awq" \
    --w_bit 4 \
    --q_group_size 128 \
    --calib_dataset "pileval"
```

### 10. AutoGPTQ Quantization
Converts an Unsloth model into 4-bit or 8-bit GPTQ format using second-order Hessian error minimization.
```bash
python quantize_gptq.py \
    --model_name_or_path "unsloth/llama-3-8b-Instruct" \
    --output_dir "./models/llama3-8b-gptq" \
    --bits 4 \
    --group_size 128
```

### 11. PyTorch TorchAO FP8 Quantization
Applies native PyTorch Float8 quantization for modern NVIDIA architectures (Ada Lovelace, Hopper, Blackwell).
```bash
python quantize_fp8_torchao.py \
    --model_name_or_path "unsloth/llama-3-8b-Instruct" \
    --output_dir "./models/llama3-8b-fp8" \
    --fp8_mode "float8_weight_only"
```

### 12. Ollama Direct Export with Modelfile
Quantizes to GGUF and automatically synthesizes a ready-to-run Ollama `Modelfile`.
```bash
python quantize_ollama.py \
    --model_name_or_path "unsloth/llama-3-8b-Instruct" \
    --output_dir "./models/llama3-ollama" \
    --quantization_method "q4_k_m" \
    --ollama_model_name "my-llama3:latest" \
    --auto_create
```
