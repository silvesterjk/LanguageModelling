#!/usr/bin/env python3
"""
Quantize LLM to FP8 using PyTorch TorchAO and Unsloth.

Method: TorchAO FP8 (Float8 Weight-Only or Float8 Dynamic Activation)
Description:
    Quantizes a model using native PyTorch TorchAO FP8 quantization primitives.
    FP8 provides 2x speedups on modern NVIDIA architectures (Ada Lovelace L40/RTX 4090,
    Hopper H100/H200, and Blackwell B200) with minimal accuracy loss compared to BF16/FP16.
    Unsloth integrates with TorchAO for native hardware-accelerated low-precision inference.

Quantization modes:
    - float8_weight_only: Quantizes Linear layer weights to FP8 (e4m3fn).
    - float8_dynamic_activation: Dynamically quantizes activations and weights to FP8 (e4m3fn).

Target deployment: PyTorch 2.4+ on NVIDIA Ada/Hopper/Blackwell GPUs, TensorRT, vLLM.
"""

import argparse
import os
import sys
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize an Unsloth model to FP8 using PyTorch TorchAO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging Face model ID or path to model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_torchao_fp8",
        help="Destination directory to save the FP8 quantized weights and tokenizer.",
    )
    parser.add_argument(
        "--fp8_mode",
        type=str,
        default="float8_weight_only",
        choices=["float8_weight_only", "float8_dynamic_activation"],
        help="TorchAO FP8 quantization scheme.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length context window.",
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="State three advantages of FP8 precision for transformer inference.",
        help="Prompt to verify FP8 generation.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: 'unsloth' package is not installed. Run: pip install unsloth", file=sys.stderr)
        sys.exit(1)

    try:
        import torchao
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import (
            float8_weight_only,
            float8_dynamic_activation_float8_weight,
        )
    except ImportError:
        print(
            "Error: 'torchao' package is not installed. Run: pip install torchao",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[1/3] Loading model from '{args.model_name_or_path}' in 16-bit precision...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=False,  # Load in 16-bit to quantize directly to FP8
        token=hf_token,
    )

    print(f"[2/3] Applying TorchAO FP8 quantization ('{args.fp8_mode}')...")
    if args.fp8_mode == "float8_weight_only":
        quantize_(model, float8_weight_only())
    else:
        quantize_(model, float8_dynamic_activation_float8_weight())

    print(f"Verifying FP8 inference with prompt: '{args.test_prompt}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([args.test_prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Generated Output:\n{generated_text}\n")

    print(f"[3/3] Saving FP8 model checkpoint and tokenizer to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_fp8.pt"))
    print(f"TorchAO FP8 quantization completed and saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
