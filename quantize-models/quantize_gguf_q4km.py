#!/usr/bin/env python3
"""
Quantize LLM to GGUF Q4_K_M using Unsloth.

Method: GGUF Q4_K_M (4-bit Medium Quantization)
Description:
    Quantizes a base model or fine-tuned LoRA checkpoint directly into llama.cpp
    GGUF format using 4-bit medium quantization (Q4_K_M). Critical attention and
    feed-forward tensors are preserved with Q6_K precision, providing an optimal
    balance between reduced memory footprint (~4-5 bits per weight effective)
    and minimal perplexity degradation.

Target deployment: llama.cpp, Ollama, LM Studio, vLLM, Jan.
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a model to GGUF Q4_K_M format using Unsloth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging Face model ID (e.g. 'unsloth/llama-3-8b-Instruct') or local directory path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_gguf_q4_k_m",
        help="Destination directory where the .gguf file will be saved.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length context window.",
    )
    parser.add_argument(
        "--first_conversion",
        type=str,
        default="f16",
        choices=["f16", "bf16", "f32"],
        help="Intermediate precision format before GGUF quantization.",
    )
    parser.add_argument(
        "--maximum_memory_usage",
        type=float,
        default=0.85,
        help="Maximum fraction of RAM allowed during merging/conversion (0.0 - 1.0).",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the quantized GGUF model directly to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="Hugging Face Hub repository ID (e.g. 'username/model-GGUF') when pushing to hub.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token (reads from HF_TOKEN env variable if omitted).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hugging Face repository private if pushing to hub.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"[1/3] Loading model and tokenizer from '{args.model_name_or_path}' via Unsloth...")
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print(
            "Error: 'unsloth' package is not installed. "
            "Install it via: pip install unsloth",
            file=sys.stderr,
        )
        sys.exit(1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detected (Float16 / Bfloat16)
        load_in_4bit=True,  # Memory efficient loading
        token=hf_token,
    )

    print(f"[2/3] Quantizing model to GGUF format with quantization_method='q4_k_m'...")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("--hub_repo_id must be specified when --push_to_hub is set.")
        print(f"Uploading quantized GGUF (Q4_K_M) to Hugging Face Hub: {args.hub_repo_id}...")
        model.push_to_hub_gguf(
            hub_path=args.hub_repo_id,
            tokenizer=tokenizer,
            quantization_method="q4_k_m",
            first_conversion=args.first_conversion,
            maximum_memory_usage=args.maximum_memory_usage,
            token=hf_token,
            private=args.private,
        )
        print(f"Successfully pushed GGUF Q4_K_M to Hugging Face Hub: {args.hub_repo_id}")
    else:
        print(f"Saving quantized GGUF (Q4_K_M) locally to: {args.output_dir}...")
        model.save_pretrained_gguf(
            args.output_dir,
            tokenizer,
            quantization_method="q4_k_m",
            first_conversion=args.first_conversion,
            maximum_memory_usage=args.maximum_memory_usage,
        )
        print(f"[3/3] Quantization complete! GGUF Q4_K_M saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
