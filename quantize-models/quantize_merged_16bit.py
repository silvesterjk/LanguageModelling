#!/usr/bin/env python3
"""
Dequantize and Merge LoRA Adapters to 16-bit (FP16/BF16) using Unsloth.

Method: Merged 16-bit Dequantization (FP16 / BF16 Safetensors Export)
Description:
    Dequantizes 4-bit quantized base weights back into full 16-bit precision
    (float16 or bfloat16) while cleanly fusing LoRA adapter weights.
    This creates an unquantized standalone Hugging Face model checkpoint with
    no BitsAndBytes dependencies, making it the ideal base for:
    1. High-throughput serving engines (vLLM, SGLang, TGI, TensorRT-LLM).
    2. Subsequent quantization workflows (AWQ, GPTQ, TorchAO, FP8).

Target deployment: vLLM, SGLang, Hugging Face Transformers, AWQ/GPTQ pre-processing.
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dequantize and merge a 4-bit model/adapter to 16-bit using Unsloth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to fine-tuned LoRA checkpoint directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_16bit_model",
        help="Destination directory where merged 16-bit safetensors will be saved.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length context window.",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Maximum checkpoint file shard size (e.g. '5GB', '10GB').",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the merged 16-bit model directly to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="Hugging Face Hub repository ID when pushing to hub.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token.",
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

    print(f"[1/3] Loading model from '{args.model_name_or_path}' via Unsloth...")
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
        dtype=None,  # Auto-select Float16 or Bfloat16 based on hardware
        load_in_4bit=True,
        token=hf_token,
    )

    print(f"[2/3] Merging LoRA adapters and dequantizing base weights to 16-bit...")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("--hub_repo_id must be specified when --push_to_hub is set.")
        print(f"Uploading merged 16-bit model to Hugging Face Hub: {args.hub_repo_id}...")
        model.push_to_hub_merged(
            hub_path=args.hub_repo_id,
            tokenizer=tokenizer,
            save_method="merged_16bit",
            max_shard_size=args.max_shard_size,
            token=hf_token,
            private=args.private,
        )
        print(f"Successfully pushed 16-bit model to Hugging Face Hub: {args.hub_repo_id}")
    else:
        print(f"Saving merged 16-bit safetensors locally to: {args.output_dir}...")
        model.save_pretrained_merged(
            args.output_dir,
            tokenizer,
            save_method="merged_16bit",
            max_shard_size=args.max_shard_size,
        )
        print(f"[3/3] Merged 16-bit model successfully exported to: {args.output_dir}")


if __name__ == "__main__":
    main()
