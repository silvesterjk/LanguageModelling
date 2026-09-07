#!/usr/bin/env python3
"""
Quantize LLM to BitsAndBytes 4-bit (NF4 / FP4) using Unsloth.

Method: BitsAndBytes 4-bit (NormalFloat4 / FP4 with Double Quantization)
Description:
    Quantizes a model to 4-bit precision using BitsAndBytes NF4 (NormalFloat4)
    accelerated with Unsloth's custom Triton kernels. Supports merging LoRA
    adapter checkpoints directly into 4-bit quantized base weights
    (`save_method='merged_4bit'` or `'merged_4bit_forced'`) or uploading
    the merged 4-bit model directly to Hugging Face Hub.

Target deployment: Hugging Face Transformers, BitsAndBytes inference, Unsloth fast inference.
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a model to BitsAndBytes 4-bit format using Unsloth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging Face model ID (e.g. 'unsloth/llama-3-8b') or local fine-tuned LoRA path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_bnb_4bit",
        help="Destination directory for the merged 4-bit safetensors checkpoint.",
    )
    parser.add_argument(
        "--save_method",
        type=str,
        default="merged_4bit",
        choices=["merged_4bit", "merged_4bit_forced"],
        help="'merged_4bit' merges LoRA into 4-bit base weights. "
        "'merged_4bit_forced' forces 4-bit quantization on non-bnb modules.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length context window.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the merged 4-bit model directly to Hugging Face Hub.",
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

    print(f"[1/3] Loading model from '{args.model_name_or_path}' with Unsloth 4-bit NF4 engine...")
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
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )

    print(f"[2/3] Exporting merged 4-bit model with save_method='{args.save_method}'...")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("--hub_repo_id must be specified when --push_to_hub is set.")
        print(f"Uploading merged 4-bit model to Hugging Face Hub: {args.hub_repo_id}...")
        model.push_to_hub_merged(
            hub_path=args.hub_repo_id,
            tokenizer=tokenizer,
            save_method=args.save_method,
            token=hf_token,
            private=args.private,
        )
        print(f"Successfully pushed 4-bit model to Hugging Face Hub: {args.hub_repo_id}")
    else:
        print(f"Saving merged 4-bit checkpoint locally to: {args.output_dir}...")
        model.save_pretrained_merged(
            args.output_dir,
            tokenizer,
            save_method=args.save_method,
        )
        print(f"[3/3] 4-bit model export complete! Saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
