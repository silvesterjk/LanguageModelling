#!/usr/bin/env python3
"""
Quantize LLM to BitsAndBytes 8-bit (LLM.int8()) using Unsloth.

Method: BitsAndBytes 8-bit Quantization (LLM.int8() Outlier Vector Decomposition)
Description:
    Quantizes a model into 8-bit precision using BitsAndBytes. Outlier activation
    values that exceed a threshold are computed in full 16-bit precision, while 99.9%
    of weights and activations are dynamically quantized to 8-bit integers.
    This provides near zero perplexity degradation while reducing VRAM by roughly 50%
    compared to FP16/BF16.

Target deployment: Hugging Face Transformers, BitsAndBytes 8-bit inference.
"""

import argparse
import os
import sys
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and quantize a model to BitsAndBytes 8-bit format using Unsloth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging Face model ID (e.g. 'unsloth/Qwen2.5-7B') or local path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_bnb_8bit",
        help="Destination directory to save model configuration and tokenizer.",
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
        default="Explain the benefits of 8-bit quantization in one concise paragraph.",
        help="Prompt to test 8-bit inference generation.",
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

    print(f"[1/3] Loading model in BitsAndBytes 8-bit precision from '{args.model_name_or_path}'...")
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
        dtype=torch.float16,
        load_in_8bit=True,
        load_in_4bit=False,
        token=hf_token,
    )

    # Enable native inference acceleration
    FastLanguageModel.for_inference(model)

    print(f"[2/3] Verifying 8-bit inference with prompt: '{args.test_prompt}'...")
    inputs = tokenizer([args.test_prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Generated Output:\n{generated_text}\n")

    print(f"[3/3] Saving tokenizer and 8-bit model config to: {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    print(f"BitsAndBytes 8-bit checkpoint successfully saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
