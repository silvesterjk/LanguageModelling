#!/usr/bin/env python3
"""
Quantize LLM to GGUF using Importance Matrix (imatrix) Dynamic Quantization.

Method: GGUF Importance Matrix (imatrix) & IQ Quantization (e.g. IQ4_NL, IQ3_XXS, IQ3_M)
Description:
    Applies data-driven dynamic GGUF quantization using an importance matrix (imatrix).
    By evaluating parameter sensitivity on a calibration text dataset, the quantization
    algorithm selectively preserves critical activation channels and sensitive weights,
    enabling extreme compression (1.5 - 4 bits per weight) with dramatically lower
    perplexity loss than static uniform quantizers.

Target deployment: llama.cpp, Ollama, resource-constrained mobile and edge devices.
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize a model to GGUF using an importance matrix (imatrix) with Unsloth.",
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
        default="./quantized_gguf_imatrix",
        help="Destination directory where the .gguf file will be saved.",
    )
    parser.add_argument(
        "--quantization_method",
        type=str,
        default="iq4_nl",
        choices=[
            "iq4_nl",
            "iq4_xs",
            "iq3_m",
            "iq3_s",
            "iq3_xxs",
            "iq2_m",
            "iq2_s",
            "iq2_xxs",
            "iq1_s",
            "iq1_m",
            "q4_k_m",
        ],
        help="Target IQ/GGUF quantization method.",
    )
    parser.add_argument(
        "--imatrix_file",
        type=str,
        default=None,
        help="Path to precomputed llama.cpp .imatrix calibration file. "
        "Required for IQ quants to prevent quality degradation.",
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

    if args.imatrix_file and not os.path.exists(args.imatrix_file):
        raise FileNotFoundError(f"Specified imatrix file does not exist: {args.imatrix_file}")

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
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )

    print(f"[2/3] Quantizing model with method='{args.quantization_method}' (imatrix: {args.imatrix_file})...")
    os.makedirs(args.output_dir, exist_ok=True)

    export_kwargs = {
        "quantization_method": args.quantization_method,
        "first_conversion": args.first_conversion,
        "maximum_memory_usage": args.maximum_memory_usage,
    }
    if args.imatrix_file:
        export_kwargs["imatrix_file"] = args.imatrix_file

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("--hub_repo_id must be specified when --push_to_hub is set.")
        print(f"Uploading quantized GGUF ({args.quantization_method}) to Hugging Face Hub: {args.hub_repo_id}...")
        model.push_to_hub_gguf(
            hub_path=args.hub_repo_id,
            tokenizer=tokenizer,
            token=hf_token,
            private=args.private,
            **export_kwargs,
        )
        print(f"Successfully pushed GGUF to Hugging Face Hub: {args.hub_repo_id}")
    else:
        print(f"Saving quantized GGUF ({args.quantization_method}) locally to: {args.output_dir}...")
        model.save_pretrained_gguf(
            args.output_dir,
            tokenizer,
            **export_kwargs,
        )
        print(f"[3/3] Quantization complete! Saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
