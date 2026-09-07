#!/usr/bin/env python3
"""
Quantize LLM to AWQ (Activation-aware Weight Quantization, 4-bit) from an Unsloth Checkpoint.

Method: AutoAWQ 4-bit (W4A16 GEMM / GEMV)
Description:
    Quantizes a model fine-tuned or managed with Unsloth into 4-bit AWQ format.
    AWQ protects the top 1% salient weights by observing activation magnitudes
    during calibration, eliminating quantization loss without requiring backpropagation.
    This produces blistering fast token throughput with vLLM, SGLang, and Hugging Face.

Pipeline:
    1. Dequantizes & merges Unsloth LoRA checkpoint into standard 16-bit safetensors.
    2. Calibrates weight sensitivities using AutoAWQ on representative text.
    3. Exports optimized AWQ 4-bit checkpoint.

Target deployment: vLLM (AWQ backend), SGLang, Hugging Face TGI, Transformers.
"""

import argparse
import os
import shutil
import sys
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize an Unsloth model or LoRA checkpoint to 4-bit AWQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging Face model ID or path to Unsloth fine-tuned model/LoRA adapter.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_awq_4bit",
        help="Destination directory for the quantized AWQ model.",
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=4,
        choices=[4],
        help="Bit precision for quantized weights.",
    )
    parser.add_argument(
        "--q_group_size",
        type=int,
        default=128,
        help="Quantization group size (typically 128 or 64).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="GEMM",
        choices=["GEMM", "GEMV", "marlin"],
        help="Kernel version backend for AWQ.",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="pileval",
        help="Calibration dataset for AutoAWQ (e.g. 'pileval', 'wikitext').",
    )
    parser.add_argument(
        "--max_calib_samples",
        type=int,
        default=128,
        help="Number of calibration samples to evaluate.",
    )
    parser.add_argument(
        "--max_calib_seq_len",
        type=int,
        default=512,
        help="Sequence length per calibration sample.",
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

    # Step 1: Check dependencies
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: 'unsloth' package is not installed. Run: pip install unsloth", file=sys.stderr)
        sys.exit(1)

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: 'autoawq' package is not installed. Run: pip install autoawq", file=sys.stderr)
        sys.exit(1)

    temp_merged_dir = tempfile.mkdtemp(prefix="unsloth_merged_16bit_")
    try:
        print(f"[1/3] Merging and dequantizing Unsloth model from '{args.model_name_or_path}' to 16-bit...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=hf_token,
        )
        model.save_pretrained_merged(temp_merged_dir, tokenizer, save_method="merged_16bit")
        print(f"Intermediate 16-bit weights exported to temporary staging area.")

        # Clean up Unsloth model from GPU memory
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[2/3] Performing AutoAWQ 4-bit calibration on dataset '{args.calib_dataset}'...")
        awq_model = AutoAWQForCausalLM.from_pretrained(temp_merged_dir, **{"low_cpu_mem_usage": True})
        awq_tokenizer = AutoTokenizer.from_pretrained(temp_merged_dir, trust_remote_code=True)

        quant_config = {
            "zero_point": True,
            "q_group_size": args.q_group_size,
            "w_bit": args.w_bit,
            "version": args.version,
        }

        awq_model.quantize(
            awq_tokenizer,
            quant_config=quant_config,
            calib_data=args.calib_dataset,
            max_calib_samples=args.max_calib_samples,
            max_calib_seq_len=args.max_calib_seq_len,
        )

        print(f"[3/3] Saving final AWQ 4-bit model to '{args.output_dir}'...")
        os.makedirs(args.output_dir, exist_ok=True)
        awq_model.save_quantized(args.output_dir)
        awq_tokenizer.save_pretrained(args.output_dir)
        print(f"AutoAWQ 4-bit model successfully generated at: {args.output_dir}")

    finally:
        if os.path.exists(temp_merged_dir):
            shutil.rmtree(temp_merged_dir)


if __name__ == "__main__":
    main()
