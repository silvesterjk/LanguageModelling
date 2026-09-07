#!/usr/bin/env python3
"""
Quantize LLM to GPTQ (4-bit / 8-bit) from an Unsloth Checkpoint.

Method: AutoGPTQ (Generalized Post-Training Quantization)
Description:
    Quantizes an Unsloth-trained or fine-tuned model to 4-bit or 8-bit GPTQ format.
    GPTQ uses approximate second-order information (Hessian matrix) to compensate
    for quantization errors across weight layers sequentially.
    Highly optimized for GPU execution with ExLlama / ExLlamaV2 and vLLM kernels.

Pipeline:
    1. Merges LoRA adapters and dequantizes Unsloth model to 16-bit safetensors.
    2. Runs AutoGPTQ layer-by-layer second-order error minimization.
    3. Saves quantized model compatible with ExLlamaV2 and vLLM.

Target deployment: ExLlamaV2, vLLM (GPTQ backend), SGLang, Hugging Face Transformers.
"""

import argparse
import os
import shutil
import sys
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize an Unsloth model or LoRA checkpoint to GPTQ.",
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
        default="./quantized_gptq_4bit",
        help="Destination directory for the quantized GPTQ model.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bit precision (4 or 8).",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for quantization (128 recommended, -1 for per-channel).",
    )
    parser.add_argument(
        "--damp_percent",
        type=float,
        default=0.01,
        help="Damping factor percentage for Hessian conditioning.",
    )
    parser.add_argument(
        "--desc_act",
        action="store_true",
        help="Enable act-order (descending activation importance) heuristic.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face authentication token.",
    )
    return parser.parse_args()


def get_calibration_dataset(tokenizer, n_samples=128, seq_len=512):
    """Fetches and tokenizes sample calibration data."""
    try:
        from datasets import load_dataset
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(traindata["text"])
    except Exception:
        # Fallback to synthetic structured text if internet/dataset is unavailable
        text = "Large Language Models are neural networks trained on vast amounts of textual data. " * 500

    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    calibration_data = []
    for i in range(0, min(len(tokens) - seq_len, n_samples * seq_len), seq_len):
        calibration_data.append(tokens[i : i + seq_len].unsqueeze(0))
        if len(calibration_data) >= n_samples:
            break
    return calibration_data


def main():
    args = parse_args()
    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: 'unsloth' package is not installed. Run: pip install unsloth", file=sys.stderr)
        sys.exit(1)

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: 'auto_gptq' package is not installed. Run: pip install auto-gptq", file=sys.stderr)
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

        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[2/3] Configuring GPTQ ({args.bits}-bit, group_size={args.group_size}, damp={args.damp_percent})...")
        quantize_config = BaseQuantizeConfig(
            bits=args.bits,
            group_size=args.group_size,
            damp_percent=args.damp_percent,
            desc_act=args.desc_act,
        )

        gptq_tokenizer = AutoTokenizer.from_pretrained(temp_merged_dir, trust_remote_code=True)
        calibration_examples = get_calibration_dataset(gptq_tokenizer)

        gptq_model = AutoGPTQForCausalLM.from_pretrained(
            temp_merged_dir,
            quantize_config=quantize_config,
        )

        print(f"Running GPTQ calibration over {len(calibration_examples)} samples...")
        gptq_model.quantize(calibration_examples)

        print(f"[3/3] Saving final GPTQ model to '{args.output_dir}'...")
        os.makedirs(args.output_dir, exist_ok=True)
        gptq_model.save_quantized(args.output_dir, use_safetensors=True)
        gptq_tokenizer.save_pretrained(args.output_dir)
        print(f"AutoGPTQ model successfully generated at: {args.output_dir}")

    finally:
        if os.path.exists(temp_merged_dir):
            shutil.rmtree(temp_merged_dir)


if __name__ == "__main__":
    main()
