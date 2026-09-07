#!/usr/bin/env python3
"""
Work with Unsloth Dynamic 4-bit Quantization.

Method: Unsloth Dynamic 4-bit Quantization (Selective Mixed-Precision NF4)
Description:
    Unsloth Dynamic 4-bit quantization overcomes the degradation of uniform 4-bit
    quantization by selectively skipping quantization on outlier/sensitive layers
    (such as early attention layers, vision encoders in multimodal models, and
    classification heads). Critical layers are preserved in 16-bit (BF16/FP16), while
    the remainder run in 4-bit NF4 with Triton kernels, achieving close-to-16-bit
    benchmark accuracy with under 10% extra memory overhead over standard 4-bit.

Target deployment: High-accuracy fine-tuning and low-VRAM inference on consumer GPUs.
"""

import argparse
import os
import sys
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load, benchmark, and export Unsloth Dynamic 4-bit models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        help="Dynamic 4-bit model repository or local path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./quantized_dynamic_4bit",
        help="Destination directory for exporting the model configuration.",
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
        default="Explain why dynamic mixed-precision quantization outperforms naive uniform quantization.",
        help="Prompt to test generation speed and accuracy.",
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

    print(f"[1/3] Loading Dynamic 4-bit model from '{args.model_name_or_path}'...")
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Error: 'unsloth' package is not installed. Run: pip install unsloth", file=sys.stderr)
        sys.exit(1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )

    # Inspect model modules and identify preserved high-precision layers
    print("\n--- Model Architecture & Layer Precision Inspection ---")
    fp16_layers = 0
    int4_layers = 0
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "Linear4bit" in module_type:
            int4_layers += 1
        elif "Linear" in module_type:
            fp16_layers += 1

    print(f"Total 4-bit Linear layers: {int4_layers}")
    print(f"Total 16-bit preserved Linear layers: {fp16_layers}")
    print("-------------------------------------------------------\n")

    FastLanguageModel.for_inference(model)

    print(f"[2/3] Generating output for prompt: '{args.test_prompt}'...")
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

    print(f"[3/3] Saving model and tokenizer to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Dynamic 4-bit model files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
