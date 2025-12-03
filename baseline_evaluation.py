"""
Baseline Evaluation Script for Qwen Models on Verus Tasks

This script evaluates multiple Qwen2.5-Coder model sizes (0.5B, 1.5B, 7B, 14B, 32B)
on Verus code generation tasks in zero-shot and few-shot settings.

Usage:
    python baseline_evaluation.py --model-size 1.5B --mode zero-shot
    python baseline_evaluation.py --model-size all --mode few-shot
    python baseline_evaluation.py --model-size 7B --mode both

Model sizes available:
    - 0.5B:  Qwen/Qwen2.5-Coder-0.5B (fast, low memory)
    - 1.5B:  Qwen/Qwen2.5-Coder-1.5B (balanced)
    - 3B:    Qwen/Qwen2.5-Coder-3B (good quality)
    - 7B:    Qwen/Qwen2.5-Coder-7B (high quality, requires 16GB+ VRAM)
    - 14B:   Qwen/Qwen2.5-Coder-14B (very high quality, requires 32GB+ VRAM)
    - 32B:   Qwen/Qwen2.5-Coder-32B (best quality, requires 80GB+ VRAM)
"""

import argparse
import json
import time
from typing import List, Dict, Any
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import test dataset and verification utilities
from test_dataset import build_test_dataset, build_few_shot_examples
from verus_verification import verify_with_verus, compute_verification_metrics


# Available Qwen2.5-Coder model sizes
QWEN_MODELS = {
    "0.5B": "Qwen/Qwen2.5-Coder-0.5B",
    "1.5B": "Qwen/Qwen2.5-Coder-1.5B",
    "3B": "Qwen/Qwen2.5-Coder-3B",
    "7B": "Qwen/Qwen2.5-Coder-7B",
    "14B": "Qwen/Qwen2.5-Coder-14B",
    "32B": "Qwen/Qwen2.5-Coder-32B",
}

# Note: build_test_dataset() and build_few_shot_examples() are imported from test_dataset.py


def load_model(model_name: str, device: str = "auto"):
    """
    Load a Qwen model and tokenizer.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('auto', 'cuda', 'cpu')

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print(f"This may take several minutes depending on model size...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )

    print(f"Model loaded successfully on {model.device}")
    return model, tokenizer


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    """
    Generate a completion for the given prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_p: Nucleus sampling parameter

    Returns:
        Generated text completion
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    num_input_tokens = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract only the newly generated tokens using token indices, then decode
    # This avoids issues with tokenizer whitespace normalization and encoding differences
    generated_tokens = outputs[0][num_input_tokens:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text.strip()


def evaluate_model(
    model_size: str,
    mode: str = "zero-shot",
    output_dir: str = "./baseline_results",
):
    """
    Evaluate a Qwen model on the Verus test set.

    Args:
        model_size: Size of the model to evaluate (e.g., "1.5B", "7B")
        mode: Evaluation mode ("zero-shot", "few-shot", or "both")
        output_dir: Directory to save results
    """
    if model_size not in QWEN_MODELS:
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(QWEN_MODELS.keys())}")

    model_name = QWEN_MODELS[model_size]
    test_examples = build_test_dataset()

    # Load model
    model, tokenizer = load_model(model_name)

    # Determine which modes to run
    modes_to_run = []
    if mode == "both":
        modes_to_run = ["zero-shot", "few-shot"]
    else:
        modes_to_run = [mode]

    all_results = {}

    for eval_mode in modes_to_run:
        print(f"\n{'='*60}")
        print(f"Running {eval_mode} evaluation on {model_size} model")
        print(f"{'='*60}\n")

        results = []
        few_shot_prefix = build_few_shot_examples() if eval_mode == "few-shot" else ""

        for i, example in enumerate(test_examples):
            print(f"Example {i+1}/{len(test_examples)}: {example['task_type']}")

            # Construct full prompt
            full_prompt = few_shot_prefix + example["prompt"]

            # Generate completion
            start_time = time.time()
            generated = generate_completion(model, tokenizer, full_prompt)
            gen_time = time.time() - start_time

            #Verify with Verus
            print(f"  Generated in {gen_time:.2f}s, verifying with Verus...", end=" ")
            verus_result = verify_with_verus(generated)

            # Store results
            result = {
                "example_id": i,
                "task_type": example["task_type"],
                "prompt": example["prompt"],
                "expected_output": example["expected_output"],
                "generated_output": generated,
                "generation_time_seconds": gen_time,
                "mode": eval_mode,
                "verus_verification": verus_result,
            }
            results.append(result)

            # Print status
            if verus_result["success"]:
                print("✓ VERIFIED")
            else:
                print(f"✗ {verus_result['error_type']}")
            print(f"  First 100 chars: {generated[:100]}...")
            print()

        all_results[eval_mode] = results

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for eval_mode, results in all_results.items():
        output_file = output_path / f"qwen_{model_size.replace('.', '_')}_{eval_mode}_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    # Print summary statistics with verification metrics
    print(f"\n{'='*60}")
    print(f"Summary for Qwen {model_size}")
    print(f"{'='*60}")
    for eval_mode, results in all_results.items():
        avg_time = sum(r["generation_time_seconds"] for r in results) / len(results)

        # Compute verification metrics
        verus_results = [r["verus_verification"] for r in results]
        metrics = compute_verification_metrics(verus_results)

        print(f"\n{eval_mode.upper()}:")
        print(f"  Total examples: {len(results)}")
        print(f"  Average generation time: {avg_time:.2f}s")
        print(f"  Total time: {sum(r['generation_time_seconds'] for r in results):.2f}s")
        print(f"\n  VERIFICATION RESULTS:")
        print(f"    Pass rate: {metrics['verification_pass_rate']:.1%} ({metrics['successful']}/{metrics['total_samples']})")
        print(f"    Syntax errors: {metrics['error_breakdown']['syntax_errors']}")
        print(f"    Mode errors: {metrics['error_breakdown']['mode_errors']}")
        print(f"    Spec errors: {metrics['error_breakdown']['precondition_errors'] + metrics['error_breakdown']['postcondition_errors']}")
        print(f"    Invariant errors: {metrics['error_breakdown']['invariant_errors']}")
        print(f"    VC failures: {metrics['error_breakdown']['vc_failures']}")
        print(f"    Timeouts: {metrics['error_breakdown']['timeouts']}")

        # Save metrics to separate file
        metrics_file = output_path / f"qwen_{model_size.replace('.', '_')}_{eval_mode}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen models on Verus code generation tasks"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="1.5B",
        help=f"Model size to evaluate. Options: {', '.join(QWEN_MODELS.keys())}, or 'all'",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "few-shot", "both"],
        help="Evaluation mode: zero-shot, few-shot, or both",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./baseline_results",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    # Handle "all" model sizes
    if args.model_size.lower() == "all":
        print("Evaluating all model sizes. This will take a long time!")
        for size in QWEN_MODELS.keys():
            try:
                evaluate_model(size, args.mode, args.output_dir)
            except Exception as e:
                print(f"Error evaluating {size}: {e}")
                print("Continuing with next model size...")
    else:
        evaluate_model(args.model_size, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
