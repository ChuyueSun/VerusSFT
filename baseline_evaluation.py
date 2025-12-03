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
from datasets import Dataset


# Available Qwen2.5-Coder model sizes
QWEN_MODELS = {
    "0.5B": "Qwen/Qwen2.5-Coder-0.5B",
    "1.5B": "Qwen/Qwen2.5-Coder-1.5B",
    "3B": "Qwen/Qwen2.5-Coder-3B",
    "7B": "Qwen/Qwen2.5-Coder-7B",
    "14B": "Qwen/Qwen2.5-Coder-14B",
    "32B": "Qwen/Qwen2.5-Coder-32B",
}


def build_test_dataset() -> List[Dict[str, str]]:
    """
    Build test examples for baseline evaluation.
    These are the same examples from sft_example.py but structured for evaluation.

    Returns:
        List of test examples with 'prompt' and 'expected_output' fields.
    """
    test_examples = [
        {
            "prompt": "Add Verus specs to this function:\n```rust\nfn abs(x: i32) -> i32 {\n    if x < 0 { -x } else { x }\n}\n```",
            "expected_output": "```verus\nfn abs(x: i32) -> i32 {\n    requires true;\n    ensures |result| == |x| && (result == x || result == -x);\n    if x < 0 { -x } else { x }\n}\n```",
            "task_type": "spec_generation",
        },
        {
            "prompt": "Generate Rust code that satisfies: requires x > 0, ensures return == x + 1.",
            "expected_output": "fn add_one(x: i32) -> i32 {\n    requires x > 0;\n    ensures result == x + 1;\n    x + 1\n}",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Add Verus specs to this max function:\n```rust\nfn max(a: i32, b: i32) -> i32 {\n    if a > b { a } else { b }\n}\n```",
            "expected_output": "```verus\nfn max(a: i32, b: i32) -> i32 {\n    ensures result >= a && result >= b;\n    ensures result == a || result == b;\n    if a > b { a } else { b }\n}\n```",
            "task_type": "spec_generation",
        },
        {
            "prompt": "Write a Verus function that doubles a number:",
            "expected_output": "```verus\nfn double(x: i32) -> i32 {\n    requires x < i32::MAX / 2;\n    ensures result == 2 * x;\n    x * 2\n}\n```",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Add Verus specification for array bounds checking:\n```rust\nfn get_first(arr: &[i32]) -> Option<i32> {\n    if arr.len() > 0 {\n        Some(arr[0])\n    } else {\n        None\n    }\n}\n```",
            "expected_output": "```verus\nfn get_first(arr: &[i32]) -> Option<i32> {\n    requires true;\n    ensures arr.len() > 0 ==> result == Some(arr[0]);\n    ensures arr.len() == 0 ==> result == None;\n    if arr.len() > 0 {\n        Some(arr[0])\n    } else {\n        None\n    }\n}\n```",
            "task_type": "spec_generation",
        },
        {
            "prompt": "Create a Verus function for subtraction:",
            "expected_output": "```verus\nfn subtract(a: i32, b: i32) -> i32 {\n    requires a >= b;\n    ensures result == a - b;\n    ensures result >= 0;\n    a - b\n}\n```",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Add Verus specs for division:\n```rust\nfn divide(a: u32, b: u32) -> u32 {\n    a / b\n}\n```",
            "expected_output": "```verus\nfn divide(a: u32, b: u32) -> u32 {\n    requires b > 0;\n    ensures result == a / b;\n    ensures result <= a;\n    a / b\n}\n```",
            "task_type": "spec_generation",
        },
        {
            "prompt": "Write a Verus function that checks if a number is positive:",
            "expected_output": "```verus\nfn is_positive(x: i32) -> bool {\n    ensures result == (x > 0);\n    x > 0\n}\n```",
            "task_type": "code_synthesis",
        },
        {
            "prompt": "Add Verus specs for minimum function:\n```rust\nfn min(a: i32, b: i32) -> i32 {\n    if a < b { a } else { b }\n}\n```",
            "expected_output": "```verus\nfn min(a: i32, b: i32) -> i32 {\n    ensures result <= a && result <= b;\n    ensures result == a || result == b;\n    if a < b { a } else { b }\n}\n```",
            "task_type": "spec_generation",
        },
        {
            "prompt": "Create a Verus function for squaring:",
            "expected_output": "```verus\nfn square(x: i32) -> i32 {\n    requires x.abs() < 46341;  // sqrt(i32::MAX)\n    ensures result == x * x;\n    ensures result >= 0;\n    x * x\n}\n```",
            "task_type": "code_synthesis",
        },
    ]
    return test_examples


def build_few_shot_examples() -> str:
    """
    Build few-shot examples to prepend to prompts.
    These serve as in-context learning examples.

    Returns:
        Formatted string with 3 example prompt-completion pairs.
    """
    few_shot = """Here are some examples of Verus code generation:

Example 1:
Prompt: Add Verus specs to this function:
```rust
fn clamp(x: i32, min: i32, max: i32) -> i32 {
    if x < min { min } else if x > max { max } else { x }
}
```

Answer:
```verus
fn clamp(x: i32, min: i32, max: i32) -> i32 {
    requires min <= max;
    ensures result >= min && result <= max;
    ensures (x >= min && x <= max) ==> result == x;
    if x < min { min } else if x > max { max } else { x }
}
```

Example 2:
Prompt: Write a Verus function that returns the sum of two positive numbers:

Answer:
```verus
fn sum_positive(a: i32, b: i32) -> i32 {
    requires a > 0 && b > 0;
    requires a < i32::MAX - b;  // Prevent overflow
    ensures result == a + b;
    ensures result > a && result > b;
    a + b
}
```

Example 3:
Prompt: Add Verus specs for safe array access:
```rust
fn get_element(arr: &[i32], idx: usize) -> Option<i32> {
    if idx < arr.len() {
        Some(arr[idx])
    } else {
        None
    }
}
```

Answer:
```verus
fn get_element(arr: &[i32], idx: usize) -> Option<i32> {
    ensures idx < arr.len() ==> result == Some(arr[idx]);
    ensures idx >= arr.len() ==> result == None;
    if idx < arr.len() {
        Some(arr[idx])
    } else {
        None
    }
}
```

Now solve this task:

"""
    return few_shot


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

            # Store results
            result = {
                "example_id": i,
                "task_type": example["task_type"],
                "prompt": example["prompt"],
                "expected_output": example["expected_output"],
                "generated_output": generated,
                "generation_time_seconds": gen_time,
                "mode": eval_mode,
            }
            results.append(result)

            print(f"  Generated in {gen_time:.2f}s")
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

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Summary for Qwen {model_size}")
    print(f"{'='*60}")
    for eval_mode, results in all_results.items():
        avg_time = sum(r["generation_time_seconds"] for r in results) / len(results)
        print(f"\n{eval_mode.upper()}:")
        print(f"  Total examples: {len(results)}")
        print(f"  Average generation time: {avg_time:.2f}s")
        print(f"  Total time: {sum(r['generation_time_seconds'] for r in results):.2f}s")


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
