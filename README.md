# VerusFT-RL - Fine-Tuning and RL for Verus Code Generation

A fine-tuning and reinforcement learning (RL) project for training language models to generate Rust code with Verus specifications.

## Overview

This project demonstrates how to fine-tune and apply reinforcement learning to a language model to generate Rust functions annotated with Verus specifications. It uses:

- **Hugging Face Transformers** for model loading and inference
- **TRL (Transformer Reinforcement Learning)** for supervised fine-tuning and reinforcement learning
- **PEFT (Parameter-Efficient Fine-Tuning)** with LoRA for efficient training
- **GPT-2** as the base model (can be replaced with code-specific models)

## Features

- ✅ Parameter-efficient fine-tuning with LoRA
- ✅ 10 diverse Verus training examples
- ✅ Configurable training hyperparameters
- ✅ Inference script for testing trained models
- ✅ Small adapter weights (~6.2MB) instead of full model

## Training Results

- **Loss reduction**: 3.40 → 2.42 (28.7% improvement)
- **Token accuracy**: 45% → 55%
- **Training time**: ~14 seconds for 8 epochs
- **Adapter size**: 6.2MB

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd VerusFT-RL

# Install dependencies
pip install transformers trl datasets peft accelerate torch
```

## Usage

### 1. Training

Run the training script to fine-tune the model on Verus examples:

```bash
python sft_example.py
```

The trained model will be saved to `./sft_output/`.

### 2. Inference

Test the trained model with new prompts:

```bash
python test_inference.py
```

## Configuration

### Training Parameters

Edit `sft_example.py` to adjust:

- `num_train_epochs`: Number of training epochs (default: 10)
- `per_device_train_batch_size`: Batch size per device (default: 2)
- `learning_rate`: Learning rate (default: 3e-4)
- `max_seq_length`: Maximum sequence length (default: 1024)

### LoRA Configuration

- `r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA alpha parameter (default: 32)
- `target_modules`: Modules to apply LoRA (default: `["c_attn", "c_proj"]`)

## Dataset

The training dataset includes 10 examples covering:

- Absolute value functions
- Min/max operations
- Arithmetic operations (add, subtract, multiply, divide)
- Array bounds checking
- Boolean predicates
- Squaring with overflow protection

### Adding More Examples

Edit the `build_dataset()` function in `sft_example.py` to add your own Verus examples:

```python
{
    "text": "Your prompt here\nYour Verus code here"
}
```

Possible sources of additional Verus examples include:

- [VOSTD dataset](https://github.com/asterinas/vostd) for automatically generated specifications
- [Verismo](https://github.com/microsoft/verismo) for verified systems examples
- [Verified Memory Allocator](https://github.com/verus-lang/verified-memory-allocator) for memory-safety focused code
- [Verified Storage](https://github.com/microsoft/verified-storage) for storage system verification examples
- [Vericoding](https://github.com/Beneficial-AI-Foundation/vericoding) for related work and potential benchmarking ideas

## File Structure

```
VerusFT-RL/
├── sft_example.py          # Main training script
├── test_inference.py       # Inference/testing script
├── README.md               # This file
├── .gitignore             # Git ignore rules
└── sft_output/            # Trained model output (not tracked)
    ├── adapter_model.safetensors
    ├── adapter_config.json
    └── tokenizer files
```

## Future Improvements

1. **Expand Dataset**: Add 50-100+ diverse Verus examples
2. **Better Base Model**: Use code-specific models like:
   - `Qwen/Qwen2.5-Coder-1.5B`
   - `bigcode/starcoder2-3b`
   - `deepseek-ai/deepseek-coder-1.3b-base`
3. **Evaluation**: Add metrics for Verus specification correctness
4. **Fine-tune Generation**: Adjust temperature, top-p, and beam search parameters

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- trl
- datasets
- peft
- accelerate

## License

MIT License

## Acknowledgments

- [Verus](https://github.com/verus-lang/verus) - The Verus verification system
- [Hugging Face](https://huggingface.co/) - Transformers and model hub
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning library

