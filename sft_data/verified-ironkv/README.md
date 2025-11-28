# VerusFT-RL Data Preparation

This directory contains tools for preparing training data from Verus codebases.

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Source Repos   │ ──▶ │  Minimizer   │ ──▶ │ Unit Extractor  │ ──▶ │ JSONL Output │
│  (Verus code)   │     │  (optional)  │     │  (Task A/B/C)   │     │  (training)  │
└─────────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
```

## Directory Structure

```
data_preparation/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.yaml                  # Configuration for data sources
│
├── collectors/                  # Source code collectors
│   ├── __init__.py
│   ├── github_collector.py      # Clone/fetch Verus repos
│   └── local_collector.py       # Process local Verus files
│
├── extractors/                  # Unit extraction logic
│   ├── __init__.py
│   ├── verus_parser.py          # Parse Verus syntax patterns
│   ├── function_extractor.py    # Extract function-level units
│   ├── spec_extractor.py        # Extract specifications
│   └── proof_extractor.py       # Extract proof blocks
│
├── tasks/                       # Task-specific generators
│   ├── __init__.py
│   ├── task_a_code_to_spec.py   # Code → Specifications
│   ├── task_b_spec_to_code.py   # Specifications → Code
│   └── task_c_error_repair.py   # Error-Guided Repair
│
├── filters/                     # Quality filtering
│   ├── __init__.py
│   ├── deduplication.py         # Near-duplicate removal
│   ├── quality_filter.py        # Quality scoring
│   └── complexity_filter.py     # Complexity-based filtering
│
├── output/                      # Output formatters
│   ├── __init__.py
│   └── jsonl_writer.py          # JSONL serialization
│
└── scripts/                     # Entry point scripts
    ├── collect_sources.py       # Step 1: Collect source code
    ├── extract_units.py         # Step 2: Extract units
    ├── generate_tasks.py        # Step 3: Generate task data
    └── run_pipeline.py          # Full pipeline
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Option 1: Run with command-line source path (simplest)
python scripts/run_pipeline.py --source ../ironsht/src --output data/

# Option 2: Run with config file
cp config.example.yaml config.yaml
# Edit config.yaml to add your source paths under sources.local
python scripts/run_pipeline.py --config config.yaml --output data/

# Option 3: Both config file AND additional source
python scripts/run_pipeline.py --config config.yaml --source ./extra_source --output data/

# Generate only specific tasks
python scripts/run_pipeline.py --source ../ironsht/src --output data/ --tasks a b

# Use detailed prompts
python scripts/run_pipeline.py --source ../ironsht/src --output data/ --prompt-style detailed

# Convert to popular LLM training formats
python output/format_converters.py data/ data/openai_format/ --format openai
python output/format_converters.py data/ data/sharegpt_format/ --format sharegpt
```

## Data Quality Verification

See **[VERIFICATION_EXAMPLES.md](./VERIFICATION_EXAMPLES.md)** for concrete evidence that the pipeline correctly extracts Verus code. This document shows side-by-side comparisons of original source code and processed JSON output.

Current quality metrics on IronSHT:
- Task A (Code → Spec): **100%** clean
- Task B (Spec → Code): **100%** clean  
- Task C (Error Repair): **100%** clean

## Task Definitions

### Task A: Code → Specifications
- **Input**: Rust/Verus function body (specs removed or minimal)
- **Output**: `requires`, `ensures`, `invariant`, `decreases`, Views

### Task B: Specifications → Verified Code  
- **Input**: Full specification + function signature
- **Output**: Executable + ghost + proof code that verifies

### Task C: Error-Guided Proof/Invariant Repair
- **Input**: Code + spec + Verus error message
- **Output**: Patched code that fixes the verification failure

## Data Sources

Priority sources for Verus code:
1. [Verus main repo](https://github.com/verus-lang/verus) - examples, tests, vstd
2. [verified-ironkv](.) - This repo (IronSHT implementation)
3. [VeriStruct examples](https://github.com/ChuyueSun/VeriStruct)
4. Other projects from [Verus publications](https://verus-lang.github.io/verus/publications-and-projects/)

## Output Format

Each training example is a JSONL entry:

```json
{
  "id": "ironkv_delegation_map_v_123",
  "task": "spec_from_code",
  "input_text": "fn get(&self, k: &K) -> (id: ID) { ... }",
  "target_text": "requires self.valid(),\nensures id@ == self@[*k],",
  "metadata": {
    "repo": "verified-ironkv",
    "file": "delegation_map_v.rs",
    "function": "get",
    "minimized": false,
    "verus_version": "0.2024.x",
    "has_loop_invariant": false,
    "has_proof_block": false,
    "spec_complexity": "medium"
  }
}
```

