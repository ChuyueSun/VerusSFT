# VerusFT-RL: Fine-Tuning and RL for Verification-Oriented Rust/Verus Code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

VerusFT-RL is a research repo for exploring **supervised fine-tuning (SFT)** and **reinforcement learning (RL)** of language models on **verification-oriented Rust/Verus code**. General-purpose code LLMs often struggle with Verus-specific concepts like `exec` / `ghost` / `proof` modes, `requires` / `ensures` specifications, View functions, typestate-like abstractions, loop invariants and `decreases` clauses, proof blocks, and Verus error traces. The goal is to make models genuinely Verus-aware and to understand when structured representations (like ASTs) add value beyond plain text.

## Table of Contents
- [Quick Start](#quick-start)
- [Project Status](#project-status)
- [Project Goals](#project-goals)
- [Motivation](#motivation)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [File Structure](#file-structure)
- [Methodology](#methodology)
- [Reinforcement Learning Extensions](#reinforcement-learning-extensions-for-verification-agents)
- [Student Subprojects](#student-subprojects)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Related Work](#related-work)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Quick Start

```bash
# Install dependencies
pip install transformers trl datasets peft accelerate torch

# Train the model
python sft_example.py

# Test inference
python test_inference.py
```

---

## Project Status

**Current Implementation Status** (as of December 2025):

- ✅ **Parameter-efficient fine-tuning with LoRA**
- ✅ **10 diverse Verus training examples** (seed set, expanding soon)
- ✅ **Configurable training hyperparameters**
- ✅ **Inference script for testing trained models**
- ✅ **Small adapter weights** (~6.2MB) instead of full model
- 🚧 **Minimizer integration** (planned)
- 📋 **Multi-task dataset builder** (design phase)
- 📋 **RL components** (design phase)

### Training Results (Prototype - Dec 2025)

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Loss | 3.40 | 2.42 | 28.7% ↓ |
| Token Accuracy | 45% | 55% | 10% ↑ |
| Training Time | - | ~14s | 8 epochs |
| Adapter Size | - | 6.2MB | - |

---

## Project Goals

### 🎯 Primary Goals

| Goal | Description | Status |
|------|-------------|--------|
| **Dataset** | Build a minimized, high-quality text-only dataset of Verus code, specs, proofs, and error traces from open-source projects | 📋 Planned |
| **Multi-Task SFT** | Train models on three core tasks (see below) | 🚧 In Progress |
| **Verus-Native Evaluation** | Evaluate using verification pass rate, not just syntax similarity | 📋 Planned |

### Three Core Tasks

1. **Task A — Code → Specifications** ✅ *Prototype Ready*
   - Input: Rust+Verus function body (no specs)
   - Output: `requires`/`ensures`, Views, invariants, `decreases` clauses

2. **Task B — Specifications → Verified Code** 🚧 *In Progress*
   - Input: Full specification + function signature
   - Output: Executable + ghost + proof code that verifies

3. **Task C — Error-Guided Repair** 📋 *Planned*
   - Input: Code + spec + Verus error message
   - Output: Patched invariant, spec, or proof block

### 🔬 Secondary Goal (Research Question)

**When do structured representations help?**
- After strong text-only baselines, introduce AST/structured encodings
- Measure incremental benefit via ablation studies
- Answer: *For what types of verification tasks do ASTs provide value beyond plain text?*

---

## Motivation

### The Problem

Formal verification in Verus embeds specifications, invariants, and proofs directly in Rust code. While systems like VeriStruct show that LLMs can assist with specification and proof tasks, **current general-purpose code models struggle** with:

| Challenge | Example Issue |
|-----------|---------------|
| **Verus Modes** | Confusing `exec`, `ghost`, and `proof` contexts |
| **Specifications** | Omitting or weakening `requires`/`ensures` clauses |
| **Proof Generation** | Generating syntactically valid but logically invalid proofs |
| **Error Interpretation** | Failing to understand Verus error messages and verification failures |
| **Ghost State** | Mishandling View functions and abstract specifications |

### Our Hypothesis

**Supervised fine-tuning on a curated corpus of minimized, self-contained Verus examples will significantly improve model performance**—even *without* structured encodings.

**Key insights:**
- 📦 **Minimized examples** are self-contained and focused
- 🎯 **Multi-task training** covers the full verification workflow
- 🔬 **Text-first approach** establishes strong baselines before exploring structure
- 📊 **Verus-native evaluation** measures actual verification success, not just syntax

**Research question:** Once we have strong text baselines, we can rigorously ask: *when do ASTs or other structural views actually help?*

---

## Related Work

VerusFT-RL builds on a growing body of literature exploring language-model-assisted verification:

- **[VeriStruct](https://github.com/ChuyueSun/VeriStruct)**: Focuses on prompt engineering and retrieval but stops short of supervised fine-tuning on minimized Verus corpora.
- **[arXiv:2505.20302](https://arxiv.org/pdf/2505.20302)**: Examines adjacent verification-aware fine-tuning strategies.
- **Key gaps this project addresses**:
  - Reproducible, minimized Verus datasets
  - Verus-native evaluation harnesses
  - Head-to-head comparisons between text-only and structure-augmented representations
  - Multi-task SFT with RL-based refinement

---

## Methodology

> **Note:** This section describes the complete research vision. The current prototype implements basic SFT training. The dataset pipeline, minimizer integration, and multi-task training are planned for future development.

The project is split into two main phases.

### Phase 1 — Text-Only Dataset + SFT (Core)

We start from existing open-source Verus code, including:

- the main Verus repo (examples, tests, verified libraries),
- Verus-based projects listed on the [Verus publications/projects page](https://verus-lang.github.io/verus/publications-and-projects/), and
- internal benchmarks and artifacts (e.g., [VeriStruct](https://github.com/ChuyueSun/VeriStruct) data-structure modules).

Large modules and full repos are too big for individual SFT samples. Instead, we use the existing **Verus minimizer** to turn big programs into small, self-contained examples.

**Minimizer tool:**
- Location: [`source/tools/minimizers/`](https://github.com/ChuyueSun/verus/tree/main/source/tools/minimizers) in the Verus repo
- Purpose: given a target file or property, automatically shrink the program while preserving a chosen verification behavior (e.g., still verifies, or still fails with a particular error).

#### Dataset Pipeline

1. **Seed Collection**
   - Select diverse Verus files from examples, tests, and verified modules.

2. **Minimization**
   - Run the minimizer to get tiny verifying (and optionally failing) examples.
   - Each minimized file is a compact unit with exactly the code/specs/proofs needed for that behavior.

3. **Unit Extraction**
   From each minimized file, extract logical units:
   - Function-level units for **Task A (code → spec)** and **Task B (spec → code)**.
   - Lemma/proof-level units for **Task C (repair)**.
   - Module-level units only when necessary (e.g., shared View functions).

4. **JSONL Serialization (Text-Only)**
   Each dataset entry is stored as a text-only example, e.g.:

   ```json
   {
     "id": "...",
     "task": "spec_from_code",
     "input_text": "...",
     "target_text": "...",
     "metadata": {
       "repo": "...",
       "minimized": true,
       "verus_version": "...",
       "code_lines": 37
     }
   }
   ```

5. **Quality Filtering & Deduplication**
   - Remove trivial/uninteresting samples.
   - Deduplicate near-duplicates across repositories.

**Two complementary products from the minimizer.**
- **SFT dataset (Q/A style, no reasoning):** every minimized snippet is converted into question/answer pairs for Tasks A/B/C. The "question" is the task-specific prompt (e.g., "add the missing `requires`"), and the "answer" is the ground-truth spec or code. We deliberately keep the answers *reasoning-free* so that SFT teaches the model crisp completions without hallucinated thought chains.
- **RL trajectory dataset (reasoning-rich):** for the exact same minimized examples, we also log multi-step interaction traces (tool calls, intermediate hypotheses, Verus logs). These richer transcripts fuel offline RL algorithms by giving them access to explicit reasoning steps, not just final Q/A endpoints.

This yields datasets that are small, focused, and tailored separately for high-quality SFT and reasoning-aware RL.

### Phase 1: Text-Only SFT Tasks

We focus on three core tasks, all in plain Verus text (no AST yet):

#### Task A — Code → Specifications

Input: Rust+Verus function body (no specs or minimal specs).
Output:
- `requires` / `ensures`
- relevant Views / ghost arguments
- `invariant` / `decreases` clauses if loops are present

#### Task B — Specifications → Verified Code

Input: full specification (pre/postconditions, Views) and function signature.
Output: executable + ghost + proof code that verifies under Verus.

#### Task C — Proof / Invariant Repair

Input: code + spec + Verus error message (or failing obligation info).
Output: patched invariant, spec, or proof block that fixes the failure.

Training is done in an instruction-style format compatible with downstream usage (e.g., [VeriStruct](https://github.com/ChuyueSun/VeriStruct) pipeline prompts).

### Phase 1: Evaluation

The main metric is **verification pass rate** on held-out modules. Secondary metrics include:

- syntax/parse correctness,
- spec completeness and tightness,
- size of repairs (how much the model changes), and
- breakdown by error type (mode errors, missing ensures, wrong invariants, etc.).

Benchmark targets include canonical Verus data structure modules (e.g., ring buffers, lists, trees, maps, atomics) and other real-world examples.

---

## Reinforcement Learning Extensions for Verification Agents

While VerusFT-RL is anchored in supervised fine-tuning, it equally emphasizes reinforcement learning (RL) strategies for LLM-based verification agents. These extensions are directly motivated by systems like VeriStruct and DeepSeek-style reasoners, and they provide a roadmap for going beyond static prompting.

### 1. Three RL Paradigms

We distinguish three complementary paradigms for “doing RL on agents,” now annotated with how they map onto **online** (on-policy) versus **offline** (batch) reinforcement learning styles:

#### Paradigm 1: True policy-gradient RL on model weights (online RL)

- Treat the LLM as a parametric policy \(\pi_\theta(a \mid s)\).
- Interact with an environment built from Verus + benchmark suites.
- Define explicit rewards (verification success, proof quality, VC reductions).
- Update \(\theta\) with policy-gradient methods (PPO, GRPO, etc.).
- This is the paradigm used in RLHF and DeepSeek-style reasoning RL.

**Key property:** the model weights change over time; the policy improves in a lasting way.

#### Paradigm 2: Prompt-based / policy-shaped RL (online-style but weight-frozen)

- Freeze the base LLM and manipulate its “policy” through prompt design, intermediate memories, and tool choices.
- Collect trajectories with success/failure signals (Verus errors, timeouts, successes).
- Perform reflection over these trajectories to write “lessons learned,” update instructions, and curate successful exemplars.
- Future prompts include these reflections, distilled rules, and relevant snippets.

**Key property:** behavior improves via context, memory, and tool choices, not gradients.

#### Paradigm 3: Offline imitation + RL finetuning (offline-to-online hybrid)

- Stage 1: collect high-quality trajectories (VeriStruct data, expert-written proofs) and perform SFT to teach syntax/patterns.
- Stage 2: apply RL finetuning on top of SFT with rewards such as Verus success, compactness, or runtime to refine behavior.

**Key property:** combines the stability of imitation with the exploration of RL.

### 2. Implementation Sketches

#### 2.1 Paradigm 1 (True RL)

- Treat “Verus + benchmark suite” as the RL environment.
- **State:** task context, partially annotated code, verification logs.
- **Action:** LLM-generated edits or suggestions (spec changes, invariant tweaks, ghost code additions).
- **Transition:** apply edits, run Verus, observe new logs/errors.
- **Reward:** shaped by verification success, VC reductions, penalties for syntax errors or excessive code.
- **Loop:** sample proof attempts, evaluate rewards, update the model via PPO/GRPO, iterate.

This approach yields a verifier-specialized LLM that internalizes Verus syntax and proof idioms beyond SFT.

#### 2.2 Paradigm 2 (Prompt-based RL / Reflection)

- Keep the base LLM fixed but implement RL through tool-aware prompting.
- Define an agent loop that decides when to call Verus, edit code, query examples, or finalize proofs.
- Log trajectories (states, tool calls, edits, outcomes).
- Run a reflection prompt after each episode to extract lessons (error patterns, invariant tips, tool-usage strategies).
- Store reflections in an agent memory and retrieve relevant items for future tasks.
- Treat this as an **online-style** process because the agent immediately exploits per-episode feedback (success/failure) to adjust its prompt/memory before the next episode.

Behavior improves over time even though weights stay fixed; the agent’s “policy” is shaped by reflection and memory.

#### 2.3 Paradigm 3 (Offline RL pipeline on top of SFT)

- **Behavior dataset construction:** collect full verification trajectories from VeriStruct, human experts, and the improved prompt-based agent. Each trajectory should contain state/action/observation tuples plus Verus outcomes.
- **Reward labeling:** assign dense rewards offline (e.g., +1 for verified module, +0.3 for VC reduction, −0.2 for syntax errors). Because this happens post hoc, we can apply sophisticated reward models or heuristics without affecting online throughput.
- **Offline RL algorithm:** apply conservative or batch RL objectives (e.g., Decision Transformer, CQL, IQL) to train a policy that respects the support of the dataset while improving toward the labeled reward. This can either fine-tune the base model directly or train a lightweight policy head that proposes edits/specs.
- **Bridging to online RL:** once offline policies are stable, we can safely switch to an online PPO/GRPO loop by seeding it with the offline-tuned policy and gradually enabling new data collection.

This offline-first pathway leverages the minimized Verus corpus plus saved agent traces, reducing risk compared to immediately launching an on-policy verifier loop.

### 3. Prompt-Only RL Plan for a Tool-Using VeriStruct-Style Agent

To ground Paradigm 2, we outline a concrete plan to improve VeriStruct via prompting-only RL. The core ideas are:

1. **Explicit tool-use interface:** expose Verus, compiler, and example libraries as callable tools (e.g., `run_verus`, `search_examples`, `simplify_vc`).
2. **Structured agent loop:** instead of a fixed pipeline, allow the agent to iteratively decide whether to refine invariants, inspect errors, call tools, or edit code.
3. **Reflection + memory:** after each episode, summarize lessons (e.g., “ring buffers need head/tail/len views,” “add lifetime bounds when encountering unconstrained lifetime errors”) and prepend them to future prompts via retrieval.

<details>
<summary><b>Click to expand: Detailed Agent Loop Implementation Sketch (~170 lines of Python)</b></summary>

The following Python skeleton illustrates how a tool-using VeriStruct agent could be organized. **This is a design document, not yet implemented.** It keeps the base LLM frozen but enables multi-step reasoning, tool calls, and reflection-driven improvements.

````python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

class Tool:
    name: str
    description: str

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class VerusTool(Tool):
    name = "run_verus"
    description = "Run Verus on the current code and return success flag and error log."

    def __init__(self, verus_bin: str):
        self.verus_bin = verus_bin

    def run(self, file_path: str) -> Dict[str, Any]:
        success = False
        error_log = "VC failure at function foo: assertion may not hold"
        return {"success": success, "error_log": error_log}

class ExampleSearchTool(Tool):
    name = "search_examples"
    description = "Search verified examples/specs related to a keyword."

    def __init__(self, examples_index: Dict[str, str]):
        self.index = examples_index

    def run(self, query: str) -> Dict[str, Any]:
        hits = [code for k, code in self.index.items() if query.lower() in k.lower()]
        return {"matches": hits[:3]}

TOOLS: Dict[str, Tool] = {}


def register_tool(tool: Tool):
    TOOLS[tool.name] = tool


@dataclass
class AgentStep:
    thought: str
    action: Literal["CALL_TOOL", "EDIT_CODE", "FINISH"]
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    edit_patch: Optional[str] = None
    observation: Optional[Dict[str, Any]] = None

@dataclass
class Episode:
    task_description: str
    initial_code: str
    steps: List[AgentStep] = field(default_factory=list)
    final_code: Optional[str] = None
    success: bool = False

REFLECTION_MEMORY: List[str] = []


def call_llm(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    raise NotImplementedError


def parse_agent_action(raw_text: str) -> AgentStep:
    raise NotImplementedError


def run_verification_episode(task_description: str,
                             initial_code: str,
                             max_steps: int = 10) -> Episode:
    episode = Episode(task_description=task_description, initial_code=initial_code)
    current_code = initial_code
    system_prompt = build_system_prompt_with_reflections()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "You are a Verus proof synthesis agent. Task:\n" + task_description + "\n\n"
                "Current code:\n```rust\n" + current_code + "\n```"
            ),
        },
    ]

    for step_idx in range(max_steps):
        raw_response = call_llm(system_prompt, messages)
        agent_step = parse_agent_action(raw_response)

        if agent_step.action == "CALL_TOOL":
            if agent_step.tool_name is None:
                raise ValueError("tool_name cannot be None for CALL_TOOL action")
            tool = TOOLS[agent_step.tool_name]
            obs = tool.run(**(agent_step.tool_args or {}))
            agent_step.observation = obs
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": f"Tool `{tool.name}` result:\n```json\n{obs}\n```"
            })

        elif agent_step.action == "EDIT_CODE":
            current_code = apply_patch(current_code, agent_step.edit_patch or "")
            episode.final_code = current_code
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": (
                    "Updated code is now:\n```rust\n" + current_code + "\n```"
                ),
            })

        elif agent_step.action == "FINISH":
            episode.final_code = current_code
            break

        episode.steps.append(agent_step)

    verus_tool = TOOLS["run_verus"]
    result = verus_tool.run(file_path="CURRENT_MODULE.rs")
    episode.success = bool(result["success"])

    reflection = run_reflection(episode, result)
    if reflection:
        REFLECTION_MEMORY.append(reflection)

    return episode


def build_system_prompt_with_reflections(k: int = 5) -> str:
    recent_reflections = "\n\n".join(REFLECTION_MEMORY[-k:])
    return f"""
You are a Verus proof-synthesis and repair agent.
You have access to tools:
- run_verus(file_path): run the Verus verifier on the current code
- search_examples(query): retrieve similar verified examples
You must think step-by-step, decide when to call tools, and iteratively improve the code until it verifies.
Here are some lessons learned from previous episodes (if any):
{recent_reflections if recent_reflections else "(no reflections yet)"}
When you respond, ALWAYS output a JSON object describing your next action.
"""


def run_reflection(episode: Episode, final_result: Dict[str, Any]) -> Optional[str]:
    traj_summary = summarize_episode(episode, final_result)
    reflection_prompt = f"""
You are analyzing a proof-synthesis episode.
Episode summary:
{traj_summary}
Please extract 1-3 concrete, generalizable lessons that would help the agent do better on future, similar Verus verification tasks. Write them as bullet points.
"""
    try:
        reflection_text = call_llm(
            "You summarize lessons for a Verus verification agent.",
            [{"role": "user", "content": reflection_prompt}],
        )
        return reflection_text
    except Exception:
        return None


def apply_patch(code: str, patch: str) -> str:
    return code

def summarize_episode(ep: Episode, final_result: Dict[str, Any]) -> str:
    return f"Task: {ep.task_description}\nSuccess: {ep.success}\nSteps: {len(ep.steps)}"


def initialize_agent(verus_bin: str, examples_index: Dict[str, str]):
    register_tool(VerusTool(verus_bin))
    register_tool(ExampleSearchTool(examples_index))
````

</details>

**Summary:** This prompt-only RL approach enhances VeriStruct by (1) enabling deliberate tool-use, (2) supporting multi-step planning instead of a rigid pipeline, and (3) accumulating experience through reflections without modifying model weights. It can also serve as a scaffolding layer for future true-RL or hybrid SFT+RL experiments.

### 4. Prioritizing Offline RL Before Online RL

Given Verus’s relatively high evaluation cost, we plan to **start with offline RL** and use it to de-risk later online experiments. The same minimizer-powered corpus that feeds SFT also seeds the RL logs: for each minimized instance we first store the clean Q/A pair for SFT, then capture the richer reasoning trajectories (tool calls, intermediate hypotheses, Verus transcripts) that offline RL needs. This ensures we are "offline-first" by construction—every RL policy sees only previously logged data until we explicitly green-light online exploration.

1. **Log every VeriStruct/Verus interaction now.** While running SFT evaluations or the prompt-based agent on minimizer-derived tasks, persist full transcripts (code snapshots, tool calls, Verus logs). This becomes the seed offline dataset that already aligns with the Q/A samples used for SFT.
2. **Label and filter trajectories offline.** Add success/failure flags, VC deltas, and metadata such as module size or error category. Remove degenerate runs to keep the distribution clean.
3. **Train offline policies.** Use Decision Transformers or conservative Q-learning variants to learn policies that predict next edits/specs conditioned on the logged context. Because training is offline, we can iterate rapidly without burning Verus cycles.
4. **Validate offline.** Roll out the learned policies in a simulator harness that replays logged states before attempting any fresh Verus calls. Ensure they do not overfit to logging artifacts.
5. **Gradually enable online updates.** Once offline evaluations look solid, transition to online PPO/GRPO or reflection-based loops that collect fresh data, using the offline policy as initialization so that exploration starts from a competent baseline.

This sequencing keeps compute costs manageable, respects Verus throughput limits, and provides a safer glide path toward full online RL.

---

### Phase 2: AST / Structured Representation Ablation (Bonus)

> **Status:** 📋 Future work after text-only baselines are established

Once text-only SFT baselines are stable, we introduce **structured representations** as a scientific ablation to answer: *when does structure actually help?*

#### Candidate Structured Views

| Representation | Description | Use Case |
|----------------|-------------|----------|
| **Linearized AST** | Serialized Rust syntax tree | Complex nested structures |
| **Control-Flow Summary** | Loop structure + invariants | Loop-heavy proofs |
| **Verification State** | VC obligations + triggers | Understanding proof obligations |
| **Proof Graph** | Lemma dependencies | Higher-order reasoning |

#### Experimental Design

For each task (A, B, C), compare:

1. 📝 **Text-only** (baseline)
2. 🌳 **Structure-only** (AST or other view)
3. 🔀 **Hybrid** (text + structure combined)

#### Expected Outcomes

Quantify *when* structure helps:
- ✅ Long, nested loops with complex invariants
- ✅ Tricky View functions and abstractions  
- ✅ Higher-order lemmas and proof obligations
- ❌ Simple specifications and straightforward code

**Goal:** A clear empirical story like "Plain text achieves X% pass rate; AST adds +Y% only for loop-heavy tasks."

---

## Student Subprojects

This repo is designed to support multiple small research projects (e.g., rotation or undergraduate projects). Each subproject can be tackled independently:

| ID | Subproject | Status | Difficulty | Description |
|----|------------|--------|------------|-------------|
| 1 | **Dataset via Minimizer** | 📋 Planned | Medium | Script minimizer calls, build JSONL datasets for Tasks A/B/C, implement deduplication |
| 2 | **SFT for Spec Generation (Task A)** | ✅ Prototype | Easy | Train models on code → spec, evaluate on held-out modules |
| 3 | **SFT for Verified Code Synthesis (Task B)** | 🚧 In Progress | Medium | Train models on spec → code, evaluate by running Verus |
| 4 | **SFT for Proof/Invariant Repair (Task C)** | 📋 Planned | Hard | Build dataset of (broken, error) → (fixed), train repair models |
| 5 | **Benchmark & Evaluation Harness** | 📋 Planned | Medium | Automate Verus compilation, execution, and metric collection |
| 6 | **AST/Structure Ablation Study** | 📋 Planned | Advanced | Design AST encodings, run controlled ablations vs. text-only |

### Getting Started with a Subproject

1. **Choose a project** from the table above
2. **Review the current prototype** in `sft_example.py`
3. **Read the relevant methodology section** in this README
4. **Start small**: implement a minimal version, test it, then expand

---

## Repo Status

Right now, the repo contains a small prototype SFT pipeline (GPT-2 + LoRA + a handful of Verus examples). The plan is to evolve it into:

- a **dataset builder** (minimizer-driven),
- a **task-specific SFT training suite**, and
- a **reproducible benchmark** for verification-oriented SFT.

Current prototype highlights:

- ✅ Parameter-efficient fine-tuning with LoRA
- ✅ 10 diverse Verus training examples (seed set, expanding soon)
- ✅ Configurable training hyperparameters
- ✅ Inference script for testing trained models
- ✅ Small adapter weights (~6.2MB) instead of full model

### Training results (prototype)

- **Loss reduction**: 3.40 → 2.42 (28.7% improvement)
- **Token accuracy**: 45% → 55%
- **Training time**: ~14 seconds for 8 epochs
- **Adapter size**: 6.2MB

---

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0+
- CUDA-capable GPU (recommended, but CPU training is possible)

### Setup

```bash
# Clone the repository
git clone https://github.com/ChuyueSun/VerusSFT.git
cd VerusSFT

# Install dependencies
pip install transformers trl datasets peft accelerate torch

# Optional: Install with specific PyTorch version for your CUDA
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
python -c "import transformers, trl, peft; print('All dependencies installed successfully!')"
```

## Usage

### 1. Training

Run the training script to fine-tune the model on Verus examples:

```bash
python sft_example.py
```

**What happens during training:**
- Loads GPT-2 base model and tokenizer
- Applies LoRA adapters for efficient fine-tuning
- Trains on 10 Verus examples for 10 epochs
- Saves adapter weights to `./sft_output/` (~6MB)

**Expected output:**
```
Loading model...
Training...
Epoch 1/10: loss=3.40
...
Epoch 10/10: loss=2.42
Training complete! Model saved to ./sft_output/
```

**Training time:** ~15 seconds on a modern GPU, ~2 minutes on CPU

### 2. Inference

Test the trained model with new prompts:

```bash
python test_inference.py
```

**What happens:**
- Loads base model + trained LoRA adapter
- Runs 3 test prompts:
  1. Adding specs to a clamp function
  2. Writing a multiply function
  3. Adding specs to a min function
- Generates Verus code completions

**Example output:**
```
Test 1: Add Verus specs to this function:
Generated Code:

fn clamp(x: i32, min: i32, max: i32) -> i32 {
    requires min <= max;
    ensures result >= min && result <= max;
    ...
}
```

### 3. Customizing Training

Edit `sft_example.py` to:
- **Add more examples**: Modify `build_dataset()` function
- **Change model**: Update `model_name` variable
- **Adjust hyperparameters**: Modify the `SFTConfig` settings
- **Change LoRA config**: Update `LoraConfig` parameters

---

## Configuration

### Training Parameters (in `sft_example.py`)

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `num_train_epochs` | 10 | Number of training epochs | Increase for larger datasets |
| `per_device_train_batch_size` | 2 | Batch size per device | Reduce if OOM errors occur |
| `learning_rate` | 3e-4 | Learning rate | Lower for stability, higher for faster convergence |
| `max_seq_length` | 1024 | Maximum sequence length | Must fit prompt + completion |
| `gradient_accumulation_steps` | 2 | Steps to accumulate gradients | Increase for larger effective batch size |

### LoRA Configuration

| Parameter | Default | Description | Notes |
|-----------|---------|-------------|-------|
| `r` | 16 | LoRA rank | Higher = more capacity but larger adapter |
| `lora_alpha` | 32 | LoRA scaling parameter | Typically 2× the rank |
| `target_modules` | `["c_attn", "c_proj"]` | Layers to apply LoRA | GPT-2 specific; adjust for other models |
| `lora_dropout` | 0.05 | Dropout rate | Prevents overfitting |

### Switching Base Models

The current implementation uses GPT-2. To use a different model, edit `sft_example.py`:

```python
# Replace this line:
model_name = "gpt2"

# With one of these recommended models:
model_name = "Qwen/Qwen2.5-Coder-1.5B"  # Recommended for code
# model_name = "bigcode/starcoder2-3b"
# model_name = "deepseek-ai/deepseek-coder-1.3b-base"
```

**Note**: When changing models, update `target_modules` in the LoRA config to match the new model's architecture.

---

## Dataset

### Current Dataset (Seed Set)

The training dataset includes **10 examples** covering common Verus patterns:

| Task Type | Examples | Coverage |
|-----------|----------|----------|
| Specification Generation | 5 | `abs`, `max`, `min`, array bounds, division |
| Code Synthesis | 3 | `add_one`, `double`, `subtract` |
| Combined Spec + Code | 2 | `is_positive`, `square` |

**Example format:**
```python
{
    "text": "Add Verus specs to this function:\n```rust\n...\n```\n```verus\n...\n```"
}
```

### Adding Your Own Examples

Edit the `build_dataset()` function in `sft_example.py`:

```python
examples = [
    {
        "text": "Your task description\n```verus\nYour Verus code here\n```"
    },
    # Add more examples...
]
```

### Additional Verus Code Sources

Expand your dataset with these verified Verus projects:

| Source | Type | Link |
|--------|------|------|
| **Verus Examples** | Core examples & tests | [verus-lang/verus](https://github.com/verus-lang/verus/tree/main/source/rust_verify_test/tests) |
| **VOSTD** | Auto-generated specs | [asterinas/vostd](https://github.com/asterinas/vostd) |
| **Verismo** | Verified systems | [microsoft/verismo](https://github.com/microsoft/verismo) |
| **Memory Allocator** | Memory safety | [verus-lang/verified-memory-allocator](https://github.com/verus-lang/verified-memory-allocator) |
| **Verified Storage** | Storage systems | [microsoft/verified-storage](https://github.com/microsoft/verified-storage) |
| **Vericoding** | Benchmarks | [Beneficial-AI-Foundation/vericoding](https://github.com/Beneficial-AI-Foundation/vericoding) |

---

## File Structure

```
VerusSFT/
├── sft_example.py          # Main training script with dataset builder
├── test_inference.py       # Inference/testing script
├── README.md               # This file: documentation + research proposal
├── .gitignore              # Git ignore rules (excludes sft_output/)
└── sft_output/             # Generated after training (not tracked by git)
    ├── adapter_model.safetensors  # LoRA adapter weights (~6MB)
    ├── adapter_config.json         # LoRA configuration
    ├── tokenizer_config.json       # Tokenizer settings
    └── [other tokenizer files]     # Vocabulary, merges, etc.
```

### What Gets Tracked

- ✅ Source code (`*.py`)
- ✅ Documentation (`*.md`)
- ✅ Configuration files
- ❌ Model outputs (`sft_output/`)
- ❌ Python cache (`__pycache__/`)
- ❌ Jupyter checkpoints

## FAQ

<details>
<summary><b>Why start with GPT-2 instead of a larger code model?</b></summary>

GPT-2 is used for the initial prototype because it:
- Trains quickly (~15 seconds) for rapid iteration
- Works on CPU or small GPUs
- Validates the training pipeline before scaling up

**Next step:** Switch to code-specific models like Qwen2.5-Coder or StarCoder2.
</details>

<details>
<summary><b>How is this different from general code fine-tuning?</b></summary>

VerusFT-RL focuses specifically on:
- **Verification-aware training**: Not just code syntax, but specs and proofs
- **Verus-native evaluation**: Success = code that actually verifies
- **Minimized examples**: Small, focused training samples vs. large codebases
- **Multi-task learning**: Spec generation, code synthesis, and error repair
</details>

<details>
<summary><b>Can I use this for languages other than Rust/Verus?</b></summary>

The approach is general and could work for:
- Dafny
- F*
- Coq
- Lean

Just replace the dataset and adjust the prompts. The SFT pipeline remains the same.
</details>

<details>
<summary><b>Do I need a GPU?</b></summary>

- **For prototyping**: CPU works fine (~2 min training time)
- **For larger models**: GPU recommended (Qwen2.5-Coder-1.5B needs ~6GB VRAM)
- **For production**: Multi-GPU setup for models like StarCoder2-3B
</details>

<details>
<summary><b>How do I add my own Verus examples?</b></summary>

Edit `build_dataset()` in `sft_example.py`:

```python
examples = [
    {"text": "Your task prompt\n```verus\nYour code\n```"},
    # Add more...
]
```

See the [Dataset](#dataset) section for more details.
</details>

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in sft_example.py
per_device_train_batch_size=1  # instead of 2

# Or reduce sequence length
max_seq_length=512  # instead of 1024

# Or reduce LoRA rank
r=8  # instead of 16
```

#### Model Loading Errors
```bash
# If you get tokenizer errors, ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Clear cache and retry
rm -rf ./sft_output
python sft_example.py
```

#### Import Errors
```bash
# Ensure all dependencies are installed
pip install --upgrade transformers trl datasets peft accelerate torch

# Check versions
pip show transformers trl peft
```

#### Training Too Slow
- Enable gradient checkpointing: `gradient_checkpointing=True` in SFTConfig
- Use mixed precision: `fp16=True` (for older GPUs) or `bf16=True` (for newer GPUs)
- Reduce `max_seq_length` if examples are shorter

### Getting Help

- **Verus Documentation**: [https://verus-lang.github.io/verus/](https://verus-lang.github.io/verus/)
- **TRL Documentation**: [https://huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- **PEFT Documentation**: [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft)

---

## Future Improvements

1. **Expand dataset**: Add 50–100+ diverse Verus examples using the minimizer
2. **Better base model**: Evaluate code-specific models (see [Configuration](#configuration))
3. **Evaluation**: Add metrics for Verus specification correctness and verification pass rate
4. **Fine-tune generation**: Adjust decoding parameters (temperature, top-p, beam search)
5. **Structured ablations**: Quantify when AST or other structured signals improve verification outcomes
6. **Integration**: Connect to Verus minimizer for automated dataset generation

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{verusfft-rl2025,
  title={VerusFT-RL: Fine-Tuning and RL for Verification-Oriented Rust/Verus Code},
  author={ChuyueSun and contributors},
  year={2025},
  url={https://github.com/ChuyueSun/VerusSFT}
}
```

## Acknowledgments

- **[Verus](https://github.com/verus-lang/verus)** - The Verus verification system for Rust
- **[Hugging Face](https://huggingface.co/)** - Transformers library and model hub
- **[PEFT](https://github.com/huggingface/peft)** - Parameter-efficient fine-tuning library
- **[TRL](https://github.com/huggingface/trl)** - Transformer Reinforcement Learning library

## Contact

For questions or collaboration opportunities, please open an issue or reach out to the maintainers.
