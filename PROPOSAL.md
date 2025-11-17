# VerusSFT: Supervised Fine-Tuning for Verification-Oriented Rust/Verus Code

## Overview

VerusSFT is a research repo for exploring **supervised fine-tuning (SFT)** of language models on **verification-oriented Rust/Verus code**.

General-purpose code LLMs often struggle with Verus-specific concepts:

- `exec` / `ghost` / `proof` modes
- `requires` / `ensures` specifications
- View functions and typestate-like abstractions
- Loop invariants and `decreases` clauses
- Proof blocks and lemma structure
- Verus error messages and verification traces

The goal of this project is to build and evaluate SFT pipelines that make models **genuinely Verus-aware**, and to understand when we really need structured representations (like ASTs) beyond plain text.

---

## Project Goals

### Primary Goals

1. **Build a minimized, high-quality text-only dataset** of Verus code, specs, proofs, and error traces, starting from open-source Verus projects.
2. **Train SFT models** on three core tasks:
   - **Task A — Code → Specifications** (generate `requires`/`ensures`, Views, invariants).
   - **Task B — Specifications → Verified Code** (fill in executable + ghost + proof code).
   - **Task C — Error-Guided Proof/Invariant Repair** (fix code/specs using Verus error messages).
3. **Evaluate models using Verus itself** (verification pass rate), not just syntax or text similarity.

### Secondary Goal (Ablation / Bonus)

4. After strong **text-only baselines** exist, introduce **AST/structured encodings** and measure their incremental benefit via ablation studies.

---

## Motivation

Formal verification in Verus embeds specifications, invariants, and proofs directly in Rust code. Systems like [VeriStruct](https://github.com/ChuyueSun/VeriStruct) and other prototype assistants show that LLMs can assist with specification and proof tasks, but current code models still:

- mis-handle Verus modes and ghost state,
- omit or weaken specs,
- generate invalid proofs, or
- fail to interpret Verus error messages.

**Hypothesis:**
Supervised fine-tuning on a curated corpus of **minimized, self-contained Verus examples** will significantly improve model performance on spec generation, verified code synthesis, and proof repair—even *without* structured encodings. Once we have strong text baselines, we can rigorously ask: *when do ASTs or other structural views actually help?*

## Related Work

VerusSFT builds on a growing body of literature exploring language-model-assisted verification. Prior systems such as VeriStruct focus on prompt engineering and retrieval but stop short of supervised fine-tuning on minimized Verus corpora. Recent work like the arXiv preprint [arXiv:2505.20302](https://arxiv.org/pdf/2505.20302) examines adjacent verification-aware fine-tuning strategies, underscoring the demand for reproducible datasets, Verus-native evaluation harnesses, and head-to-head comparisons between text-only and structure-augmented representations. This proposal positions VerusSFT to complement that line of research by emphasizing minimized examples, multi-task SFT, and rigorous Verus-based metrics.

---

## Methodology

The project is split into two main phases:

1. **Phase 1 — Text-Only Dataset + SFT (Core)**
2. **Phase 2 — AST / Structured Representation Ablations (Bonus)**

### Phase 1: Dataset Preparation (Minimized Text-Only Data)

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

This yields a dataset that is small, focused, and tailored for SFT.

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

While VerusSFT is anchored in supervised fine-tuning, we also plan to explore reinforcement learning (RL) strategies for LLM-based verification agents. These extensions are directly motivated by systems like VeriStruct and DeepSeek-style reasoners, and they provide a roadmap for going beyond static prompting.

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

The following Python-style skeleton illustrates how a tool-using VeriStruct agent could be organized. It keeps the base LLM frozen but enables multi-step reasoning, tool calls, and reflection-driven improvements.

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

This prompt-only RL approach enhances VeriStruct by (1) enabling deliberate tool-use, (2) supporting multi-step planning instead of a rigid pipeline, and (3) accumulating experience through reflections without modifying model weights. It can also serve as a scaffolding layer for future true-RL or hybrid SFT+RL experiments.

### 4. Prioritizing Offline RL Before Online RL

Given Verus’s relatively high evaluation cost, we plan to **start with offline RL** and use it to de-risk later online experiments:

1. **Log every VeriStruct/Verus interaction now.** While running SFT evaluations or the prompt-based agent, persist full transcripts (code snapshots, tool calls, Verus logs). This becomes the seed offline dataset.
2. **Label and filter trajectories offline.** Add success/failure flags, VC deltas, and metadata such as module size or error category. Remove degenerate runs to keep the distribution clean.
3. **Train offline policies.** Use Decision Transformers or conservative Q-learning variants to learn policies that predict next edits/specs conditioned on the logged context. Because training is offline, we can iterate rapidly without burning Verus cycles.
4. **Validate offline.** Roll out the learned policies in a simulator harness that replays logged states before attempting any fresh Verus calls. Ensure they do not overfit to logging artifacts.
5. **Gradually enable online updates.** Once offline evaluations look solid, transition to online PPO/GRPO or reflection-based loops that collect fresh data, using the offline policy as initialization so that exploration starts from a competent baseline.

This sequencing keeps compute costs manageable, respects Verus throughput limits, and provides a safer glide path toward full online RL.

---

### Phase 2: AST / Structured Representation Ablation (Bonus)

Once text-only SFT baselines are stable, we introduce **structured representations** as a scientific ablation, not as a dependency of the core pipeline.

Candidate structured views:

- linearized Rust AST,
- control-flow summaries (loops + invariants),
- high-level "verification state" descriptors (e.g., obligations, triggers),
- simplified proof-tree or lemma-call graphs.

For each core task (A/B/C), we compare:

1. **Text-only** (baseline)
2. **Structure-only** (AST or other view)
3. **Text + structure** (concatenated or multi-part prompt)

The goal is to quantify *when* structure helps, e.g.:

- long, nested loops and complex invariants,
- tricky Views/abstractions,
- higher-order lemmas.

This yields a clear story: "For verification-style Rust, plain text SFT gets us X%; AST adds +Y% only for Z-type tasks."

---

## Student Subprojects

This repo is designed to support multiple small research projects (e.g., rotation or undergraduate projects). Example subprojects:

1. **Dataset via Minimizer (Core Plumbing)**
   - Script the minimizer calls.
   - Build JSONL datasets for Tasks A/B/C.
   - Implement deduplication and quality filters.

2. **SFT for Spec Generation (Task A)**
   - Train models on code → spec.
   - Evaluate on held-out modules.

3. **SFT for Verified Code Synthesis (Task B)**
   - Train models on spec → code.
   - Evaluate by running Verus on generations.

4. **SFT for Proof/Invariant Repair (Task C)**
   - Build a dataset of (broken example, error) → (fixed example).
   - Train models to repair common errors.

5. **Benchmark & Evaluation Harness**
   - Automation to compile, run Verus, and collect metrics.

6. **AST / Structure Ablation Study (Advanced)**
   - Design AST/structured encodings.
   - Run controlled ablations vs. text-only baselines.

---

## Repo Status

Right now, the repo contains a small prototype SFT pipeline (GPT-2 + LoRA + a handful of Verus examples). The plan is to evolve it into:

- a **dataset builder** (minimizer-driven),
- a **task-specific SFT training suite**, and
- a **reproducible benchmark** for verification-oriented SFT.

Contributions, experiments, and issues are very welcome—especially around new Verus benchmarks, better base models, and structured representation ideas.
