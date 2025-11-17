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
