# colmap_bae — Claude project guide

This project replaces Ceres BA in COLMAP's global mapper with the GPU-based BAE solver. Read these documents and follow these rules before making any change.

## Read first, every session

1. **[problem.md](../problem.md)** — task, success criteria, editable file set.
2. **[info.md](../info.md)** — every prior experiment, what failed, what worked.
   Don't re-derive what's already documented.
3. **[ceres.md](../ceres.md)** — verified, line-cited reference for how Ceres BA
   works in this codebase. §13 is the deep dive into the solver internals.

Supporting:

- **[ref.md](../ref.md)** — BAE compute-graph patterns (BAL, gauge-fixed, PGO).
- **[pcg.md](../pcg.md)** — PCG vs Schur explanation.
- **[notes.md](../notes.md)** — earlier session notes / scratch.
- **[issue.md](../issue.md)** — original integration problem statement.

Source dumps (use these instead of memory — see [skills/use-source-dumps.md](skills/use-source-dumps.md)):

- `sair-lab-bae-8a5edab282632443 (1).txt` — `bae` library
- `ceres-solver-ceres-solver-8a5edab282632443.txt` — Ceres Solver
- `cre185-instantsfm-8a5edab282632443.txt` — InstantSfM

## Rules (non-negotiable)

Read [.claude/rules/](rules/) before any action. Each rule has examples of what to do and what not to do.

- **[01-editable-files.md](rules/01-editable-files.md)** — you may edit only four C++/Python files. Don't proactively expand.
- **[02-no-fabrication.md](rules/02-no-fabrication.md)** — every citation must be verified from a source you just inspected.
- **[03-diagnose-before-tune.md](rules/03-diagnose-before-tune.md)** — exit reason + diagnostic numbers before any parameter change.
- **[04-tone.md](rules/04-tone.md)** — terse, no preambles, no lectures.
- **[05-cross-language-boundary.md](rules/05-cross-language-boundary.md)** — always verify probe equality before suspecting the optimizer.
- **[06-gpu-only.md](rules/06-gpu-only.md)** — BAE runs on GPU; no CPU fallback.
- **[07-instantsfm-not-oracle.md](rules/07-instantsfm-not-oracle.md)** — InstantSfM is one data point, not ground truth.
- **[08-no-overpromising.md](rules/08-no-overpromising.md)** — every proposal needs cost, falsifier, and a diagnostic-first alternative.
- **[09-handle-pushback.md](rules/09-handle-pushback.md)** — after pushback, respond with code, measurement, or "I don't know yet" — never a rephrased claim.
- **[10-coding-style.md](rules/10-coding-style.md)** — C++ (clang-format Google) and Python (ruff) conventions enforced by the formatters plus project-specific habits.

## Skills (how-to references)

Read the one(s) relevant to the current task.

- **[colmap-codebase.md](skills/colmap-codebase.md)** — directory map, module layers, key classes, build & test commands, code style. Read when you need to locate something in COLMAP or rebuild after a C++ change.
- **[bae-compute-graph.md](skills/bae-compute-graph.md)** — `TrackingTensor` / `@map_transform` / `trim_SE3_grad`. Read before modifying the forward model in `bae_solver.py`.
- **[bae-library-quirks.md](skills/bae-library-quirks.md)** — the dropped-kernel bug, the `RobustModel` wrapper, `A_base.clone()`, manifold step semantics, pypose `TrustRegion` behavior. Read before modifying `_debug_step`.
- **[ceres-as-reference.md](skills/ceres-as-reference.md)** — when and how to consult `ceres.md`. Includes the §13.8 correction (global_mapper.h disables auto-select).
- **[instantsfm-as-data-point.md](skills/instantsfm-as-data-point.md)** — what InstantSfM actually does and what its silence implies.
- **[use-source-dumps.md](skills/use-source-dumps.md)** — `grep` patterns for navigating the three flat-file source dumps.
- **[read-bae-logs.md](skills/read-bae-logs.md)** — mapping log lines to diagnoses. Healthy vs broken iter1.full signatures.
- **[pcg-vs-schur.md](skills/pcg-vs-schur.md)** — one-page distinction, with the empirical caveat.
- **[propose-experiment.md](skills/propose-experiment.md)** — six-field template for any proposed change.
- **[load-bearing-fixes.md](skills/load-bearing-fixes.md)** — fixes that took the longest to find and have the biggest effect. Don't regress them.

## Persistent memory

Cross-session notes live at:

`/home/eb-anands/.claude/projects/-home-eb-anands-exp-colmap-bae/memory/`

Contents (`MEMORY.md` indexes the rest):

- **`project_bae_convergence.md`** — BAE+Huber(1.0) stalls when errors >>1px; pre-BA filtering observation.
- **`project_bae_kernel_dead_code.md`** — bae lib's `LM.step` ignores kernel; apply corrector inline.
- **`feedback_instantsfm_not_universal_baseline.md`** — InstantSfM design choices aren't ground truth.

Update memory when you discover a load-bearing pattern that would save the next agent time. Don't dump every observation; memory is for surprising / non-obvious / load-bearing things.

## Current state in one paragraph

The kernel correction (info.md §3.31) is the biggest single fix. After it,
bridge and other moderate datasets reach near-Ceres quality at ~2× speed.
kushimoto / mihama / barn show step-direction-quality issues that
parameter tuning has been shown not to fix (info.md §3.32–§3.34, cross-dataset
benchmark in §3.35). The diag(JᵀJ) lower bound on κ does **not** correlate
with the failure pattern across datasets, contradicting the textbook
"Schur fixes BA conditioning" narrative without measurement. Real κ
estimation on our matrices is the cheapest next diagnostic before any
larger structural change.
