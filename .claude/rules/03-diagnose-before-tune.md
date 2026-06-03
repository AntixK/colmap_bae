# Rule 03 — Diagnose before tuning

Before proposing any parameter change, you must state, from the actual run log:

1. **The exit reason of the LM** on the failing call. It is logged in
   `bae_solver.py`'s `_run_ba exit:` line:
   ```
   [BAE] _run_ba exit: reason=func_tol n_it=14/300 cost_first=... cost_last=...
       cost_drop_total=... windowed_imp=...
   ```
2. **The diagnostic numbers** at the failure: `|Jᵀr|` per block, `quality`,
   `damping`, `step_norm`, `cost_drop_total`, `windowed_imp`.
3. **How the proposed change addresses the *measured* failure mode**, not
   an imagined one.

If you cannot fill in (1)–(3), you are not allowed to propose the tuning change.

## Why this rule exists

`info.md` Part III is a graveyard of parameter-tuning experiments that all
failed because they were guessed, not diagnosed:

- §2.2.1 — kernel delta δ=0.5/1.0/2.0/median
- §2.2.6 — adaptive `[2, 5, 10, 20, 50, 100, 500]` pixel ladder
- §2.4.1 — PCG tolerance `tol=1e-5 → 1e-7`
- §3.14 — trust-region radius `1e4 → 0.3`
- §3.21 — `up=2.0/1.5/1.3, down=0.5/0.7`
- §3.22 — dataset-fragile policy sweep (confirmed no global policy works)
- §3.33 — `func_tol=5e-4 → 5e-5`, `max_iter=150 → 300` (didn't fix kushimoto)
- §3.34 — pre-filter threshold tightening (no-op on hard datasets)

Each tuning attempt was reasonable on the surface. None was load-bearing.
The actual fix (§3.31, kernel correction) came from **reading the source code
of the bae library** and noticing that `LM.step()` in `bae/optim/optimizer.py`
never calls the kernel corrector. That is a code observation, not a tuning idea.

## The exit-reason → action table

| `_run_ba exit reason` | What it means | Tuning lever that *might* help |
|---|---|---|
| `max_iter` with `cost_drop_total > 0` | Iter-budget-bound; cost still descending at exit | Raise `max_num_iterations` |
| `func_tol` with `windowed_imp ≈ func_tol` | LM at a local plateau in the kerneled cost surface | Nothing tuning can do — this is a step-direction issue (§3.29). Investigate the linearization, not the budget. |
| `loss_repeat` | Cost values literally repeating; numerical underflow or damping saturated | Look at `damping_saturation=True` flag; fix the LM step path, not the threshold |

## Bad

```
"Let me try a smaller initial radius — `radius=0.05` instead of `0.3`."
```
(No measurement supporting it; tried in §3.21–3.22, failed.)

## Good

```
"On kushimoto iter1.full, `_run_ba exit: reason=func_tol n_it=14/300
windowed_imp=4.29e-5`. The exit is plateau-driven, not budget-driven, so
raising max_iter or tightening func_tol won't help (§3.33 confirms).
The next experiment must address step-direction quality, not the budget."
```
