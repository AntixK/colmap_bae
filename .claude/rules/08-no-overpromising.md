# Rule 08 — No overpromising

When recommending a structural change, state honestly:

- Cost estimate (lines of code, days)
- Risk level (low / medium / high)
- What would refute the hypothesis if the change doesn't work
- Concrete falsifier: what measurement would tell us we were wrong

Never present a speculative fix as "the answer." Hedge with the uncertainty.

## Why this rule exists

The Schur-reduction recommendation went through three over-promised iterations
in this project:

1. First proposal framed Schur as "the structural fix" based on the textbook
   κ(S) ≤ κ(H) inequality.
2. User asked to verify the BAL paper citation. Assistant had **fabricated**
   the citation specifics. Real BAL paper has no Table 1 on conditioning.
3. User asked to actually measure κ on our datasets. Diag-range lower bound
   showed **κ_lb is similar across all 8 datasets** — the dataset with the
   highest κ_lb (ignatius) succeeds, the one with the lowest (mihama) fails.
   The textbook narrative does not track our failure pattern.

The honest framing all along should have been: "Schur is the textbook fix
for an ill-conditioned BA Hessian, but I haven't measured κ on our matrices.
Before recommending the implementation, let's spend half a day on a Lanczos
estimator and check whether κ actually correlates with failure."

## The honest-recommendation template

When proposing a structural change, fill in all six fields:

1. **Mechanism**: what's broken, in measurable terms
2. **Proposed fix**: what changes
3. **Cost**: rough LoC + days
4. **Falsifier**: the measurement that would tell us the proposal is wrong
5. **Diagnostic-first alternative**: the cheapest experiment to gain
   confidence before committing
6. **Prior status**: has this been tried? cite `info.md` section if so

## Bad

```
"Schur reduction is the answer. Implementing it should close the kushimoto gap."
```

## Good

```
**Mechanism**: PCG-on-full-system step-direction quality on ill-conditioned JᵀJ
(info.md §3.29). Hypothesized as the kushimoto failure mode.

**Proposed fix**: Implicit-Schur PCG in `_debug_step`, with SCHUR_JACOBI
preconditioner mirroring `ceres.md §13.4-13.6`.

**Cost**: ~220 lines new Python, ~1 week including correctness debugging.

**Falsifier**: If iter1.full p50 on kushimoto stays above 3.5 px after a
correctly-implemented Schur, the diagnosis is refuted.

**Diagnostic-first alternative**: Add a Lanczos-based κ estimator (~50 lines,
half a day). Run it on bridge / kushimoto / ignatius / mihama at iter1.full
LM entry. If real-κ doesn't correlate with failure across datasets, Schur
is unlikely to help — invest the diagnostic effort first.

**Prior status**: Untested per info.md §3.30. The earlier 'Schur is the
fix' framing leaned on an unmeasured textbook claim (caveats in §13.10).
```
