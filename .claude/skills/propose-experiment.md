# Skill — Proposing an experiment

Every proposed change to the BAE pipeline must be presented in this six-field shape. This makes it easy for the user to accept, reject, or counter-propose without re-deriving the reasoning.

## The template

1. **Measured observation** — what the run log / probe / benchmark shows.
   Quote the log line. Cite the file or section.
2. **Hypothesis** — the mechanism that would explain (1).
3. **Predicted outcome if hypothesis is right** — concrete number, with units.
4. **Falsifier** — the measurement that would tell us the hypothesis is wrong.
5. **Cost** — files touched (must be in editable set), LoC estimate, days.
6. **Prior status** — has this been tried? cite `info.md` section. If
   superseded or failed, explain what's changed since.

## Bad — speculation with no measurements

> "Let me try smaller initial radius."

## Bad — missing falsifier

> "Schur reduction will fix kushimoto. ~1 week, contained to bae_solver.py."

## Good

> **Measured observation**: bridge iter1.full hit `_run_ba exit:
> reason=max_iter n_it=150/150 cost_drop_total=3.7e-2 windowed_imp=nan`
> in the post-§3.31 benchmark (info.md §3.32). Cost was still descending
> linearly at exit.
>
> **Hypothesis**: iter1.full is iter-budget-bound on bridge. The current
> `max_num_iterations=150` is the binding constraint, not optimizer quality.
>
> **Predicted outcome**: raising the cap to 200 should let iter1.full
> drop below the current p50 of 1.213, toward Ceres's 0.821.
>
> **Falsifier**: if iter1.full p50 doesn't change measurably between
> max_iter=150 and max_iter=200, the iter cap was not the bottleneck.
>
> **Cost**: 1 line in `bundle_adjustment_bae.h:16`. C++ rebuild required.
>
> **Prior status**: max_iter=100 → 150 in info.md §3.26 (no measurement
> recorded). max_iter=150 → 300 in §3.33 worked on bridge specifically
> but made other datasets slower per the §3.35 benchmark — we settled on
> 200 as the middle ground.

## Notes on each field

### (1) Measured observation

A log line or probe number from a real run. Not "I think" or "presumably."
Grep the relevant `bench/<dataset>/<dataset>_bae/run.log` and quote the line.

### (2) Hypothesis

One mechanism. Two is too many. If you have two competing hypotheses, do
the cheapest discriminating measurement first and propose one of them.

### (3) Predicted outcome

A specific number with units. "Faster" / "better" / "improved" are not
predictions; they're escape hatches. "iter1.full p50 drops below 1.0" is
a prediction.

### (4) Falsifier

What measurement would tell you the hypothesis is wrong? If you can't
write a falsifier, the hypothesis isn't testable — back to (2).

### (5) Cost

Read [rules/01-editable-files.md](../rules/01-editable-files.md) before
naming a file. If the file isn't editable, you need a different design.

LoC estimate matters because it sets the bar for "diagnostic-first
alternative" (rule 08): a 200-line change should be preceded by a
half-day diagnostic confirming the hypothesis.

### (6) Prior status

`info.md` Part III is sorted by section and contains every prior
experiment with a confirmed/failed/refuted tag. Search it. Don't repeat
a failed tuning experiment.
