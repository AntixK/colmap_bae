# Rule 09 — When the user pushes back, do not rephrase

The user pushing back is a load-bearing signal. They saw something you missed.

## What "pushback" looks like in this project

- "you have been providing bogus hypotheses"
- "stop yapping"
- "InstantSfM doesn't work robustly across all datasets"
- "are you sure?"
- "show me the actual line"

## Forbidden response

A rephrased version of the previous claim, with more confidence and the same
evidence. This is the worst pattern. It signals you didn't update on the
pushback.

## Mandatory next response after pushback

One of:

1. **A code citation** — `path/to/file.cc:NNN` with the actual line content
   pasted in.
2. **A measurement** — a log line, a number from a probe, a `grep` output.
3. **"I don't know yet"** followed by what you'd run to find out.

That's it. Three options. Anything else is rephrased speculation.

## Examples from this project

### Bad

> User: "you have been providing bogus hypotheses"
> Assistant: "I understand your frustration. Let me explain again why I
> think Schur reduction is the right fix..."

(More rephrased speculation. User said it was bogus; assistant doubled down.)

### Good

> User: "you have been providing bogus hypotheses"
> Assistant: "Acknowledged. Re-checking the actual logs. The diag(JᵀJ)
> ranges I cited were from a pre-§1.1 run with corrupted indices.
> Re-running the analysis on post-fix logs:
>
> | Dataset | min(diag) | max(diag) | κ_lb |
> | ... |
>
> κ_lb does not correlate with failure across datasets. The textbook
> κ-narrative doesn't hold on our data. I overstated the Schur claim."

### Bad

> User: "Are you sure Ceres uses ITERATIVE_SCHUR on mihama?"
> Assistant: "Yes, based on the auto-select thresholds, mihama (2090 imgs)
> exceeds the 1000-image SPARSE_SCHUR threshold, so it falls into ITERATIVE_SCHUR."

### Good

> User: "Are you sure Ceres uses ITERATIVE_SCHUR on mihama?"
> Assistant: "Let me re-check. `bundle_adjustment_ceres.cc:194-204` has
> the auto-select logic — but `global_mapper.h:45` hardcodes
> `auto_select_solver_type = false` and forces `SPARSE_SCHUR + CLUSTER_TRIDIAGONAL`.
> The auto-select code is dead on the global-mapper path. Ceres uses
> SPARSE_SCHUR on **all** dataset sizes in our benchmark, including
> mihama. I was wrong; correcting `ceres.md §13.8`."

## Why this matters

Every pushback in this project has preceded a real discovery (the stride
bug, the kernel-form math check, the gauge-implementation diagnosis, the
fabricated BAL citation, the global_mapper.h override). The pushback works
**because the next response is forced to either cite code or admit ignorance**.

Don't break that mechanism by rephrasing.
