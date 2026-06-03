# Rule 02 — Never fabricate citations or claims

If you cite a paper, table, line number, quote, function name, or measurement, **it must be verifiable from a source you have just inspected**. Citing from training-data memory is fabrication.

## Hard rules

- A "Table N in paper X" reference is a claim about the paper's actual content. Either fetch the paper (WebFetch / pdftotext) and verify, or say "I'm recalling from memory; verify before relying on this."
- A "file.cc:123" reference is a claim about line numbers in this repo. `grep -n` or `Read` first.
- A "Ceres does X" reference is a claim about the Ceres source dump. Open it (`ceres-solver-ceres-solver-8a5edab282632443.txt`) and confirm.
- A "BAE library does X" reference must point at `sair-lab-bae-8a5edab282632443 (1).txt` or the installed `bae/` source.
- A "InstantSfM does X" reference must point at `cre185-instantsfm-8a5edab282632443.txt` or `InstantSfM/instantsfm/`.

## When in doubt

Say "I don't know yet" or "let me verify." Both are cheap. Fabrication is expensive — it has cost trust in this project before.

## Concrete failure mode that happened in this project

Earlier in the investigation, the assistant claimed:

> "From their Table 1 / Section 4: on the Trafalgar dataset, they report κ of the full normal equations is intractable to even estimate... They explicitly say 'the Schur complement matrix S is much better conditioned than H'..."

After `WebFetch` and `pdftotext` extraction, the BAL paper:
- Has **no Table 1 with conditioning measurements**
- Contains **no quote** matching "much better conditioned than H"
- States only `κ(S) ≤ κ(Hµ)` as a theoretical inequality, no numerical comparison

That fabrication wasted user time and damaged trust. Don't repeat the pattern.

## Bad

```
"The BAL paper measures κ at ~10^8 on Trafalgar."
```
(no such measurement exists in the paper)

## Good

```
"Agarwal et al. (BAL, 2010) state κ(S) ≤ κ(Hμ) as a theoretical bound
(see ceres-solver source dump `internal/ceres/iterative_schur_complement_solver.h`
docstring, which cites the paper). The bound is not measured numerically
in the paper; doing so on our matrices is open work."
```

## Good — admitting uncertainty

```
"I recall the BAL paper measuring this, but I don't have the PDF open.
Let me fetch it before claiming the exact number."
```
