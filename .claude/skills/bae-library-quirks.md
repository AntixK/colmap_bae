# Skill — Known quirks of the `bae` library

Use when modifying `_debug_step` or the LM logic in `bae_solver.py`. These are non-obvious things about the `bae` library (`sair-lab/bae`, `sair-lab-bae-8a5edab282632443 (1).txt`) and pypose that have bitten this project. Don't re-discover them.

## Quirk 1 — `LM.step()` drops the robust kernel

`bae/optim/optimizer.py:19-56` overrides `pypose.optim.LevenbergMarquardt.step` and **never applies the configured kernel** to (R, J) before the GN solve. The configured `Huber(delta)` is dead code in the bae LM. Pypose's upstream `_step_dense` calls `self.corrector[i](R=R[i], J=J[i])` (`pypose/optim/optimizer.py:585-586`). The bae override does not.

The accept/reject side IS kerneled — via `opt.model.loss(input, target)`, because `opt.model` is `RobustModel` (`pypose/optim/optimizer.py:480`), whose `.loss()` calls `kernel(r.square().sum(-1)).sum()`.

**Net effect**: GN step descends pure L2; accept/reject scores Huber-corrected. Inconsistent on outlier-heavy distributions → kushimoto/mihama stalls.

**Fix**: `_apply_huber_correction` in `bae_solver.py` applies Triggs FastTriggs (`w_i = sqrt(ρ'(s_i))`, then `R_i ← w_i · R_i`, `J_i ← w_i · J_i`) before the GN solve. See `info.md §3.31`.

If you ever rewrite `_debug_step`, this fix is load-bearing. Don't drop it.

## Quirk 2 — `opt.model(input)` is *not* the raw user model

After `LM(model, ...)`, pypose wraps the user's model in `RobustModel`:

```python
# pypose/optim/optimizer.py:480
self.model = RobustModel(model, kernel)
```

`RobustModel.forward(input, target=None)` calls `self.model_forward(input)`, which **unpacks dicts** via `self.model(**input)`. It then wraps the output in a tuple via `self.residuals(...)`.

So:
- `opt.model(inp)` with dict `inp` returns `(R,)` — works even though `ColmapReproj.forward()` takes positional args.
- `list(opt.model(inp))[0]` extracts `R`.
- `opt.model.loss(inp, None)` applies the kernel.

**Bad pattern (we already hit this and crashed)**:
```python
def _compute_huber_loss_from_model(model, inp, delta):
    R = list(model(inp))[0]   # ← model here is the RAW ColmapReproj, not RobustModel!
                              # crashes on dict input
```

**Right pattern**: just call `opt.model.loss(inp, None)`. It's already kerneled.

## Quirk 3 — LM retry: `A_attempt = A_base.clone()` matters

`bae/optim/optimizer.py:LM.step` has a retry loop that does:

```python
diagonal_op_(A, op=partial(torch.mul, other=1 + pg["damping"]))
```

in place on `A` each attempt. The damping is **cumulative across rejected attempts** — diag becomes `D · ∏(1+λᵢ)`, not the intended `D · (1+λₖ)`.

`bae_solver.py:_debug_step` fixed this by cloning per attempt:

```python
A_base = opt.mm(J_T, J)
diagonal_op_(A_base, op=partial(torch.clamp_, min=pg["min"], max=pg["max"]))
while opt.last <= opt.loss:
    A_attempt = A_base.clone()  # ← fresh copy each attempt
    diagonal_op_(A_attempt, op=partial(torch.mul, other=1 + pg["damping"]))
    step = opt.solver(A_attempt, rhs)
```

Don't undo this. The catastrophic damping spiral in the pre-retri camera-only polish call (info.md §3.31) was the same root cause.

## Quirk 4 — `pp.SE3` `add_` interprets 6-D tangent as `se3` Lie algebra

`bae/optim/optimizer.py:LM.update_parameter:67-74`:

```python
if getattr(param, 'trim_SE3_grad', False):
    param[..., :7] = pp.SE3(param[..., :7]).add_(pp.se3(d.view(param.shape[0], -1)[..., :6]))
```

The 6-D step `d[..., :6]` is interpreted as `(δtx, δty, δtz, δrx, δry, δrz)` in `se3` Lie algebra and applied via the exponential map.

This is the *correct* manifold step on SE(3). Don't replace it with naive quaternion addition.

## Quirk 5 — pypose `TrustRegion.update` parameters

pypose's `TrustRegion(radius, max, up, down)` strategy:

- `radius` — initial trust-region radius (damping ≈ `1/radius`).
- `up` — multiply radius by this on accept (default `2.0`).
- `down` — multiply radius by this on reject (default `0.5`).
- Radius grows on accept **only when step quality crosses an internal threshold**.

On low-quality accepted steps (quality < ~0.5), radius is **unchanged** — not multiplied by `up`. So damping can get stuck at the initial value for many iterations if every accepted step is low-quality. Observed pattern on kushimoto iter1.full (info.md §3.32).

This is **different** from Ceres's Nielsen rule: `radius / max(1/3, 1 - (2q-1)^3)` on accept (cubic in quality), `radius / decrease_factor` on reject (decrease_factor *= 2 each reject). See `ceres.md §13.2`.

If you propose to swap the strategy, be aware that the quality threshold for radius growth is what makes pypose's behavior dataset-fragile.

## Quirk 6 — Sparse `diagonal_op_` may need triton or pure-Python wrapper

`bae.sparse.py_ops.diagonal_op_` uses Triton kernels under CUDA. On `SparseCsrCUDA` matrices, the kernel registration competes with PyTorch's default (`warp_wrappers.py:74`) — that's where the registry warning at startup comes from. Don't worry about it.

If you see `UnboundLocalError: cannot access local variable 'indices' where it is not associated with a value` in `py_ops.diagonal_op_`, it's the `bae` library's bug; check the issue.md notes — there's an older workaround.
