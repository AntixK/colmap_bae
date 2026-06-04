# Rule 10 — Coding style

This codebase has strict, machine-enforceable style. Match it. The formatters are authoritative; this rule documents the conventions the formatters enforce plus the project-specific habits they don't.

## Run the formatters before claiming a change is done

```bash
scripts/format/c++.sh       # clang-format on C++ files
scripts/format/python.sh    # ruff format + lint on Python files
```

CI fails on unformatted code. If you can't run the script (e.g., toolchain unavailable), at minimum match the style of the file you're editing.

## C++ style (`.clang-format`)

Base: **Google style** with these project overrides:

- `BinPackArguments: false` and `BinPackParameters: false` — when a function call or signature wraps, **one argument per line**, never partially packed.
- `DerivePointerAlignment: false` — references and pointers attach to the **type**: `const Foo& x`, `Foo* p`. Never `Foo &x` or `Foo *p`.
- `IncludeBlocks: Regroup`, `SortIncludes: true` — clang-format reorders includes. The categories (priority shown):
  1. `"colmap/..."` headers
  2. `"pycolmap/..."` headers
  3. `"thirdparty/..."` headers
  4. System angle-bracket headers `<algorithm>`, `<vector>`, etc.
  5. Everything else (e.g., `<pybind11/...>`)

  Blank lines separate the groups. Don't fight the formatter on this.

### Indentation, braces, line length

- 2-space indent. No tabs.
- 80-column line limit (Google default).
- Opening braces on the same line as the declaration. Closing brace alone.
- `else` on the same line as the closing `}` of the previous block.

### Naming (from `AGENTS.md` / `colmap-codebase.md`)

| Element | Convention |
|---|---|
| Classes / structs | `PascalCase` |
| Methods, free functions | `PascalCase` (`FindNextImages()`, `Solve()`) |
| Member variables | `snake_case_` (trailing underscore) |
| Local variables, parameters | `snake_case` |
| Constants / enums | `kPascalCase` or `UPPER_SNAKE_CASE` |
| Files | `snake_case.h` / `snake_case.cc` / `snake_case_test.cc` |
| Transforms | `target_from_source` (e.g. `cam_from_world`, `rig_from_world`) |
| Coordinates | `x_in_y` (e.g. `point3D_in_world`) |

Use the special identifier types from `util/types.h`: `camera_t`, `image_t`, `frame_t`, `rig_t`, `point2D_t`, `point3D_t`, `sensor_t`. Don't use bare `int` / `size_t` for these.

### Headers

- `#pragma once` (not include guards).
- One blank line after the pragma, then includes in the regrouped order, then the namespace block.

### Comments

- `//` only. No `/* */` blocks.
- Comment the **why**, not the **what**. Reference investigation notes when relevant — links to `info.md §X.Y` sections are first-class.
- File-section banners use the 80-char `//` line:
  ```cpp
  ////////////////////////////////////////////////////////////////////////////////
  // BaeBundleAdjustmentOptions
  ////////////////////////////////////////////////////////////////////////////////
  ```
- Long rationale comments above declarations are encouraged (see `bundle_adjustment_bae.h` lines 11-17 for an example).

### Logging and error handling

- glog macros: `LOG(INFO) << "..."`, `LOG(WARNING)`, `LOG(ERROR)`.
- `LOG_FIRST_N(INFO, 1)` for one-shot warnings.
- `VLOG_IS_ON(N)` / `VLOG(N)` for verbosity-gated logs.
- Option checks: `CHECK_OPTION_GT(field, 0)` in `Check()` methods.
- Pre-conditions in hot paths: `THROW_CHECK(...)`, `THROW_CHECK_EQ(...)`, `THROW_CHECK_LT(...)`.
- Stream concatenation, not `printf`-style. Use `std::ostringstream` only when you must build a string.

### Namespaces

- All project code in `namespace colmap { ... }`. Close with `}  // namespace colmap` (two spaces before the comment).
- Translation-unit-local helpers go in an **anonymous namespace** at the top of the `.cc`:
  ```cpp
  namespace colmap {
  namespace {
  void LocalHelper() { ... }
  }  // namespace
  ```
- Avoid `using namespace`. Aliases like `namespace py = pybind11;` are fine.

### Modern C++

- Range-based for: `for (const auto& [image_id, image] : reconstruction.Images())`.
- Structured bindings encouraged when the names are meaningful.
- `auto` for lambdas, iterator-typed locals, and obvious types. Spell the type when it's load-bearing for the reader.
- `std::unique_ptr` for ownership, `std::shared_ptr` only when sharing is required (`Solve()` returns `std::shared_ptr<BundleAdjustmentSummary>`).
- `[[nodiscard]]` is not used as a project default; don't add it without precedent.

### Bad

```cpp
void
foo(const Bar &a,Bar* b){
    if(a.x==0){LOG(INFO)<<"zero";}
}
```

### Good

```cpp
void Foo(const Bar& a, Bar* b) {
  if (a.x == 0) {
    LOG(INFO) << "zero";
  }
}
```

## Python style (`ruff.toml`)

Ruff config selects: `E` (pycodestyle), `F` (pyflakes), `UP` (pyupgrade), `B` (bugbear), `SIM` (simplify, with SIM117 ignored), `I` (isort). Line length **80**.

### Imports

isort enforces the order:

1. `__future__`
2. stdlib (`os`, `sys`, `functools`, `pathlib`)
3. third-party (`numpy`, `torch`, `pypose`)
4. project-local (`bae.*`, `pypose.optim.*`)

Each group separated by a blank line. Example (from `bae_solver.py`):

```python
import os
import sys
from functools import partial

import numpy as np
import pypose as pp
import torch
import torch.nn as nn

from bae.autograd.function import TrackingTensor, map_transform
from bae.autograd.graph import jacobian
from bae.optim import LM
from bae.sparse.py_ops import diagonal_op_
from bae.utils.ba import rotate_quat
from bae.utils.pysolvers import PCG
from pypose.optim.kernel import Huber, Cauchy
```

Run-time imports (`os.environ.setdefault(...)`) before the third-party block are tolerated when there's a documented reason (see `bae_solver.py:12-21`).

### Naming

| Element | Convention |
|---|---|
| Modules / files | `snake_case.py` |
| Classes | `PascalCase` (`ColmapReproj`, `_LoggingPCG`) |
| Functions / methods | `snake_case` (`_run_ba`, `_apply_huber_correction`) |
| Module-private | leading underscore (`_log`, `_env_float`, `_FULL_BA_SOLVE_COUNT`) |
| Constants | `UPPER_SNAKE_CASE` |
| Locals / parameters | `snake_case` |

### Docstrings and comments

- Top-of-file module docstring: short purpose + key context (see `bae_solver.py:1-6`).
- Functions get a one-line docstring when their purpose isn't obvious from the name; longer multi-paragraph docstrings for non-trivial logic.
- Inline comments explain **why** the code does something, especially for workarounds. Reference `info.md §X.Y` when there's a documented reason.

### Strings and formatting

- f-strings (`f"foo {x:.3e}"`) for all formatting. No `%` or `.format`.
- Multi-line strings: implicit concatenation across line breaks, not `+`:
  ```python
  _log(
      "trust-region fixed_rot: "
      f"radius={radius:.3e} max={max_radius:.3e} "
      f"up={up:.3e} down={down:.3e}"
  )
  ```
- Format precision: `:.3e` (3-decimal exponential) is the project default for diagnostics.

### Error handling

- Don't use bare `except:`. `except Exception:` at minimum. Be specific when you can.
- Defensive coding for the embedded-Python boundary: `try: sys.stdout.flush(); except Exception: pass` — silent fallback is OK at known-fragile boundaries (see `_log` in `bae_solver.py`).

### Type hints

This project does **not** consistently use type hints in `bae_solver.py`. Don't add them piecemeal; either annotate a whole function or leave it. New code may use them when they materially clarify shape (e.g., `def f(x: np.ndarray) -> torch.Tensor:`), but don't refactor existing code just to add them.

### Globals and module state

- Module-level mutable state is allowed for solver counters (`_FULL_BA_SOLVE_COUNT`, `_SE3_TANGENT_DIAG_LOGGED`).
- Functions that mutate them must declare `global _NAME` at the top.
- Don't use class-level state where a module-level pattern already exists.

### Bad

```python
def my_function( x,y ):
   if(x==None):return
   else:
     msg="error: "+str(y)
     try: do_something()
     except: pass
     print( "{}".format(msg) )
```

### Good

```python
def my_function(x, y):
    if x is None:
        return
    msg = f"error: {y}"
    try:
        do_something()
    except Exception:
        pass
    print(msg)
```

## Style choices specific to this project

Beyond the formatters, these are habits the codebase uses consistently:

- **Long, narrative comments are encouraged** when documenting a fix or a non-obvious decision. Don't compress them out. See `bundle_adjustment_bae.h:11-17` and `bae_solver.py:12-21` for examples.
- **Cite `info.md §X.Y`** in comments when a piece of code exists to fix a documented investigation outcome.
- **Diagnostic-first logging**: any new BAE-side change should print log lines `[BAE] ...` for the new behavior; that's how we read run logs (see `skills/read-bae-logs.md`).
- **Probe instrumentation lives on both sides of the C++↔Python boundary** (see `rules/05-cross-language-boundary.md`); don't strip a probe without first verifying the boundary equality invariant still holds.

## When in doubt

Match the surrounding file. Don't import a new style from elsewhere; the project is internally consistent.
