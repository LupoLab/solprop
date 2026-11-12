---
orphan: true
---

# SEP 49, Part II draft

The phase reports of SEP 49's implementation, kept here until the SEP's own file reaches
this branch; they then move, unchanged, into its Part II, and this file goes.

## Phase 0 — Baseline

- **Scope.** Phase 0, 2026-11-12, Opus 5.5 (AI), branch
  `feature/sep-0049-tdse-on-native`, based on `feature/two-backends-binary64` at
  `1dc64b5f` (ADR-077's change).
- **Delivered.** This record; no code.
- **Size baseline** (`uv run poe size --all` at `1dc64b5f`, lines):

  | tree | files | total | code |
  |---|---:|---:|---:|
  | `src/solprop` | 269 | 225,295 | 113,232 |
  | `native/src/solprop_native` (Python) | 24 | 13,859 | 8,855 |
  | `tests/` (all `.py`, `wc -l`) | — | 271,204 | — |
  | `native/tests/` (all `.py`, `wc -l`) | — | 13,602 | — |
  | `native/src` C++ (`.cpp`, `.hpp`, `.h`, `wc -l`) | — | 34,575 | — |

  `poe size-check` is within budget. `src/solprop/backends/numpy_ref/tdse.py` is 845
  lines; `src/solprop/driver/tdse.py` 2,575.
- **The host assembly's time per evaluation.** Measured by a one-off script, not
  committed: the `march_problem` problem of `tests/backends/test_tdse_march.py`
  (256 µm helium capillary, 800 nm, one mode), the interaction formulation, a
  manufactured binding returning analytic histories (`J = 1e-30 E + 1e-40 E^3`,
  `j_1 = 1e-30 E`, and a third-order current for the cubic replacement), and
  `NumpyRefBackend._nonlinear(..., purpose="predictor")` timed 60 times, the first ten
  discarded, the binding's own time subtracted. That is steps 1 and 3–5 of §3.2: the
  synthesis to the nodes, the assembly, the projection with every diagnostic, and the
  stage output.

  | lattice | descriptor | `n_nodes` | `n_to` | `n_omega` | median | min |
  |---|---|---:|---:|---:|---:|---:|
  | the test window (`n_r = 4`, 60 fs, 0.4–1.6 µm) | `tdse_native`, gated | 5 | 384 | 97 | 0.26 ms | 0.25 ms |
  | the test window | `optmat_cubic` | 5 | 384 | 97 | 0.13 ms | 0.12 ms |
  | production-sized (`n_r = 16`, 400 fs, 0.1–4 µm) | `tdse_native`, gated | 17 | 9,216 | 2,305 | 1.9 ms | 1.9 ms |
  | production-sized | `optmat_cubic` | 17 | 9,216 | 2,305 | 1.3 ms | 1.2 ms |

  Apple M5 Max, Python 3.14, the machine shared (load average about 10), so the medians
  are upper estimates. §6.8 expected 10–100 ms; the measured assembly is one to two
  milliseconds at a production-sized lattice, below $10^{-5}$ of the fastest recorded
  atomic call (93 s).
- **Deviations.** The SEP's status (`In progress`) and Implementation field are not yet
  set, because the SEP's file is not on this branch (it is on the proposal's branch,
  #765); that commit follows when the file arrives. This draft is named
  `draft-sep-0049-part2.md`, not `sep-0049-part2-draft.md`: `tests/test_seps.py`
  collects every `sep-*.md` file in `docs/dev/seps/` as a SEP and checks its title and
  header.
- **Handoff.** Phase 1 moves the assembly; the same script then times it again.
