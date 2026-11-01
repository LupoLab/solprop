# Roadmap

```{note}
Planned work is tracked in [GitHub issues](https://github.com/LupoLab/solprop/issues) (ADR-062), which can be filtered by type,
priority and area. It is not listed here. This page now holds only the status lists of
accepted and draft SEPs below, and it goes when those lists move.

The entries this page used to hold moved to issues on 2026-11-01. Where an older page or an
archived SEP says that something is "on the roadmap", look in the issues. The
[page as it stood before the move](https://github.com/LupoLab/solprop/blob/11cd808fe03db935457f432f790939d44a243c16/docs/dev/roadmap.md)
shows what it said.
```

Each item here is weighed against the project's priorities (ADR-020 in
[decisions.md](decisions.md)): accuracy and reliability first, then CPU performance, then
GPU. The lists are not a schedule, and none of the proposals is promised. What exists is
described in the user manual; how it is built, in the rest of these developer pages. A
major item, such as a new model, route or numerical scheme or a changed default or
convention, starts as an [enhancement proposal](seps/index.md) (SEP, ADR-035).

## Accepted enhancement proposals

SEPs that have been accepted and are not yet final, each waiting to be picked up or
being implemented.

- [SEP 17 — Broadband two-photon continuum response beside strong-field ionisation](seps/sep-0017-gas-two-photon-ionisation.md):
  prescribed-field diagnostics first, then an Ar/He continuum driven by the squared VUV
  field, with polarisation and optical work from one interaction. The first model gives
  final yield and optical feedback; additional plasma needs a separate electron-timing
  qualification. Existing optmat OEP 6 quantities suffice for this approximation.
  Before production implementation, its qualification covers threshold data and
  completion, reactive matching, atomic validity and the complex convolution's work,
  passivity and window convergence. A coupled optmat OEP is needed only if qualification
  identifies a new material contract. Ne/Xe, mixtures, mode-averaged and multimode
  routes, bound response and other geometries remain follow-up work; the first route is
  a single-mode capillary with node quadrature.
- [SEP 41 — Recipes, and the hooks a front end needs](seps/sep-0041-recipes-and-front-ends.md):
  a versioned TOML recipe that solprop loads, checks, builds, runs, exports as Python,
  records in its output and resumes from (`solprop run`, `solprop resume`), with the
  parameter paths and pins, metadata, structured errors, capability queries for routes,
  species and results, plane observer, scan progress, cost estimate, portable stop,
  previews, mode launch, beam displacement and tilt, design solvers, pulse loaders and
  geometry drawings a front end needs. The Python interface changes are additive and no
  output-schema bump is needed. It lands after SEP 42, whose comparison decides every resume.
- [SEP 42 — Resuming by what matters to the run](seps/sep-0042-tolerant-resume.md):
  the resume gate compares the stored and requested records path by path instead of by
  hash. The layout must agree, physics and launch parameters must agree unless the user
  passes `resume="force"`, numerics such as `rtol` may change on an unfinished run and are
  recorded, and provenance is ignored. Numbers agree per element to a rounding-level bound,
  a new recorded field needs a declared neutral value, and optmat models compare by digest.
  The output schema is bumped only for what an older reader would misread. It answers the
  resume-compatibility policy ADR-058 left open.

## Draft enhancement proposals

SEPs under discussion; their scope may change and implementation has not been accepted.
The [SEP index](seps/index.md) lists every proposal and its current status.

- [SEP 1 — Efficient causal Raman evaluation](seps/sep-0001-raman-response.md):
  exact sampled recurrences, including the current lag taper and silica endpoints,
  beside default exact FFT evaluation. A common execution contract retains analytic
  poles, permits exact FFT evaluation of density-shifted lines and supports SEP 10's
  explicit approximation. Population dynamics are separated into SEP 32.
- [SEP 2 — Two-photon absorption](seps/sep-0002-two-photon-absorption.md):
  constant-coefficient, narrowband scalar absorption on field and envelope grids.
  Broadband and tensor absorption, and solid carrier dynamics, remain outside its scope.
- [SEP 4 — Bounded execution autotuning](seps/sep-0004-execution-autotuning.md):
  explicit CPU policy search on short windows, with isolated processes, numerical
  validation, bounded total cost and reproducible execution plans. Native GPU
  controls, chunking, propagator selection and implicit tuning remain open in [#303](https://github.com/LupoLab/solprop/issues/303).
- [SEP 5 — Solid ionisation and bulk filamentation](seps/sep-0005-solid-ionisation.md):
  scalar bulk silica/YAG photoionisation, explicit simple avalanche and carrier
  removal, collisional currents, optmat solid rate models and parameter bundles
  (which need upstream optmat work first), using SEP 34's shared medium flux and
  radial adiabatic transport.
  Delayed-heating kinetics, subcycle rates and solid-guide ionisation remain
  follow-up work in [#282](https://github.com/LupoLab/solprop/issues/282); source-normalisation questions precede acceptance.
- [SEP 7 — Laser gain dynamics in solid-core fibres](seps/sep-0007-laser-gain-dynamics.md):
  Yb/silica spectroscopy through optmat, conventional steady and time-dependent
  populations, pump and ASE transport, and scalar spatial gain dynamics for nonlinear
  amplifiers. The broadband filtered-field closure has a separate qualification gate;
  general vector gain, other ions, stochastic field ASE and oscillators remain open.
- [SEP 8 — Second-order nonlinearity and birefringent crystal optics](seps/sep-0008-quadratic-crystal-optics.md):
  vector forward propagation in homogeneous crystals, quadratic frequency mixing,
  BBO/LBO material curation, field/envelope grids, interfaces, physical flux and
  a separately qualified statistical OPG model. Guided overlaps, varying crystal
  profiles, resonant THz response and bidirectional propagation remain open.
- [SEP 9 — Broadband beam measurements and propagation statistics](seps/sep-0009-propagation-statistics.md):
  frequency-dependent modal fluence and time-resolved beam widths, spectral filters,
  per-mode durations, transverse peak intensity and distinct quadrature/modal-span
  diagnostics. General-guide embedded quadrature and current-source span diagnostics
  remain open; SEP 34 covers the shared medium-flux correction.
- [SEP 10 — Low-rank Raman kernels along density paths](seps/sep-0010-low-rank-raman.md):
  explicitly requested spectral compression of density-dependent Raman kernels,
  extending SEP 1's contract and exact shifted-line FFT reference. Separate basis and
  coefficient-path budgets, independent-state validation and observable convergence
  bound the approximation; zero tolerance selects exact evaluation.
- [SEP 11 — Finite-density gas index and mixture optics](seps/sep-0011-finite-density-gas-index.md):
  optmat's Lorentz–Lorenz law with refractivity virials, applied once to a mixture,
  and a composition-preserving mixture EOS. One resolved state feeds linear optics,
  Raman amplitudes, widths and shifts. Current upstream model selections, estimates
  and validity records govern admission; nonlinear local-field corrections remain open.
- [SEP 32 — Raman Maxwell–Bloch dynamics](seps/sep-0032-raman-maxwell-bloch.md):
  a separate scalar two-level population/coherence response, with equilibrium
  subtraction, a weak-response limit and local temporal integration. Its ensemble,
  data and integration questions must be settled before acceptance; shared-level
  ladders, vector saturation and repeated-window histories remain future work.
- [SEP 40 — Transverse absorbing boundaries for free-space propagation](seps/sep-0040-transverse-absorbing-boundaries.md):
  the radial, Cartesian, line and planar real-space collar placed outside the requested
  aperture and sized by a declared protected band of transverse wavenumbers rather than
  the run length, in-stage under a collar-crossing step limit (split optional for RK),
  with the k⊥ ramp kept to a numerical safety band (now also radial and planar), a
  monitor that judges returning light, and a signed removal ledger. Perfectly matched
  layers, adaptive transverse windows, the escaped transverse field and a flux-weighted
  spatial ledger remain open.
