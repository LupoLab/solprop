# SEP 49 — The TDSE response on native, and numpy_ref as a slim test reference

| | |
|---|---|
| **SEP** | 49 |
| **Title** | The TDSE response on native, and numpy_ref as a slim test reference |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-12) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Change |
| **Created** | 2026-11-12 |
| **Resolution** | — |
| **Requires** | [SEP 12](archive/sep-0012-tdse-nonlinear-response.md), [SEP 20](archive/sep-0020-hacc-helium-tdse.md) and [SEP 26](archive/sep-0026-edge-free-tdse-source.md) (the TDSE response and its source); [SEP 39](archive/sep-0039-native-adams-march.md) and [SEP 45](archive/sep-0045-native-rk-march.md) (the native marches and their decision specification); [SEP 47](archive/sep-0047-run-ownership.md) (backend-owned sessions); [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) (agreement to the error bound); ADR-077 and the change it calls for (§3.4) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Not started |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

ADR-077 fixes solprop at two backends computing in binary64: `native`, which runs every
production run, and `numpy_ref`, the numerical reference. It also fixes the reference's end
state: a slim library of the arithmetic native is checked against, not a second complete
executor. Today `numpy_ref` *is* a complete executor, about 24k lines with its own step
loops, sessions, groups, lockstep batch, monitors and registry row, and it is the only
backend that runs the TDSE response. This SEP does two things, in order:

1. **Phase 1 — the TDSE response on native.** Native's Adams march evaluates a TDSE
   problem's source through a *host source hook*: one Python callable, bound by the native
   session, which the compiled march invokes holding the GIL at each source evaluation of a
   trial. The hook runs solprop's existing host TDSE path (synthesis to the radial nodes,
   the provider call, source assembly and projection, unchanged) and writes the projected
   spectral source into a native buffer. The provider runs in a worker process of its own
   (§6.8), so its runtime never shares solprop's process and may use a GPU while solprop
   runs on the CPU. The hook is refused on native device builds. It needs an ADR
   exception to the rule "no Python call in a trial step", limited to this hook.
2. **Phase 2 — `numpy_ref` slimmed to a test and development reference.** It keeps the
   contract arithmetic (the linear flow, the nonlinear and source evaluation, the
   Runge–Kutta and exponential-Adams trials, the coupled flow, the statistics) and the
   decision specification (`adams_core`, `rk_core`), exposed as a small API, and reads the
   same `PropagationProblem`. It loses its step loops, sessions, groups, monitors, batch,
   registry row and driver hooks; users can no longer select `backend="numpy_ref"`. The
   full-run parity check becomes native's adaptive run *replayed* on the reference along
   native's own accepted steps and orders, compared step by step within rounding budgets,
   plus native's decision traces checked against the decision specification.

Success means: a TDSE run on native takes the reference's steps and agrees with it within
the rounding bounds of [testing.md](../testing.md) §3, with its atomic ledger, labels,
reservations and decision records identical; and after phase 2, every check that the
deleted executor provided is held by the parity ladder (§7.2), shown by a perturbation that
each removed test caught and the ladder still catches, with about 11.5k fewer source lines
and 12k fewer test lines.

### 2. Motivation and scope

**Why now.** ADR-077 retired single precision, the backend tiers and the Julia backend,
and with them the reference-only methods (CF4 coupled-flow sections, `ComparisonMultistep`,
`PIController` overrides). After that change the TDSE response is the only feature that runs
on `numpy_ref` alone, and it is what keeps the reference's whole executor alive: its two
Runge–Kutta loops, its Adams and low-storage marches, its sessions and group sessions, its
lockstep batch and its registry row. Every one of them is a second implementation of
something native already owns (SEP 47), and every new feature that touches a session, a
monitor, a statistic's channel or a group must be written, tested and documented twice.
The reference's value is elsewhere: in the arithmetic, which is plain, readable and
independent of native's, and against which every native kernel is qualified.

**Who it serves.**

- *TDSE users.* The TDSE run today steps on the unthreaded reference, whose optical side is
  6 to 40 times slower than native's ([Choosing a backend](../../guide/performance/backends.md)),
  and lacks native's session features where they differ. The optical side's share of a TDSE
  run is not measured ([#247](https://github.com/LupoLab/solprop/issues/247)), and an
  atomic call dominates (§6.5), so the gain is mostly uniformity: one executor for every run.
- *Developers.* One executor, one session protocol, one set of marches, and a reference
  that is a library of pure functions over a problem rather than a stateful backend.
- *Physics SEPs.* New physics (SEPs 1, 2, 5, 7, 8, 9, 10, 11, 17, 32) still lands its
  arithmetic in a readable reference first, or beside native, and is qualified against it
  by the same ladder, without writing a second session for it.

**In scope.**

- The TDSE response on the native CPU build, with the reference's admission (§3.2), on the
  native Adams march, adaptive, in both formulations, with its ledgers, labels, metadata,
  memory rows, resume and failures as the reference has them.
- A host source hook in the native Adams engine, its binding in the native adapter and
  session, and the ADR exception that admits it.
- Moving the host TDSE source assembly out of `numpy_ref` into production code, since
  native then depends on it.
- The slim reference: its API (§5.2), its replay of a recorded schedule, its decision-trace
  check, the parity ladder built on them, and the deletion of the executor around it.
- The tests, CI lanes, tools, design pages, manual pages and process text that change.

**Not in scope.**

- **TDSE physics, providers and qualification.** The atomic models, the descriptor, the
  gated source, the calibration and the window are unchanged
  ([#246](https://github.com/LupoLab/solprop/issues/246),
  [#249](https://github.com/LupoLab/solprop/issues/249)). The provider's own cost and
  Pyroraptor's publication stay in [#247](https://github.com/LupoLab/solprop/issues/247)
  and [#248](https://github.com/LupoLab/solprop/issues/248). Changes to Pyroraptor that
  the worker of §6.8 calls for are in scope, as companion changes in its repository.
- **TDSE on device builds of native.** There is no use for one: the atomic calls dominate
  a TDSE run, so the provider takes the GPU (or every CPU core) and solprop's optical side
  runs on one CPU thread (§6.8). A native device build refuses a TDSE problem by name.
- **A compiled TDSE source assembly**, or a C++ link to the provider (§10.3).
- **The batch axis, the route union and the integrator–controller split** of
  [#110](https://github.com/LupoLab/solprop/issues/110), and
  [#618](https://github.com/LupoLab/solprop/issues/618)'s leftover batch axis, beyond what
  deleting the reference's lockstep runners removes.
- **Anything ADR-077's preceding change does** (single precision, CF4, the comparison
  methods, the registry's plugin rows, decoupling native from `numpy_ref`): this SEP
  starts after it (§3.4).

### 3. Context

#### 3.1 The reference today

At commit `78d57c43` the reference is three bodies of code
([backends.md](../backends.md) §1–§2):

| part | lines | what it is |
|---|---|---|
| `src/solprop/backends/numpy_ref/` | 11.2k | the passes (`passes.py`), the general modal route (`general_modal.py`), the envelope plasma (`envelope_plasma.py`), the TDSE source (`tdse.py`), the backend and its seam (`backend.py`, 3.9k), the lockstep batch (`batch.py`), admission, and the single-precision backend that ADR-077's change removes |
| `src/solprop/backends/host/` | 7.4k | `HostDriver` (`driver.py`, 3.2k: the Runge–Kutta loops, the history, the statistics channel, the monitors' calls, noise, resume), `HostAdamsMarch` (`adams.py`), the low-storage march, the comparison methods (`general.py`), `HostSession`, `HostGroup` and `HostGroupSession` |
| reference-only code elsewhere | about 5k | the host Adams engine `HostHistory` and its protocol in `backends/multistep.py`, the host low-storage engine, the `HostStepping` protocol in `backends/protocol.py`, the registry row and the `REMOVED`/`auto` rules for it |

Native runs no Python march. Every native run is its native session's
(`solprop_native.session`), whose compiled marches advance to an event from one policy
record (SEP 47; [kernel-contract.md](../kernel-contract.md) §11–§12). `HostAdamsMarch` and
`HostDriver._march_to` are the *references* for the compiled marches' decisions, through
`solprop.backends.adams_core` and `solprop.backends.rk_core`, and the decision-trace tests
drive each compiled march a call at a time over a scripted host backend
(`tests/backends/native_trace_marches.py`) and hold it to the reference's decisions on the
same readings.

The parity tests are the main correctness net: native and the reference run the same
problem, on one prescribed schedule at rounding level (class R) or adaptively to an
accuracy envelope (class T), on every route, response and statistic. The protected
`HostDriver` surface is frozen in `tests/backends/test_driver.py` (ADR-068), and CI keeps
a `no-native` lane that installs solprop without its native package and runs the reference
on a few driver modules; `tools/setup_light.py` sets up such a checkout for contributors
without a compiler.

#### 3.2 The TDSE response today

The TDSE response ([TDSE response](../../guide/media/tdse.md); SEPs 12, 20 and 26) is
admitted on one route only, by `PropagationProblem._check_tdse` and
`translate.problem._check_tdse_assembly_route`:

- the **modal route** (`ModalRoute(tdse=TDSESource(...))`), **scalar** (`n_pol = 1`),
  **one mode** (`shape[:2] == (1, 1)`), **resolved** over the fixed radial quadrature (not
  mode-averaged), on the **field grid**;
- the **adaptive `Multistep()`** march: no fixed schedule, no interpolated planes, never the
  Runge–Kutta or low-storage marches;
- the **in-stage collar** (`absorber_scheme="nonlinear"`), **no coupled flow**, **no
  plasma**, **no Raman**, no dispersive Kerr response, no density override, and no
  `modified` noise (input noise drives the atom);
- statistics in `TDSE_UNAVAILABLE_STATISTICS` refused by name.

`backend="auto"` resolves a TDSE request to `numpy_ref` before the registry
([backends.md](../backends.md) §1), and native refuses the route by name
(`solprop_native.backend._refuse_unsupported`). The driver owns the provider binding
(`solprop.driver.tdse`, 3.2k lines: the optional Pyroraptor import, stable row chunks,
the atomic-call budget, the raw archive, consistency checks) and passes it to
`registry.select(..., tdse_binding=...)`. The binding's protocol is
`solprop.translate.tdse.TDSEBinding`: `reserve(logical_calls)` before a trial,
`evaluate(fields, z=, purpose=, density=)` per source evaluation, `record_trial(accepted=)`
after each attempt, `release_reservation()` on every exit path, `restore_ledger` on resume,
and `metadata`, `memory_ledger` and `close`.

Inside the reference one source evaluation is `NumpyRefBackend._nonlinear_tdse`
(`backends/numpy_ref/backend.py`). From the lab-frame modal field at the evaluation
position (the input noise added) it:

1. transforms to time on the fine field grid (`passes.to_time`) and synthesises the real
   field at each radial node (`passes.synthesise`), `(n_nodes, n_to)`;
2. calls the binding: `binding.evaluate(fields, z=z, purpose=purpose, density=density)`,
   which returns the unscaled microscopic histories (`TDSEResult`);
3. assembles the macroscopic sources (`numpy_ref.tdse.atomic_sources`): the remainder
   current $S = J - j_1$ (less the third-order current for a cubic replacement), gated and
   scaled by density and the declared scale, and the polarisation of the descriptor's Kerr
   model (the third-order dipole, the optmat cubic through `passes.nonlinear_polarisation`,
   or none), and for a gated source the edge array $g' S$ (`atomic_edge_source`);
4. projects both onto the mode and crops (`project_atomic_sources`): the fine projection
   with the prefactor, the gated current's edge-free spectrum
   $X - E/(\mathrm{i}\omega)$ on the active bins, and the same-row embedded-quadrature and
   edge diagnostics with their rounding floors;
5. applies the spectral collar, the in-stage absorber term and, in the interaction
   formulation, the inverse propagator (`passes.stage_output`).

The Adams march reserves each trial's calls before the trial (`HostAdamsMarch._afford`:
one startup evaluation when the history is empty, plus one for PEC or two for PECE), labels
each evaluation `startup`, `predictor` or `corrected`, and books every attempt
(`record_trial`) from its step record. The run's metadata carries `atomic_calls`,
`tdse_execution`, `tdse_last_evaluation` and `tdse_source`, and the memory ledger the
binding's `tdse_*` rows. A resume restores the call ledgers
(`restore_call_ledgers(logical, atomic)`).

Manufactured bindings, which return analytic histories with no provider, drive the TDSE
tests in CI (`tests/backends/test_tdse_march.py`, `tests/edge_source_cases.py`); a real
provider is not installable from a public release yet
([#248](https://github.com/LupoLab/solprop/issues/248)).

**The provider.** Pyroraptor is solprop's sibling package, written and maintained by the
same maintainer and not yet published. Its interface, initialisation, thread handling,
exit-time clean-up and packaging can therefore change in step with this SEP, as companion
changes in its own repository. It is built on Kokkos and OpenMP, like solprop-native, and
today the binding imports it into solprop's process. Only the atomic calculation needs to
be fast: everything on solprop's side of an evaluation is below $10^{-3}$ of a call (§6.5).

#### 3.3 What native would have to do

The native Adams march (SEP 39, ADR-060) runs the trial in its resident engine and
evaluates the source natively; its kernel rules forbid "allocation, planning, logging,
Python call or unplanned fence in a trial step" ([backends.md](../backends.md) §3.2,
`native/AGENTS.md`). A TDSE source cannot be evaluated natively without Pyroraptor's
runtime inside solprop-native, and its evaluation is a field-dependent, minutes-long call
that cannot be precomputed. The march calls are already bounded by 256 attempts or 0.1 s of
wall time (kernel contract §11), so with a TDSE trial lasting minutes each call returns to
the session after one attempt anyway.

#### 3.4 Decisions and changes this SEP builds on

- **ADR-077** (two backends, binary64 only; [decisions.md](../decisions.md)), which amends
  ADR-068's "`numpy_ref` stays comprehensive" and states the end state this SEP specifies.
- **The change ADR-077 calls for**, which must be merged before phase 1: single precision,
  CF4 sections, `ComparisonMultistep` and `PIController` overrides removed, the registry a
  fixed table of two rows, and native importing nothing from `numpy_ref` (the general
  modal derivations and the quadrature alarm moved to translation). This SEP's line counts
  assume it.
- **ADR-060 and ADR-069**: the compiled marches, with solprop's Python marches as their
  references. Phase 2 makes the decision specification (`adams_core`, `rk_core`) and the
  reference's trial arithmetic the references instead, and the ADR that records phase 2
  amends both.
- **SEP 25**: independent computations agree to an error bound, never to bits.

#### 3.5 Related issues

From a search of the open issues on 2026-11-12:

- **Answered or closed by this SEP.**
  [#110](https://github.com/LupoLab/solprop/issues/110) (a SEP for the step loops and the
  seam): its step-loop question is answered here, since one set of compiled marches and one
  decision specification remain; the batch axis, the route union and the integrator split
  stay with it. [#104](https://github.com/LupoLab/solprop/issues/104) (the Python Adams
  march's per-step allocations) closes with the march.
  [#247](https://github.com/LupoLab/solprop/issues/247) (the cost of a TDSE run): its
  "compiled coupling" item is phase 1, and phase 0 measures the optical side's share it asks
  for.
- **To resolve before phase 2.** [#270](https://github.com/LupoLab/solprop/issues/270)
  (the Adams decision traces cover only the exponential path): once the Python march is
  gone, the traces are the only check of the fused formulation's decisions.
- **Changed by this SEP.** [#618](https://github.com/LupoLab/solprop/issues/618) (the
  leading batch axis): the reference's lockstep runners, which read `max_batch`, are
  deleted. [#171](https://github.com/LupoLab/solprop/issues/171) (one diverging point fails
  a `Batch` group) and [#623](https://github.com/LupoLab/solprop/issues/623) (group
  construction is not transactional): the reference's group paths go, and what remains is
  the native group session's behaviour. [#230](https://github.com/LupoLab/solprop/issues/230)
  and [#155](https://github.com/LupoLab/solprop/issues/155) (native against the reference
  outside the active band, and at a Nyquist bin) become ladder tests (§7.2).
  [#107](https://github.com/LupoLab/solprop/issues/107) (one host statistics module): the
  reference's statistics become a library function of the slim API.
  [#109](https://github.com/LupoLab/solprop/issues/109) (split `backends/multistep.py`): the
  host engine and the march leave it. [#570](https://github.com/LupoLab/solprop/issues/570)
  and [#312](https://github.com/LupoLab/solprop/issues/312) (development wheels that need
  glibc 2.35 and a system FFTW; no source build offline): a contributor without a compiler
  then depends on fetched or published wheels alone (§9.3).
- **Unchanged, for the record.** [#246](https://github.com/LupoLab/solprop/issues/246),
  [#248](https://github.com/LupoLab/solprop/issues/248),
  [#249](https://github.com/LupoLab/solprop/issues/249) and
  [#583](https://github.com/LupoLab/solprop/issues/583) (TDSE qualification, provider
  publication, certificate and cache).

### 4. Theory

The SEP changes no equation. What needs stating is what the hook computes, which
comparisons phase 2 relies on, and what bounds them.

**The source the hook returns.** Let $a(z)$ be the lab-frame modal state at an evaluation
position $z$ (one mode, one polarisation, $n_\omega$ bins, the input noise added). The hook
returns the projected spectral source before the spectral collar,

$$
F(a, z) = p(z)\,\mathcal{F}\big[Q\,P(t)\big] + c(z)\,\Big(\mathcal{F}\big[Q\,g\,S(t)\big]
- \frac{\mathcal{F}\big[Q\,g' S(t)\big]}{\mathrm{i}\omega}\Big)\Big|_{\text{active}},
$$

where $P$ and $S = J - j_1$ are the macroscopic polarisation and remainder current at the
$n_{\text{nodes}}$ radial nodes (density and the declared scales applied once), $Q$ the
modal projection with the taper scale, $g$ and $g'$ the gate and its analytic derivative
(SEP 26), $p$ the stage prefactor and $c$ the independent current factor
([kernel-contract.md](../kernel-contract.md) §6). Native then forms what it forms for every
source: the stage output $w\,F$ with the spectral collar $w$, the in-stage absorber term,
and in the interaction formulation the inverse propagator $e^{-\Delta\Phi}$. That is
exactly the reference's composition in step 5 of §3.2, split at one point.

**Why the split costs nothing in accuracy.** Everything up to $F$ is the same Python code
on the same inputs whichever backend calls it, so the hook's $F$ equals the reference's to
the bit for the same $a$ and $z$ (class P). The remaining operations (a product with the
collar, a sum with the absorber term and a phase factor) are a few roundings per bin, so a
TDSE stage on native agrees with the reference's to the one-stage bound already used for
the other sources, $16\varepsilon$ peak-normalised ([accuracy.md](../accuracy.md) §7).

**Replay along a recorded schedule (phase 2).** A native adaptive run records, per attempt,
$(z_k, h_k, \lVert e_k\rVert, \text{accepted}_k, q_k)$, and its restarts. Replaying the
accepted attempts on the reference, with each step's order $q_k$ and the same startup and
restart history, is a *matched discrete computation* (class R): both sides evaluate the same
method on the same steps from the same inputs. After $n$ steps the states differ by at most
the accumulated one-step rounding, amplified by the flow, which `tests/bounds.py` already
bounds (`step_budget` and `propagation_growth`); the comparison is anchored per step where
native exports its state (§6.7). An adaptive run's *own* result is not an oracle
([testing.md](../testing.md) §3, *Replays*), and the replay does not make it one: it
checks every step native took, not the steps native should have taken.

**Decisions.** What the replay cannot see (whether native chose its steps, orders,
rejections and restarts correctly) is checked separately: each attempt row's norm, fed to
the decision specification (`adams_core` for the Adams march, `rk_core` for Runge–Kutta)
with the march's policy and state, must reproduce native's decision and next proposal
exactly. Both sides compute the decisions without contraction and with the same elementary
functions (kernel contract §11, *Rounding*), so this comparison is exact (class P), as the
decision-trace tests are today.

### 5. Proposed interface

#### 5.1 Users

A TDSE run is written as today, with no backend named:

```text
run = prop_capillary(..., tdse=response, tdse_execution=TDSEExecution(max_atomic_calls=400))
# backend="auto" -> "native" (its CPU build), where it was "numpy_ref"
```

- `backend="auto"` resolves a TDSE request to `native`, like every other request. The
  rule that sends it to `numpy_ref` goes.
- On a device build of native a TDSE problem is refused by name before anything is
  allocated: "the TDSE response runs on the native CPU build; this build is CUDA/HIP".
- After phase 2, `backend="numpy_ref"` raises `BackendError` naming `native`, as
  `numba_cpu` does; a recorded name in old output and snapshots still loads. The GUI's and
  recipes' backend choices lose the option.
- Results, metadata keys (`atomic_calls`, `tdse_execution`, `tdse_last_evaluation`,
  `tdse_source`), budgets, warnings and errors are unchanged. `RunMetadata.backend` names
  `native`. The output schema does not change; the backend is outside the resume key, so a
  TDSE run started on `numpy_ref` resumes on native (class T).

#### 5.2 The slim reference (phase 2; developers and tests)

`solprop.backends.numpy_ref` becomes a library of functions over a `PropagationProblem`.
It has no state beyond what a call is given, and no registry row. Sketch:

```text
from solprop.backends import numpy_ref as ref

ref.linear(problem, state, z0, z1)              # exp(Phi(z1) - Phi(z0)) a, with a coupled flow's action
ref.source(problem, lab, z, *, binding=None)    # F(a, z): every response, every route, TDSE via a binding
ref.rk_trial(problem, controller, state, z, dz) # one IP-DOPRI5 trial: new state, scaled norm, stages
ref.adams_trial(problem, method, history, z, dz, order)  # one exponential or fused Adams trial:
                                                # predicted/corrected state, defects, history row;
                                                # compensation and the carrier frame included
ref.replay(problem, request, trace)             # the accepted steps of a recorded run, in order:
                                                # yields (z, state) per step and the plane statistics
ref.stats(problem, state, z, spec, *, grid, modes, profiles)
ref.check_decisions(trace, policy)              # every attempt re-decided by adams_core / rk_core
```

- `trace` is a `StepTrace`: the attempt rows a native march already returns (end, step,
  norm, accepted, order), with its restarts, startup and landings, collected by the native
  session when a test asks for it.
- `adams_core` and `rk_core` keep their names and modules (`solprop.backends.adams_core`,
  `solprop.backends.rk_core`): they are the decision specification.
- The functions are importable but carry no stability promise; they are documented in the
  developer pages, not the user manual (§12, Q5).
- What is refused is refused by translation (`validate_run`), once, before either side is
  called; the library adds no admission of its own.

### 6. Implementation strategies

#### 6.1 Layers

- **Translation** is unchanged, except that `auto`'s TDSE rule and the reference's
  admission go (phase 1 and 2).
- **Host TDSE source** (production): `numpy_ref/tdse.py` and `_nonlinear_tdse`'s
  synthesis and projection move to a production module beside the other host execution
  code (proposed `solprop.backends.tdse_source`; §12, Q3), because native depends on them.
  The move is a pure move; the reference's `source()` calls the same module.
- **Native**: a host source hook in the resident Adams engine (C++), its binding
  (nanobind), and the adapter's hook object and session wiring (Python).
- **Reference**: the slim library (§5.2), built from the existing passes and trial code.

#### 6.2 The host source hook (phase 1)

**C++.** The resident Adams engine gains one optional member, a type-erased callable
`HostSource` (a `std::function` of the input view, the output view, the evaluation
position, the purpose label and the trial's call count). When it is set, the engine's
source evaluation for the startup, predictor and corrector calls copies nothing: it hands
the hook the lab-frame field it would hand its own source pass (on the CPU build a host
view of the state, the input noise added) and an output view allocated once at
construction and listed in the memory ledger (`tdse_host_source`, two `n_omega` complex
rows). It then runs its usual stage output on that buffer. The engine's own response pass
is not run on a TDSE problem: the hook's source is the whole nonlinear term, the optmat
cubic included when the descriptor asks for it, as on the reference. The engine's
translation unit sees no Python; the binding layer wraps the Python callable, acquiring
the GIL (`nb::gil_scoped_acquire`) for the call.

**The three entry points.** The hook object mirrors the reference's binding calls at the
same points of an attempt:

| entry point | when | does |
|---|---|---|
| `begin(cost)` | after the march's budget check, before an attempt's first evaluation | `binding.reserve(cost)`, with `cost` computed by the march as `_afford` does (§3.2) |
| `evaluate(lab, z, purpose, out)` | each source evaluation: `startup`, `predictor`, `corrected` | the host TDSE source (§3.2 steps 1–4), written into `out` |
| `end(accepted)` | after the decision, rejected attempts included | `binding.record_trial(accepted=...)`, then `release_reservation()` |

All three are per trial or per evaluation; none runs per stage of a non-TDSE problem.

**Failures.** A Python exception in any entry point (a provider failure, a budget refusal,
`KeyboardInterrupt`) is caught at the binding layer as `nb::python_error` and stored; the
engine abandons the attempt uncommitted and the march returns with a host-source reason
carrying the stage (startup, trial, accept). The session releases the reservation and
re-raises the stored exception unchanged, so a `ResponseBudgetError` still carries the last
accepted position, field and calls. The state stays at the last accepted step, as on the
reference.

**Device builds.** A device build refuses a TDSE problem by name at preparation, and the
hook is compiled out of device kernels. Nothing is lost: the GPU belongs to the provider
(§6.8).

**Session.** `NativeSession` constructs the hook from the driver's binding when the problem
carries a TDSE route, binds it before the march starts and unbinds it on close; it records
`atomic_calls`, `tdse_execution`, `tdse_last_evaluation` and `tdse_source` as the reference
does, adds the binding's `tdse_*` memory rows, and restores the call ledgers on resume.
The registry passes `tdse_binding` to native's factory, which today refuses it.

**Threads.** A TDSE run gives the machine to the provider (§6.8): solprop's native
session runs on one thread, and the worker takes the rest of the run's thread budget
(ADR-052) or its GPU.

#### 6.3 Options for evaluating the TDSE source on native

| option | what | for | against |
|---|---|---|---|
| **A. Host callback per evaluation** (recommended) | the hook of §6.2 | the TDSE path stays one Python implementation; about 0.3k C++ and 0.4k adapter lines; native's march, history, monitors and statistics serve TDSE unchanged | an exception to "no Python call in a trial step"; CPU builds only |
| B. Suspend the march at each evaluation | the march returns to the session with a "source needed" event mid-trial and resumes after the host fills the buffer | no Python call inside a trial | splits the hottest native code (the exponential Adams trial) into a resumable state machine at each evaluation, a path every non-TDSE run then carries, for a feature whose overhead does not matter |
| C. Link Pyroraptor into solprop-native | C++ calls into the provider, zero-copy, device to device | no host crossing; could run on a device | Pyroraptor is in-house and could be adapted, but the distributed wheel's build would carry it; both would need one Kokkos version, configuration and ABI, which rules out a GPU provider beside the CPU wheel; the binding's budget, chunks, archive and consistency logic (3.2k lines) and the source assembly and diagnostics (1k lines) transcribed to C++ or still called from Python; no measurable gain (§6.5) |
| D. Precompute the response | tabulate the TDSE response and evaluate it natively | none | impossible: the response depends on the whole field waveform at each node |
| E. Keep the reference's executor for TDSE alone | `auto` keeps sending TDSE to `numpy_ref` | no native work | keeps `HostDriver`, `HostAdamsMarch`, the host engine and `HostSession`, most of the 24k lines, so phase 2 cannot happen |

#### 6.4 The slim reference (phase 2)

**What it keeps**, recast as functions of §5.2:

- `passes.py`, `general_modal.py` and `envelope_plasma.py`: the linear application and the
  nonlinear source on every route, the general modal route and the envelope plasma
  included;
- the Runge–Kutta trial (`_trial_step`'s arithmetic, the absorber, collar and mask
  finishing) and the exponential and fused Adams trials (the trial tables, compensation,
  the carrier frame), from `numpy_ref/backend.py`, `backends/exponential.py` and
  `HostHistory` in `backends/multistep.py`;
- the coupled flow's constant-Schur and frozen-residual actions;
- the statistics (`stats`), including the transverse edge's host reduction and the
  boundary monitor's reading;
- `adams_core` and `rk_core`, unchanged;
- the low-storage step's arithmetic, as one function `low_storage_step`, so that the
  low-storage replay is checked as the other marches are.

**What it deletes:** `backends/host/` (the drivers, marches, sessions and groups),
`NumpyRefBackend`'s seam and state, `numpy_ref/batch.py` (`NumpyBatchBackend`), the
`HostStepping` protocol, the reference's `prepare`/`prepare_group` and session parts that
only it uses, the registry row and the TDSE `auto` rule, the protected `HostDriver`
surface frozen in `tests/backends/test_driver.py`, the `no-native` CI lane and
`tools/setup_light.py` (`poe fetch-wheel` covers development without a compiler).

**Replay.** `replay` runs the trace's accepted steps through `rk_trial` or `adams_trial`
with each step's order, with startup and restarts where the trace has them, and lands on
the run's planes as the trace did. It takes no decision. A trace from an adaptive native run
therefore exercises every native step, rejected attempts excepted, whose norms the decision
check covers.

**Tests built on it.**

- *Parity ladder* per route family: `linear`, `source` (each response), one trial, eight
  steps replayed, each statistic, and a short replayed run, all class R.
- *Decision traces*: `check_decisions` over every native trace the ladder records, plus
  the scripted corpus of `tests/backends/adams_traces.py`, re-based from the host march
  onto `adams_core` and `rk_core` directly.
- *Session bookkeeping* (planes, landings, the statistics channel, monitors, the edge,
  resume, groups) becomes native tests with exact expectations (class P): the expected
  positions, rows and records are written out, not compared with a second executor.
- *Re-pointed modules*: about 78 test modules name `numpy_ref` only as the backend to run
  on (neither native nor the reference's internals appear in them); they run on native with
  their tolerances unchanged.

#### 6.5 Cost

**Time.** One TDSE evaluation is a complete atomic calculation at every radial node:
complete calls on compressed fields took 93 s (production basis, 17 nodes) to 615 s
(strong-field basis, 33 nodes) in [SEP 12](archive/sep-0012-tdse-nonlinear-response.md)'s
node selection (Part II),
and 366 s (haCC(1)) and 1,307 s (haCC(5)) on one AMD MI325X
([SEP 20](archive/sep-0020-hacc-helium-tdse.md), Part II). The hook adds a GIL acquisition
and a Python call, microseconds, and the host source assembly, which the reference already
performs (one transform to time, a synthesis to at most a few tens of nodes, the
projections and their diagnostics; of order 10–100 ms at the response lattices in use, to
be measured in phase 0). That is below $10^{-3}$ of the fastest recorded call. What native
saves is the optical side of each step, whose share of a TDSE run is unmeasured (#247);
phase 0 measures it with a manufactured binding.

**Memory.** Two `n_omega` complex rows for the hook, plus the binding's own rows, as today.

**Code** (estimated; `poe size` before and after, compared at Final):

| | `src/solprop` | native Python | C++ | tests |
|---|---|---|---|---|
| phase 1 | about +0.3k (the worker; the rest a move) | +0.4k | +0.3k | +0.7k |
| phase 2 | about −11.5k | about 0 | 0 | about −12.5k |
| total | about −11.2k | +0.4k | +0.3k | about −11.8k |

**Dependencies.** None new.

#### 6.6 The kernel contract and the frozen tables

The kernel problem does not change: the TDSE descriptor already crosses it, and the hook
is a session binding outside it, as the binding is today. Phase 1 adds the hook's binding to
the members `check_build` looks up by name, and rewrites
[kernel-contract.md](../kernel-contract.md) §6's "the native backend refuses TDSE by name
and never holds a binding". Phase 2 deletes the frozen `HostDriver` tables of
`tests/backends/test_driver.py`, so ADR-068's second crossing goes, and rewrites §11 and
§12's "one reference implementation, solprop's `HostAdamsMarch`" and "`HostDriver._march_to`"
to name the decision specification and the reference's trial functions. Each rewrite adds a
release-note fragment.

#### 6.7 Anchored and free-running replay

Two forms of the step-by-step comparison, both class R:

- **Free-running**: the reference replays the whole trace from the launch and is compared
  with native's state at each step native exports, within `propagation_growth` of the
  accumulated one-step budget. It needs no state from native but the launch.
- **Anchored**: at each exported step the reference starts from native's own state and
  history and takes one step, compared within the one-step bound. It isolates the step that
  departs, and its bound does not grow.

The tests keep to about ten steps (`AGENTS.md`), so native exports its state at every step
of a test run through a test-only session option; both forms run on the ladder's cases.

#### 6.8 The provider runs in a worker process

The hook of §6.2 calls `binding.evaluate` and does not know where the atomic calculation
runs. From phase 1 the binding (`solprop.driver.tdse`) runs Pyroraptor in a worker process
of its own, never in solprop's process:

- **The worker.** The binding starts one long-lived worker per run, which imports
  Pyroraptor and nothing of solprop-native, initialises its Kokkos runtime once (on the CPU
  or on a device, as Pyroraptor's build and the run's settings choose) and serves
  evaluations until the run closes.
- **What crosses.** Each `evaluate` sends the node fields (`n_nodes × n_to` float64, a few
  MB) and receives the histories (`TDSEResult`) through shared memory, with a small
  control message for `z`, `purpose` and `density`. The reservation, the ledgers, the
  archive, the consistency checks and the source assembly stay in solprop's process; only
  the atomic call crosses.
- **Failures.** An exception in the worker returns as the provider failure it is today. A
  worker that dies, or does not answer within the run's limit, ends the attempt as a
  provider failure with the state at the last accepted step (§6.2); `close` stops the
  worker on every exit path.
- **Threads and devices.** The atomic calls dominate a TDSE run, so the provider gets the
  machine. By default solprop's native session runs a TDSE problem on one thread (its
  optical side is negligible), and the worker gets every other core of the run's thread
  budget (ADR-052), or, with a Pyroraptor device build, the whole GPU. A native thread
  pool that is already wider, started earlier in the process, sleeps while the hook waits
  (ADR-056). The two never share a device.

**Why a worker.** Only the atomic calculation needs to be fast (§3.2, §6.5). A copy each
way and a context switch cost milliseconds against calls of minutes. In exchange:

- two Kokkos and OpenMP runtimes never share an address space, so their initialisation,
  finalisation, symbols, thread pools and exit-time FFTW clean-up cannot interact, and
  need no qualification or care in either package;
- a Pyroraptor device build runs beside the native CPU wheel, the usual arrangement for
  real runs (solprop on one CPU thread, the atomic calls on the whole GPU), with no common
  Kokkos configuration;
- either package can change its Kokkos version, backend or build without the other.

The cost is about 0.3k lines in `solprop.driver.tdse` and its tests, and one process start
per run. Running Pyroraptor in solprop's process is not supported, and today's in-process
import goes with phase 1.

### 7. Verification and validation

#### 7.1 Phase 1: TDSE on native

Against the reference, which still executes TDSE in phase 1, with the manufactured
bindings in CI:

1. **One evaluation** (class P then R): the hook's $F$ against the reference's for the
   same lab field and position, bit for bit; the stage output against
   `_nonlinear_tdse`'s within $16\varepsilon$ peak-normalised, on gated and ungated
   descriptors, each Kerr model, with and without input noise.
2. **The call sequence** (class P): the exact sequence of `reserve`, `evaluate(purpose,
   z)`, `record_trial` and `release_reservation` calls on a recording binding, for PEC and
   PECE, a rejection, a restart, a budget refusal mid-trial, a provider failure and a
   resume; the ledgers and metadata equal to the reference's.
3. **Eight adaptive steps** (as SEP 39's native-against-reference Adams tests): the same
   accepted steps and orders, and the fields within the eight-step budget ($128\varepsilon$);
   a fork at a decision threshold is reported, not loosened (class T then applies).
4. **The existing TDSE tests** (`tests/backends/test_tdse_march.py`,
   `tests/driver/test_tdse_*.py`, `tests/driver/test_sep12_sep14_integration.py`) run on
   native with their tolerances unchanged.
5. **Refusals**: a device build and every route outside §3.2, by name.
6. **With a real provider, by hand** (not in CI, #248): one short segment of at most three
   atomic calls on the smallest preset, native against the reference (class T), through
   the worker, once with Pyroraptor on the CPU and once on a GPU; recorded in the phase
   report.
7. **The worker** (§6.8): the call-sequence and one-evaluation tests of items 1–2 with the
   manufactured binding behind the worker, bit for bit against calling it directly; a
   worker that raises, dies or hangs mid-trial ends as a provider failure with the state at
   the last accepted step; `close` leaves no worker behind.

#### 7.2 Phase 2: the slim reference

1. **The ladder** of §6.4 on every route family the parity tests cover today, with bounds
   fixed in advance from `tests/bounds.py`.
2. **Decision traces**: every trace the ladder records, exact, and the scripted corpus,
   with #270's fused-path scenarios.
3. **Before deletion**: in phase 2a the slim library and the old executor coexist, and the
   library's `replay` agrees with `HostDriver`'s and `HostAdamsMarch`'s replay of the same
   schedule within rounding, on every ladder case. This shows that the library is the same
   reference, before the executor goes.
4. **Nothing loosened**: per module, the counts of collected tests and of `assert` and
   `match=` sites before and after, each drop mapped to a deleted mechanism or a merged
   table; no tolerance in `tests/` or `tests/bounds.py` changes. For each full-run parity
   test removed, a deliberate perturbation of the corresponding arithmetic (a dropped term,
   a flipped sign, a wrong stage weight) is shown to fail the ladder ([testing.md](../testing.md)
   §3, *Loosened bounds*).
5. **Coverage** stays at 100 % of statements and branches; the reference's functions are
   covered by the ladder, not exempted.

### 8. Documentation

- **Manual.** [TDSE response](../../guide/media/tdse.md): runs on native's CPU build;
  [Choosing a backend](../../guide/performance/backends.md): one backend for runs, the
  reference's role, the list of features that run only on the reference removed; the
  tutorials and examples that name `numpy_ref` (for example
  `docs/tutorials/basics/plot_01_package_overview.py`) revised.
- **Design pages.** [backends.md](../backends.md) §1, §2, §9–§11 (the seam without
  `HostStepping`, the reference as a library, what runs where);
  [kernel-contract.md](../kernel-contract.md) §6, §11, §12; `architecture.md`;
  [testing.md](../testing.md) §2–§3 (backends, replays, the ladder); `tooling.md` and
  `contributing.md` (development without a compiler); `AGENTS.md` and `native/AGENTS.md`
  (the hook's exception, the backend row).
- **Records.** One ADR in phase 1 (the host source hook: the one Python call in a native
  trial, its scope and its refusal on device builds) and one in phase 2 (the reference as a
  test library; parity as ladder, replay and decision traces; amending ADR-060, ADR-068,
  ADR-069 and ADR-077). [SEP 0](sep-0000-process.md) §6–§7 rewritten so that a feature
  lands on native with its arithmetic in the reference library and is `Final` with the
  ladder, with a line in its history (§12, Q8).
- **Release notes** for the `auto` change (phase 1) and the removal of
  `backend="numpy_ref"` (phase 2).

### 9. Compatibility and impact

#### 9.1 Users

- `backend="numpy_ref"` stops working after phase 2 (ADR-077 allows the break; nothing
  public depends on it beyond scripts, recipes and the GUI's choice). A recorded name still
  loads.
- TDSE runs need the native CPU build; a GPU-only environment cannot run one (Q2).
- No physics default, output schema or resume key changes.

#### 9.2 Environments without native

Today `numpy_ref` by name runs where native does not: a platform with no wheel (Intel
macOS, ppc64le, Arm below Neoverse N1), and in process beside a GPU build, for a problem the
device build refuses (a z table, the carrier frame). After phase 2 neither exists: solprop
runs only where a native build runs, and a problem a device build refuses needs a second
environment with the CPU wheel ([backends.md](../backends.md) §11). This follows from
ADR-077 (native is required); it is stated here so that it is decided, not discovered (Q4).

#### 9.3 Contributors

`tools/setup_light.py` and the `no-native` lane go. A contributor without a compiler runs
`poe fetch-wheel`, which installs CI's wheel for the checkout, or the published wheel. CI's
Linux wheel needs glibc 2.35 and a system FFTW (#570), and a source build needs the network
(#312); the documentation states both.

#### 9.4 Risks

- **R1. The worker's transport.** Shared memory and process start-up differ by platform
  (Linux, macOS, Windows). Mitigation: phase 0 runs a real provider through the worker on
  each CPU platform and one GPU host; the manufactured bindings run behind the worker in CI
  (§7.1 item 7).
- **R2. A full-run oracle is lost.** The replay checks the steps native took, not its
  choice of steps. A controller defect that computes every step correctly is caught only by
  the decision traces, so their coverage must be complete before deletion (#270, and every
  march's options in the trace corpus).
- **R3. The reference drifts.** A library no run executes can rot. Mitigation: it is
  exercised by the default lane, held at 100 % coverage, and every new feature's ladder
  tests call it.
- **R4. Python exceptions through C++.** A failure inside the hook crosses the native
  march. Mitigation: the exception is caught at the binding layer, never inside a Kokkos
  kernel; the failure tests of §7.1 cover every entry point and exit path.
- **R5. Lost coverage in deletion.** Mitigation: §7.2 items 3–5.
- **R6. Contributors without a compiler** depend on wheels (§9.3).

### 10. Alternatives considered

#### 10.1 Keep `numpy_ref` a complete executor, decoupled from native

The interim state ADR-077 allows: the reference keeps its loops, sessions, groups, batch
and registry row, and native stops importing it. Rejected for cost: it keeps about 24k
lines whose only purpose is to execute runs nobody runs on it, and every feature touching
a session, a monitor, a statistic's channel or a group is written, tested and documented
twice. Its one unique capability, the TDSE response, moves to native in phase 1.

#### 10.2 Delete `numpy_ref` entirely

Native would then be checked against analytic solutions, invariants and stored results
alone. Rejected: there would be no independent, readable implementation against which new
physics is developed and qualified at rounding level. The open SEPs that add physics, SEPs
[1](sep-0001-raman-response.md), [2](sep-0002-two-photon-absorption.md),
[5](sep-0005-solid-ionisation.md), [7](sep-0007-laser-gain-dynamics.md),
[8](sep-0008-quadratic-crystal-optics.md), [9](sep-0009-propagation-statistics.md),
[10](sep-0010-low-rank-raman.md), [11](sep-0011-finite-density-gas-index.md),
[17](sep-0017-gas-two-photon-ionisation.md) and [32](sep-0032-raman-maxwell-bloch.md), each
need one, and stored results only catch a change, not an error present from the start.

#### 10.3 TDSE options

§6.3: suspending the march at each evaluation, linking the provider into native,
precomputing the response, or keeping the reference executor for TDSE alone.

#### 10.4 The reference under `tests/`

Moving the slim reference out of the package into the test tree would keep it out of
installs. Rejected: physics SEPs develop their arithmetic in it before native has it, the
design pages cite it, and in the package it keeps the lint, type, docstring and coverage
rules. Its size after phase 2 makes shipping it cheap.

### 11. Phases

The phases follow SEP 0's one branch and one pull request, unless the maintainers allow a
milestone pull request per phase as for SEP 47 (Q9).

| phase | deliverables | verifies |
|---|---|---|
| **0. Prerequisites and baseline** | ADR-077's change merged; #270 resolved; `poe size` baseline; the optical side's share of a TDSE run on the reference and the host assembly's time, with a manufactured binding on the short TDSE test window (#247); the worker's transport tried with a real provider on each CPU platform and one GPU host (R1) | the baseline numbers recorded in the phase report |
| **1. TDSE on native** | the host TDSE source moved out of `numpy_ref` (a pure move); the provider worker (§6.8) in the binding; the C++ hook, its binding and the adapter's hook object; session wiring, metadata, memory rows, resume; `auto` sends TDSE to native; device builds refuse; the ADR; the manual and design pages of §8 for TDSE | §7.1 |
| **2a. The slim reference beside the executor** | the API of §5.2, `StepTrace` and the session's test-only state export, the ladder, the decision-trace re-basing; the comparison of the library's replay with the old executor's | §7.2 items 1–3 |
| **2b. Delete the executor** | the deletions of §6.4; the ~78 re-pointed modules; session bookkeeping tests on native; `backend="numpy_ref"` refused; the registry, GUI and recipes; the ADR, SEP 0, the design and manual pages of §8 | §7.2 items 4–5; the size report against §6.5 |

### 12. Open questions

1. **Q1. The hook's shape.** Three entry points (`begin`, `evaluate`, `end`) at the
   reference's points (recommended), or `evaluate` alone, with the march's call bound set to
   one attempt for a TDSE run so that the session reserves and books each trial between
   calls. The second keeps the exception to one entry point but moves the reservation's
   cost rule (`_afford`) to the session.
2. **Q2. Device builds.** *Decided:* a native device build refuses a TDSE problem by
   name. There is no use for one: the provider takes the GPU or all the CPU cores, and
   solprop runs on one thread (§6.8).
3. **Q3. Where the host TDSE source lives**: `solprop.backends.tdse_source` (proposed),
   beside the binding in `solprop.driver.tdse`, or in translation. It must not be under
   `numpy_ref`.
4. **Q4. Losing the in-process reference executor** on platforms without a wheel and
   beside a GPU build (§9.2). Recommended: accept, as ADR-077's consequence.
5. **Q5. Is the slim API public?** Recommended: importable, documented in the developer
   pages, with no stability promise and not in the user manual.
6. **Q6. The replay's form**: free-running, anchored, or both (§6.7; recommended both on
   the ladder's short cases).
7. **Q7. The module's name**: keep `solprop.backends.numpy_ref` (recommended, since every
   design page, SEP and test names it) or rename it `solprop.reference`.
8. **Q8. SEP 0's rules.** Whether the change to SEP 0 §6–§7 (what a feature lands on, and
   what "parity" means at `Final`) rides with this SEP's phase 2 (recommended, since ADR-077
   decided the end state) or needs its own Process SEP.
9. **Q9. Delivery.** One pull request for the SEP, or a milestone pull request per phase
   (phase 1 is useful alone and smaller to review), under an exception like SEP 47's.
10. **Q10. A real provider in CI.** Pyroraptor is in-house, so CI could install it before
    it is published (#248), from a private wheel or a deploy key on its repository, and run
    §7.1 item 6's CPU half on the smallest preset in a scheduled job. Recommended: do so in
    phase 1, with the worker; the GPU half stays a recorded hand check.
11. **Q11. The worker's transport.** Shared memory with a pipe for control (recommended),
    or a pipe alone, whose extra copy is also negligible. A worker on another host (the
    atomic calls on a cluster's GPU node, solprop on a workstation) would use the same
    protocol over a socket; it is not part of this SEP.

### 13. References

1. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM, 2002).
   doi:10.1137/1.9780898718027
2. E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential Equations I:
   Nonstiff Problems*, 2nd ed. (Springer, 1993). doi:10.1007/978-3-540-78862-1
3. M. Hochbruck and A. Ostermann, "Exponential integrators," *Acta Numerica* **19**,
   209–286 (2010). doi:10.1017/S0962492910000048
4. nanobind documentation, "GIL management" and "Exceptions",
   <https://nanobind.readthedocs.io/en/latest/>

## Part II — Implementation record

## Part III — History

- 2026-11-12: Drafted.
- 2026-11-12: Tracking issue [#764](https://github.com/LupoLab/solprop/issues/764).
- 2026-11-12: The provider's placement (§6.8, Q11) added after the maintainer noted that
  Pyroraptor is in-house and unpublished, so it can change with solprop, and that only the
  atomic calculation needs to be fast; R1, option C, phase 0, §7.1 and Q10 revised to match.
- 2026-11-12: The maintainer chose the worker process outright: no in-process option, since
  a GPU provider beside the CPU wheel is a likely arrangement and a few milliseconds per
  call are negligible. §6.8, R1, §7.1, phase 0 and Q11 rewritten.
- 2026-11-12: Q2 decided by the maintainer: no TDSE on native device builds and no shared
  GPU. A TDSE run gives the provider the GPU or every CPU thread, and solprop's optical
  side runs on one thread. §2, §6.2 and §6.8 revised to match.
