# SEP 49 — The TDSE response on native

| | |
|---|---|
| **SEP** | 49 |
| **Title** | The TDSE response on native |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-12) |
| **Status** | Accepted |
| **Review status** | Reviewed |
| **Reviewers** | GPT-6 (AI); John Travers (human) |
| **Type** | Change |
| **Created** | 2026-11-12 |
| **Resolution** | Accepted 2026-11-12 by John Travers, in [#765](https://github.com/LupoLab/solprop/pull/765) |
| **Requires** | [SEP 12](archive/sep-0012-tdse-nonlinear-response.md), [SEP 20](archive/sep-0020-hacc-helium-tdse.md) and [SEP 26](archive/sep-0026-edge-free-tdse-source.md) (the TDSE response and its source); [SEP 39](archive/sep-0039-native-adams-march.md) (the native Adams march); [SEP 47](archive/sep-0047-run-ownership.md) (backend-owned sessions); [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) (agreement to the error bound); ADR-077 and the change it calls for (§3.4) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Not started |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

ADR-077 fixes solprop at two backends computing in binary64: `native`, which runs every
production run, and `numpy_ref`, the numerical reference. Native implements every feature
except the TDSE response, which runs on `numpy_ref` alone, with the atomic provider
(Pyroraptor) imported into solprop's own process. This SEP brings the TDSE response to
native's CPU build and moves the provider out of solprop's process, for both backends:

- **A host source hook.** Native's Adams march evaluates a TDSE problem's source through
  three Python callbacks bound by the native session, `begin(cost)` before an attempt,
  `evaluate(...)` at each source evaluation and `end(accepted)` after its decision. They
  are the one Python call admitted inside a native trial, by an ADR exception limited to
  them. Native device builds refuse a TDSE problem.
- **A shared host source.** The assembly of the macroscopic source from the provider's
  histories (subtraction, scaling, gating, transforms, projection) moves out of
  `numpy_ref` into `solprop.backends.tdse_source`, which both backends call; native never
  imports `numpy_ref`.
- **A provider worker.** The driver's TDSE binding runs Pyroraptor in a worker process
  of its own, started fresh, talking over a pipe, and cancellable at any moment. Neither
  backend imports Pyroraptor into solprop's process, and the provider can use a GPU while
  solprop runs on the CPU.

`numpy_ref` stays a complete, selectable backend, its sessions, groups and batch
included. Slimming it to a test library, which this SEP first proposed, is deferred
(§10.1).

Success means: a TDSE run on native agrees with the same run on `numpy_ref` within the
rounding budgets of [testing.md](../testing.md) §3 derived for its cases, with its atomic
ledger, labels, reservations and archive records identical; the shared source meets
independent analytic and manufactured expectations; the worker's contract holds in CI
with a manufactured provider; and an explicit stop ends a run within the wait loop's
poll interval and the worker's bounded shutdown, whatever the provider is doing, with the
last accepted step checkpointable.

### 2. Motivation and scope

**Why.** After ADR-077 the TDSE response is the one feature native does not run.
`backend="auto"` sends a TDSE request to `numpy_ref`, native carries refusals for it, and
a TDSE run lacks native's session features where they differ and steps its optical side
on the unthreaded reference. Separately, the provider shares solprop's process today: two
Kokkos and OpenMP runtimes (Pyroraptor's and, once imported, solprop-native's) in one
address space, and no way to run a GPU provider beside the native CPU wheel, which is the
usual arrangement for real runs.

**Who it serves.**

- *TDSE users.* One executor for every run: `auto` picks native for every problem, and
  TDSE runs gain native's session features. Ctrl-C, SIGTERM and a stop button end a run
  promptly even during an atomic call of many minutes. The optical side's share of a TDSE
  run is negligible beside the atomic calls (§6.8), so speed is not the reason: native is
  the executor every production run uses, and the one most exercised and most reliable,
  so TDSE runs should use it too (the maintainer's decision, §12 P4).
- *Developers.* No runtime interaction between Pyroraptor and solprop-native to qualify
  or work around, and one host source implementation with independent tests.

The worker serves `numpy_ref`'s TDSE runs from the moment it lands, independently of the
native hook. Neither the hook nor the worker is specific to TDSE or Pyroraptor (§6.2,
§6.5, §12 P10): TDSE is their first user.

**In scope.**

- The TDSE response on the native CPU build, with the existing admission (§3.2), on the
  native Adams march, adaptive, in both formulations, with its ledgers, labels, metadata,
  memory rows, resume and failures as `numpy_ref` has them.
- The host source hook in the native Adams engine, its binding and session wiring, and
  the ADR exception that admits it.
- The shared host source module, `solprop.backends.tdse_source`.
- The provider worker for every use of Pyroraptor: the run's binding, the preset and
  basis queries, and diagnostic sessions.
- Explicit cancellation during an atomic call, and the failure boundary at the numerical
  commit, on both backends.
- The tests, design pages, manual pages and records that change, and the companion
  changes in Pyroraptor's repository.

**Not in scope.**

- **Any reduction of `numpy_ref`** (§10.1). Its executor, sessions, groups, batch,
  registry row, `no-native` lane and `tools/setup_light.py` are unchanged.
- **TDSE physics, providers and qualification.** The atomic models, the descriptor, the
  gated source, the calibration and the window are unchanged
  ([#246](https://github.com/LupoLab/solprop/issues/246),
  [#249](https://github.com/LupoLab/solprop/issues/249)); the provider's cost and its
  publication stay in [#247](https://github.com/LupoLab/solprop/issues/247) and
  [#248](https://github.com/LupoLab/solprop/issues/248).
- **TDSE on native device builds** (§12, Q2), **a compiled TDSE source assembly** or a C++
  link to the provider (§6.4).
- **A shared-memory transport** and an automatic time limit on an atomic call (§10.2),
  and a worker on another host.

### 3. Context

#### 3.1 The two backends today

`numpy_ref` is a complete executor: its own Runge–Kutta, Adams and low-storage marches
(`backends/host/`), sessions, group sessions, a lockstep batch and a registry row
([backends.md](../backends.md) §1–§2). Native runs every run through its native session
(`solprop_native.session`), whose compiled marches advance to an event from one policy
record (SEP 47; [kernel-contract.md](../kernel-contract.md) §11–§12). Each `advance` call
returns after 256 attempts or 0.1 s of wall time; the session wraps it in
`_interrupts.committing()`, so a SIGINT or SIGTERM arriving during the call is held and
raised when the call returns. The run's stop flag is polled before each native call, and
a `StopToken` also forwards a stop to the native march, which reads it between attempts.

#### 3.2 The TDSE response today

The TDSE response ([TDSE response](../../guide/media/tdse.md); SEPs 12, 20 and 26) is
admitted on one route only, by `PropagationProblem._check_tdse` and
`translate.problem._check_tdse_assembly_route`:

- the **modal route** (`ModalRoute(tdse=TDSESource(...))`), **scalar** (`n_pol = 1`),
  **one mode**, **resolved** over the fixed radial quadrature, on the **field grid**;
- the **adaptive `Multistep()`** march only;
- the **in-stage collar**, **no coupled flow**, **no plasma**, **no Raman**, no dispersive
  Kerr response, no density override, and no `modified` noise;
- statistics in `TDSE_UNAVAILABLE_STATISTICS` refused by name.

`backend="auto"` resolves a TDSE request to `numpy_ref` before the registry, and native
refuses the route by name (`solprop_native.backend._refuse_unsupported`).

**The binding.** The driver owns the provider binding (`solprop.driver.tdse`, 3.2k
lines: the optional import, stable row chunks, the atomic-call budget, the raw archive,
consistency checks) and passes it to `registry.select(..., tdse_binding=...)`. Its
protocol is `solprop.translate.tdse.TDSEBinding`: `reserve(logical_calls)` before a
trial, `evaluate(fields, z=, purpose=, density=)` per source evaluation,
`record_trial(accepted=)` after each attempt, `release_reservation()` on every exit path,
`restore_ledger` on resume, and `metadata`, `memory_ledger` and `close`. One evaluation is
one atomic call per row chunk (`plan.evaluate(drive)`), each charged to the budget
before it starts and its raw result archived as it returns. `_load_provider` imports
`pyroraptor` dynamically; the binding checks the build's source revision against
`_PROVIDER_REVISION` and its execution space against `TDSEExecution.device`, and records
the build. `basis_from_preset`, `hacc_preset`, `hacc_basis` and `DiagnosticSession`
call `_load_provider` too.

**One source evaluation** in `numpy_ref` (`NumpyRefBackend._nonlinear_tdse`), from the
lab-frame modal field at the evaluation position, the input noise added:

1. transform to time on the fine field grid and synthesise the real field at each radial
   node, `(n_nodes, n_to)`;
2. `binding.evaluate(...)`, which returns the unscaled microscopic histories
   (`TDSEResult`);
3. assemble the macroscopic sources (`numpy_ref.tdse.atomic_sources`): the remainder
   current $S = J - j_1$ (less the third-order current for a cubic replacement), gated and
   scaled by density and the declared scale, the polarisation of the descriptor's Kerr
   model, and for a gated source the edge array $g' S$ (`atomic_edge_source`);
4. project both onto the mode and crop (`project_atomic_sources`), with the edge-free
   spectrum $X - E/(\mathrm{i}\omega)$ on the active bins and the embedded-quadrature and
   edge diagnostics;
5. apply the spectral collar, the in-stage absorber term and, in the interaction
   formulation, the inverse propagator.

The host Adams march reserves each trial's calls before the trial (`_afford`: one startup
evaluation when the history is empty, plus one for PEC or two for PECE), labels each
evaluation `startup`, `predictor` or `corrected`, commits an accepted step and then books
the attempt (`record_trial`). The run's metadata carries `atomic_calls`,
`tdse_execution`, `tdse_last_evaluation` and `tdse_source`, and the memory ledger the
binding's `tdse_*` rows. Manufactured bindings, which return analytic histories with no
provider, drive the TDSE tests in CI (`tests/backends/test_tdse_march.py`,
`tests/edge_source_cases.py`).

**The provider.** Pyroraptor is solprop's sibling package, by the same maintainer, built
on Kokkos and OpenMP and not published. Its interface, initialisation, thread handling and
exit-time clean-up can change in step with this SEP, as companion changes in its own
repository. An atomic call takes minutes; everything on solprop's side of an evaluation
is below $10^{-3}$ of a call (§6.8).

#### 3.3 What native would have to do

The native Adams march runs the trial in its resident engine and evaluates the source
natively; its rules forbid "allocation, planning, logging, Python call or unplanned fence
in a trial step" ([backends.md](../backends.md) §3.2, `native/AGENTS.md`). A TDSE source
cannot be evaluated without the provider, and its evaluation is a field-dependent,
minutes-long call that cannot be precomputed. With a trial lasting minutes, each `advance`
call returns after one attempt anyway.

#### 3.4 Decisions and changes this SEP builds on

- **ADR-077** (two backends, binary64 only; [decisions.md](../decisions.md)), and the
  change it calls for, which must be merged before implementation starts: in particular
  native importing nothing from `numpy_ref`. ADR-077 records that `numpy_ref` stays a
  complete, selectable backend (amended with the maintainer's decision in this SEP's
  review, §10.1), so ADR-068's rules stand: a feature lands on both backends, and a SEP is
  `Final` with parity on both.
- **ADR-052** (one thread budget per process), **ADR-056** and **ADR-071** (how idle
  native threads wait: sleeping at once, or after about ten microseconds of spinning on
  GNU's runtime and a 1 ms block time inside native calls on LLVM's).
- **ADR-060**: the compiled Adams march, with solprop's as its reference.
- **SEP 25**: independent computations agree to an error bound, never to bits.

#### 3.5 Related issues

From a search of the open issues on 2026-11-12:

- **Changed by this SEP.** [#247](https://github.com/LupoLab/solprop/issues/247) (the cost
  of a TDSE run): its "compiled coupling" item is the native hook; the optical side's
  share it asks about is negligible beside the atomic calls (§6.8).
- **Unchanged, for the record.** [#246](https://github.com/LupoLab/solprop/issues/246),
  [#248](https://github.com/LupoLab/solprop/issues/248),
  [#249](https://github.com/LupoLab/solprop/issues/249) and
  [#583](https://github.com/LupoLab/solprop/issues/583) (TDSE qualification, provider
  publication, certificate and cache). [#110](https://github.com/LupoLab/solprop/issues/110)
  (the step loops and the seam) is not answered here, since both executors stay.

### 4. Theory

The SEP changes no equation. What needs stating is what the hook computes and what bounds
its agreement with `numpy_ref`.

**The source the hook returns.** Let $a(z)$ be the lab-frame modal state at an evaluation
position $z$ (one mode, one polarisation, $n_\omega$ bins, the input noise added). The
shared source module returns the projected spectral source before the spectral collar,

$$
F(a, z) = p(z)\,\mathcal{F}\big[Q\,P(t)\big] + c(z)\,\Big(\mathcal{F}\big[Q\,g\,S(t)\big]
- \frac{\mathcal{F}\big[Q\,g' S(t)\big]}{\mathrm{i}\omega}\Big)\Big|_{\text{active}},
$$

where $P$ and $S = J - j_1$ are the macroscopic polarisation and remainder current at the
$n_{\text{nodes}}$ radial nodes (density and the declared scales applied once), $Q$ the
modal projection with the taper scale, $g$ and $g'$ the gate and its analytic derivative
(SEP 26), $p$ the stage prefactor and $c$ the independent current factor
([kernel-contract.md](../kernel-contract.md) §6). Each backend then forms what it forms
for every source: the stage output $w\,F$ with the spectral collar $w$, the in-stage
absorber term, and in the interaction formulation the inverse propagator
$e^{-\Delta\Phi}$. That is the composition of steps 1–5 in §3.2, split after step 4.

**What the comparison can show.** Up to $F$ both backends run the same code on the same
inputs, so their $F$ agree to the bit for the same $a$ and $z$ (class P). That checks the
wiring only: an error inside the shared module is invisible to it, and is caught by the
module's own independent tests (§7.1). After $F$, each backend's finish (a product with
the collar, a sum with the absorber term, a phase factor) and its optical march are its
own, so a TDSE stage and a short TDSE run on native agree with `numpy_ref`'s within the
one-stage and eight-step budgets `tests/bounds.py` derives for the existing source
comparisons ($16\varepsilon$ and $128\varepsilon$ peak-normalised on those cases,
[accuracy.md](../accuracy.md) §7). These are budgets derived for these cases, checked on
them, not guarantees for every TDSE problem or variable-step history. Where independently
rounded norms fall on either side of a decision threshold, the two runs fork; such a fork
is reported and the pair is then compared to an accuracy envelope (class T), not counted
as a failure and not hidden by a wider bound.

### 5. Proposed interface

#### 5.1 Users

A TDSE run is written as today, with no backend named:

```text
run = prop_capillary(..., tdse=response, tdse_execution=TDSEExecution(max_atomic_calls=400))
# backend="auto" -> "native" (its CPU build), where it was "numpy_ref"
```

- `backend="auto"` resolves a TDSE request to `native`, like every other request.
- `backend="numpy_ref"` still runs it, through the same provider worker.
- On a native device build a TDSE problem is refused by name before anything is
  allocated: "the TDSE response runs on the native CPU build; this build is CUDA/HIP".
- Pyroraptor is installed in the run's environment as today, and solprop still declares
  no dependency on it. It is never imported into solprop's process; the worker imports it.
- `TDSEExecution.device` chooses the provider's device, as today. During a TDSE run the
  provider gets the run's whole thread budget or its device (§6.6).
- Ctrl-C, SIGTERM and the run's stop flag stop a run during an atomic call as well as
  between steps (§6.7). The run ends interrupted or stopped, as any stopped run, with a
  terminal checkpoint at the last accepted step.
- Results, metadata keys (`atomic_calls`, `tdse_execution`, `tdse_last_evaluation`,
  `tdse_source`), budgets, warnings and errors are unchanged. The build record is the
  worker's report of the provider it loaded. The output schema does not change; the
  backend is outside the resume key, so a TDSE run resumes on either backend (class T).

#### 5.2 Developers

`solprop.backends.tdse_source` is internal. Sketch:

```text
from solprop.backends import tdse_source

fields = tdse_source.node_fields(problem, lab)              # step 1 of §3.2: (n_nodes, n_to)
sources = tdse_source.assemble(problem, result, z, density) # step 3: P, g S, g' S
F, diagnostics = tdse_source.project(problem, sources, z)   # step 4: F of §4, with its floors
```

`numpy_ref` calls it in place of its own `tdse.py`, which goes; native's hook calls the
same three. The worker's protocol is private to `solprop.driver.tdse`.

### 6. Implementation strategies

#### 6.1 Layers and responsibilities

| layer | owns |
|---|---|
| translation | the TDSE parameters and tables: admission, the descriptor, the gate, the node quadrature, the prefactors |
| `solprop.backends.tdse_source` | at each evaluation: synthesis to the nodes, the current subtraction, density and taper scaling, gating, transforms, projection and the diagnostics' floors |
| `solprop.driver.tdse` (the binding) | the provider worker and its lifetime, budgets and reservations, chunks, the archive, consistency checks, memory admission, cancellation while waiting, every resource's release |
| the worker (`solprop.driver._source_worker`, with Pyroraptor's adapter) | Pyroraptor: its import, initialisation, models, plans and atomic calls |
| `numpy_ref` | its own optical march and history; calls `tdse_source` and the binding |
| native | its own optical march and history (C++); the hook calls `tdse_source` and the binding |

Import boundaries, each enforced by a test: native never imports `numpy_ref`;
solprop's process never imports `pyroraptor`; the worker never imports
`solprop_native`, `numpy_ref` or the user's `__main__`.

#### 6.2 The host source hook

**C++.** The resident Adams engine gains one optional member, a type-erased callable
`HostSource` with three entry points. When it is set, the engine's source evaluation for
the startup, predictor and corrector calls hands the hook the lab-frame field it would hand
its own source pass (a host view of the state, the input noise added) and an output view
allocated once at construction and listed in the memory ledger (`tdse_host_source`, two
`n_omega` complex rows). It then runs its usual stage output on that buffer. The engine's
own response pass is not run on a TDSE problem: the hook's source is the whole nonlinear
term, the optmat cubic included when the descriptor asks for it, as on `numpy_ref`. The
engine calls the hook from the thread that called `advance`, outside any parallel region.
The engine's translation unit sees no Python; the binding layer wraps the Python object
and acquires the GIL for each call.

**Not TDSE-specific.** The C++ member and its session protocol know a lab-frame field in,
a spectral source out, and an attempt's start and end; nothing about atoms, nodes or
Pyroraptor. TDSE is the first hook object; another host-evaluated source would be a second
one, with its own admission (§12 P10).

**The three entry points** (Q1) mirror the reference's binding calls at the same points
of an attempt:

| entry point | when | does |
|---|---|---|
| `begin(cost)` | after the march's budget check, before an attempt's first evaluation | `binding.reserve(cost)`, `cost` computed by the march as `_afford` does |
| `evaluate(lab, z, purpose, out)` | each source evaluation: `startup`, `predictor`, `corrected` | `tdse_source.node_fields`, `binding.evaluate`, `assemble`, `project`, written into `out` |
| `end(accepted)` | after the decision and, for an accepted attempt, after the commit | `binding.record_trial(accepted=...)`, then `release_reservation()` in a `finally` |

None runs on a problem without a TDSE route.

**Session.** `NativeSession` constructs the hook from the driver's binding when the problem
carries a TDSE route, binds it before the march starts and unbinds it on close; it records
the TDSE metadata as `numpy_ref` does, adds the binding's `tdse_*` memory rows, and
restores the call ledgers on resume. The registry passes `tdse_binding` to native's
factory, which today refuses it. A device build refuses a TDSE problem at preparation,
and the hook is compiled out of device kernels.

#### 6.3 The failure boundary

The numerical commit of an accepted attempt is the failure boundary (R10). The same
contract holds on both backends:

| failure | when | state afterwards |
|---|---|---|
| in `begin` (budget refusal, archive not writable, cancellation) | before the attempt | last accepted field, position and history; nothing reserved |
| in `evaluate` (provider failure, worker death, protocol error, archive write, cancellation) | before commit | the trial discarded; last accepted field, position and history |
| a rejected attempt, whatever follows | — | the accepted state never advances |
| in `end` after an accepted commit (archive recording, a signal held over the commit) | after commit | the new accepted state stands and is booked; the run stops with the original error, the archive failure named, and the state checkpointable where the output allows |

On every path: the unused reservation is released; charges for atomic calls already
attempted stay (a call is charged before it starts, so a cancelled call stays charged);
completed atomic work and archive writes are not undone; clean-up failures are attached
to the original error as notes, never replace it. In C++ a Python exception from an entry
point is caught at the binding layer as `nb::python_error` and stored, never inside a
Kokkos kernel. Before commit the engine abandons the attempt and the march returns with a
host-source reason naming the stage; after commit the march returns the committed attempt
in its batch with the same reason. The session books the batch and re-raises the stored
exception unchanged, so a `ResponseBudgetError` still carries the last accepted position,
field and calls.

#### 6.4 Options for evaluating the TDSE source on native

| option | what | for | against |
|---|---|---|---|
| **A. Host callback per evaluation** (recommended) | the hook of §6.2 | the TDSE path stays one Python implementation; native's march, history, monitors and statistics serve TDSE unchanged | an exception to "no Python call in a trial step"; CPU builds only |
| B. Suspend the march at each evaluation | the march returns to the session with a "source needed" event mid-trial and resumes after the host fills the buffer | no Python call inside a trial | splits the exponential Adams trial into a resumable state machine that every non-TDSE run then carries |
| C. Link Pyroraptor into solprop-native | C++ calls into the provider | no host crossing | both would need one Kokkos version, configuration and ABI, ruling out a GPU provider beside the CPU wheel; the binding and source assembly (4k lines) transcribed or still called from Python; no measurable gain (§6.8) |
| D. Precompute the response | tabulate it and evaluate natively | — | impossible: the response depends on the whole field waveform at each node |
| E. Keep TDSE on `numpy_ref` | `auto` keeps its TDSE rule | no native work and no ADR exception; the shared source and the worker still land and serve it | native keeps its refusals and `auto` its exception; TDSE runs keep the reference's optical side and session features |

With `numpy_ref` retained, E is the status quo for TDSE. A is chosen (§12 P4): production
runs belong on native.

#### 6.5 The provider worker

The worker has two layers. A provider-agnostic one (`solprop.driver._source_worker`)
owns the start, framing, handshake, lifetime, signals and cancellation; a provider adapter
runs inside it, Pyroraptor's for real runs and a manufactured one in CI, so the generic
layer is exercised by two providers from the start (§12 P10). The binding
(`solprop.driver.tdse`) runs Pyroraptor's adapter in a worker for every use of the
provider: a run's evaluations, `basis_from_preset`, `hacc_preset`, `hacc_basis` and
`DiagnosticSession`. `_load_provider` is called only in the worker.

- **Start.** `subprocess` launches the same interpreter with the parent's import path and
  environment, running an isolated entry point (`sys.executable -m
  solprop.driver._source_worker <adapter>`). `multiprocessing`'s spawn and forkserver methods are not
  used, since they re-import the parent's `__main__` (a user's script or notebook), and
  fork never, since it inherits the parent's runtimes. The entry point imports neither
  `solprop_native` nor `numpy_ref`, and a test holds it to that. Before importing the
  provider it moves its protocol to private descriptors and points its standard output at
  standard error, so the provider's or Kokkos's printing cannot corrupt a frame. It starts
  in its own process group and ignores SIGINT and SIGTERM, so a terminal's Ctrl-C or a
  scheduler's signal to every process of a job reaches the parent, which then ends it
  (§6.7); and a watchdog thread ends it at once when its pipe from the parent reaches end
  of file, so a parent killed outright does not leave it computing.
- **Environment.** The parent's, except that the OpenMP wait variables solprop-native set
  itself (those its marker `SOLPROP_NATIVE_WAIT` identifies, ADR-071) are removed, so the
  provider's runtime keeps its own defaults; a user's own settings pass through. The
  thread count and device are given explicitly (§6.6).
- **Transport** (Q11). One pipe each way, control and arrays alike; no shared memory.
  Frames are length-prefixed. A control frame is a small JSON record; an array travels as
  raw bytes after a header naming its dtype, shape and byte order, unchanged. No object is
  pickled, so a message cannot execute code and the two sides need not share a pickle
  protocol. Every frame has a bound computed from the prepared shapes; a larger declared
  length is a protocol error before anything is allocated.
- **Handshake.** The worker's first frame carries the protocol version and the provider's
  identity: its build record (source revision, execution space, Kokkos configuration) and
  the worker's process id. The parent refuses a protocol mismatch, a revision other than
  `_PROVIDER_REVISION` or an execution space other than `TDSEExecution.device` by name,
  before any evaluation. The build record in a run's metadata is this report.
- **What crosses.** Preparation sends the descriptor's plain records (basis, model,
  outputs, chunks); the worker builds the models and plans and replies with its memory
  rows and plan records. Each atomic call sends one chunk's drive rows and receives its
  raw histories as arrays with a plain record of the provider's diagnostics. Budgets,
  reservations, ledgers, the archive, validation, normalisation, consistency checks and
  the source assembly stay in solprop's process.
- **Failures.** A provider exception is classified in the worker (which has the
  provider's error types, as `_classify` needs) and returned as its type name, message,
  classification and the worker's traceback text; the parent raises the existing provider
  failure (`BackendError`, "TDSE failure at z = …, rows …") with the traceback attached as
  a note. A worker that exits, closes its pipe or sends a malformed frame ends the call as
  a provider failure naming its exit status and the tail of its standard error. A failure
  during start-up or preparation, a missing or stale provider included, is reported by
  name as today.
- **Lifetime.** The parent owns the worker and both pipe ends from `prepare_binding` to
  `close`, through the binding's `ExitStack`. Every exit path, a failure in start-up
  included, ends the worker and then reaps it and closes both ends: between calls by a
  shutdown request with a short grace period, then a kill; during a call, after a failure
  or on cancellation by a kill at once, since the worker holds nothing solprop needs. A
  worker is never reused after a failure, a protocol error or a cancellation, and a pipe
  interrupted by termination is discarded with any partial frame. A dead worker is always
  an observable failure, although a live call has no time limit. The preset and basis
  queries use a short-lived worker of their own; their results are plain records and
  may be cached for the process's life.
- **Memory.** The binding's ledger, admitted against `TDSEExecution.memory_limit` as
  today, gains rows for the transport copies on both sides (`tdse_transport`, the largest
  chunk's drive and result in each process) beside the provider's own rows, which the
  worker reports.

#### 6.6 Threads and devices

During a TDSE run the worker gets the run's **whole** thread budget (ADR-052), or, with
a device build of Pyroraptor, its device. solprop's native pool keeps its size, set at
import and not resizable, and its threads wait while the hook blocks: they sleep at once
under the passive policy (ADR-056), after about ten microseconds of spinning on GNU's
runtime, and after LLVM's 1 ms block time inside native calls (ADR-071), all negligible
against calls of minutes. The pools take turns, as ADR-052 already assumes, so a one-core
budget gives the provider its one thread rather than none, and the optical side's work
between atomic calls, which is negligible, runs on the native pool while the provider is
idle. This keeps Q2's intent, that the provider gets the machine, without requiring
native's pool to shrink to one thread (§12, P1). The two never share a device: a native
device build refuses a TDSE problem.

#### 6.7 Cancellation

There is no automatic per-evaluation timeout (R11): a live atomic call may take as long
as it takes. Cancellation is explicit: Ctrl-C (SIGINT), SIGTERM (a Slurm time limit, a
GUI job's stop) and the run's `StopFlag` (a stop button, a scan's supervisor).

- **Waiting.** The parent reads the worker's frames on a helper thread, which works alike
  on POSIX and Windows pipes. The hook's wait, and the binding's wait during start-up and
  preparation, is a loop on that reader with a short timeout (about 0.2 s). Between
  polls Python's signal handlers run, and the loop checks the run's `StopFlag`, which is
  otherwise polled only before native calls.
- **Signals.** On `numpy_ref` the binding waits outside a commit, so a handler raises
  `KeyboardInterrupt` or `RunTerminated` at once (`_interrupts.py`). On native the wait
  is inside `advance`, which the session wraps in `committing()`, so the handler holds
  the signal; the loop takes the held signal (a small addition to `_interrupts`) and
  raises it there. A stop flag raises `StopRequested`.
- **Ending.** On any of these the parent kills the worker without waiting for the call
  (§6.5, *Lifetime*), discards the unfinished trial and keeps the last accepted state for
  checkpointing (§6.3). The call in flight stays charged and the archive records the
  evaluation as interrupted. A user's cancellation is an interrupted or stopped run, never
  a provider failure, and a worker exit observed after the parent began to cancel belongs
  to the cancellation. The stop takes at most the poll interval and the kill; there is no
  evaluation timeout.
- **Threads.** Signals reach only the main thread, so they cancel a run started there,
  as scripts, notebooks, scan workers and the GUI's jobs start them. A run started on
  another thread (an embedding application's worker thread) is cancelled only by its
  `StopFlag`, which the loop polls on every thread.

#### 6.8 Cost

**Time.** One TDSE evaluation is a complete atomic calculation at every radial node:
93 s (production basis, 17 nodes) to 615 s (strong-field basis, 33 nodes) on compressed
fields in [SEP 12](archive/sep-0012-tdse-nonlinear-response.md)'s node selection
(Part II), and 366 s (haCC(1)) and 1,307 s (haCC(5)) on one AMD MI325X
([SEP 20](archive/sep-0020-hacc-helium-tdse.md), Part II). Per evaluation the hook adds
a GIL acquisition and a Python call (microseconds), the worker a copy each way of a few MB
and a context switch (milliseconds), and the host source assembly, which `numpy_ref`
already performs (of order 10–100 ms at the response lattices in use, to be measured in
phase 0). Together that is below $10^{-3}$ of the fastest recorded call. A worker start
costs one interpreter and provider import per run. The optical side of each step is
negligible beside an atomic call, so the change is not a speed-up.

**Memory.** Two `n_omega` complex rows for the hook, the transport copies of §6.5, and the
provider's rows, now in the worker.

**Code** (estimated additions; `poe size` before and after, compared at `Final`; nothing
is removed but `numpy_ref/tdse.py`, which moves):

| part | `src/solprop` | native Python | C++ | tests |
|---|---|---|---|---|
| shared source | about 0 (a move) | — | — | +0.2k |
| provider worker | +0.6k | — | — | +0.6k |
| native hook | — | +0.4k | +0.3k | +0.5k |
| total | +0.6k | +0.4k | +0.3k | +1.3k |

**Dependencies.** None new.

#### 6.9 The kernel contract

The kernel problem does not change: the TDSE descriptor already crosses it, and the hook
is a session binding outside it, as the binding is today. The hook's binding joins the
members `check_build` looks up by name, and [kernel-contract.md](../kernel-contract.md)
§6's "the native backend refuses TDSE by name and never holds a binding" is rewritten,
with a release-note fragment.

### 7. Verification and validation

**The oracle boundary.** What is independent: each backend's optical march, history,
stage finish and decisions, and the manufactured providers' analytic histories. What is
shared: translation's tables and parameters, the host source module, and the binding.
Agreement between the backends checks the independent parts and the wiring of the shared
ones; the shared parts are checked by the independent expectations of §7.1 and by the
existing analytic TDSE tests. §7.1–§7.3 run in CI with manufactured providers.

#### 7.1 The shared source (class P and R)

The module's tests state their expectations independently of the module, analytic or
manufactured, so that a defect in a shared term fails them even though both backends
agree: the sign of $S = J - j_1$ and of the cubic replacement, density and taper scaling,
the gate and its edge term $g' S/(\mathrm{i}\omega)$, the active-band masking and
cropping, and the diagnostics' floors. The present `numpy_ref` TDSE tests, which already
state such expectations (`tests/edge_source_cases.py`), run unchanged after the move, and
a deliberate defect in each named term is shown to fail one of them.

#### 7.2 The worker contract (class P)

A manufactured provider module, importable in place of `pyroraptor`, inside a real
worker:

1. evaluations through the worker equal calling the manufactured provider directly, bit
   for bit, and the binding's call sequence, ledgers and archive are unchanged;
2. the handshake refuses a protocol, revision or execution-space mismatch by name;
3. a provider exception, a worker exit, a closed pipe, a malformed or oversized frame, and
   a failure during start-up and preparation each end as the named failure of §6.5, with
   the worker reaped and both pipe ends closed;
4. explicit cancellation of a worker that hangs inside a call, by SIGINT, SIGTERM and a
   stop flag, on both backends: the run ends interrupted or stopped within the poll
   interval and the kill, the call stays charged and the last accepted state is checkpointed; test
   deadlines detect a broken cancellation, and the production code has none;
5. signals: SIGINT or SIGTERM sent to the whole process group (as a terminal and a
   scheduler send them) ends the run as interrupted, not as a provider failure; a parent
   killed with SIGKILL during a hanging call leaves no worker behind (the watchdog);
6. import isolation: after a TDSE run `pyroraptor` is not in solprop's `sys.modules`, and
   the worker never loads `solprop_native`, `numpy_ref` or the parent's `__main__`,
   started from a script whose re-import would fail, from `python -c`, and from a scan
   executor's worker process;
7. memory rows for the transport and the worker's report appear in the ledger, and an
   admission over `memory_limit` is refused before start.

#### 7.3 TDSE on native (class P, R and T)

Against `numpy_ref`, with manufactured bindings in solprop's process (they replace the
binding, not the provider, and need no worker):

1. **One evaluation**: the hook's $F$ against `numpy_ref`'s for the same lab field and
   position, bit for bit; the stage output within the one-stage budget of §4, on gated
   and ungated descriptors, each Kerr model, with and without input noise.
2. **The call sequence** (class P): the exact sequence of `reserve`, `evaluate(purpose,
   z)`, `record_trial` and `release_reservation` on a recording binding, for PEC and PECE,
   a rejection, a restart, a budget refusal mid-trial, a provider failure and a resume;
   the ledgers and metadata equal to `numpy_ref`'s.
3. **The failure boundary** of §6.3 on both backends: a failure in each entry point
   before commit, an archive failure in `end` after an accepted commit, and a signal at
   the boundary, each leaving the stated state, ledgers and checkpoint.
4. **Eight adaptive steps**: the same accepted steps and orders, and the fields within
   the eight-step budget of §4; a fork at a decision threshold is reported and compared
   as class T. Run with a one-attempt call bound and with the production bound, so that
   the resident call's normal path is tested too.
5. **The existing TDSE tests** (`tests/backends/test_tdse_march.py`,
   `tests/driver/test_tdse_*.py`, `tests/driver/test_sep12_sep14_integration.py`) run on
   both backends with their tolerances unchanged.
6. **Refusals**: a device build, by name.

#### 7.4 A real provider, by hand

Not in CI (Q10). Pyroraptor is tested by hand as it is used: one short segment of at most
three atomic calls on the smallest preset, native against `numpy_ref` (class T), through
the worker, with Pyroraptor on the CPU and, where a GPU build is in use, on a GPU; and one
cancellation during a call. Each check is reported in Part II with its configuration
(provider revision, build, machine, threads or device). CI establishes the manufactured
contracts, not compatibility with an installed provider; the handshake's revision and
execution-space check (§6.5) refuses a mismatched provider at run time.

### 8. Documentation

- **Manual.** [TDSE response](../../guide/media/tdse.md): runs on native's CPU build and
  on `numpy_ref`; the provider in its worker, its threads or device, and stopping a run;
  [Choosing a backend](../../guide/performance/backends.md): TDSE leaves the list of
  features that run only on the reference.
- **Design pages.** [backends.md](../backends.md) §1 (the `auto` rule) and §3.2 (the
  hook's exception); [kernel-contract.md](../kernel-contract.md) §6;
  [architecture.md](../architecture.md) (the shared source module, the worker);
  [testing.md](../testing.md) (the oracle boundary for shared code; worker-contract
  tests); `AGENTS.md` and `native/AGENTS.md` (the hook's exception).
- **Records.** One ADR: the host source hook (the one Python call in a native trial, its
  scope, its refusal on device builds), the provider worker, and the failure boundary;
  it amends ADR-077's "every feature except the TDSE response, for now".
- **Release notes** for the `auto` change, the worker and cancellation during an atomic
  call.

### 9. Compatibility and impact

#### 9.1 Users

- `backend="auto"` runs TDSE on native; `backend="numpy_ref"` still runs it.
- A run no longer imports Pyroraptor into the user's process; a script may still import
  it itself.
- No physics default, output schema or resume key changes.

#### 9.2 Contributors

Nothing changes for development without a compiler: the `no-native` lane and
`tools/setup_light.py` stay, and `numpy_ref`'s TDSE tests run there with manufactured
providers.

#### 9.3 Risks

- **Transport and process start across platforms.** Pipes, process start and termination
  differ on Linux, macOS and Windows. Mitigation: §7.2 in CI on every platform CI runs;
  the reader thread avoids platform-specific polling.
- **Provider compatibility untested in CI** (Q10). Mitigation: the handshake refuses a
  mismatched revision or execution space by name; manual evidence with its configuration
  (§7.4).
- **Python exceptions through C++.** Mitigation: caught at the binding layer, never inside
  a Kokkos kernel; §7.3 items 2–3 cover every entry point and the boundary.
- **A shared-code defect invisible to parity.** Mitigation: §7.1's independent
  expectations and the defect checks.

### 10. Alternatives considered

#### 10.1 Slim `numpy_ref` to a test library (this SEP's former phase 2)

The first draft also reduced `numpy_ref` to a library of the contract arithmetic and the
decision specification, with no step loops, sessions, groups, batch or registry row, and
replaced full-run parity with native's run replayed on the reference plus decision
traces. The maintainer considered and deferred it on 2026-11-12 (Part III, review
follow-up), keeping independent complete runs for development, debugging and numerical
verification. The review's size audit, at PR #767 (`1acdffc4`), found 7,474 code lines in
`backends/host/` and `backends/numpy_ref/` outside the TDSE assembly, of which the session,
group and batch modules are 1,217 lines of gross removal; the draft's saving of about
11.5k source lines is not substantiated for a design that keeps independent integration,
and replay and decision checks alone do not replace independent runs (R1–R4). Q4–Q8 went
with it. A later proposal may revisit it, or a narrower part of it such as the reference's
group and batch paths.

#### 10.2 Other choices

- **Keep TDSE on `numpy_ref`** (option E, §6.4): the status quo; the shared source and the
  worker would still land.
- **Delete `numpy_ref` entirely.** Rejected: physics SEPs need an independent, readable
  implementation to develop and qualify against.
- **The provider in solprop's process.** Rejected by the maintainer: it shares two Kokkos
  and OpenMP runtimes in one address space and rules out a GPU provider beside the CPU
  wheel, to save milliseconds per call of minutes.
- **Shared-memory transport.** Rejected (Q11): its ownership and clean-up machinery buys
  nothing measurable against calls of minutes.
- **`multiprocessing` start methods.** Fork inherits the parent's runtimes; spawn and
  forkserver re-import the parent's `__main__`. Both rejected for `subprocess` (§6.5).
- **An automatic per-evaluation timeout.** Rejected (R11): a call's length is legitimate;
  stopping is the user's explicit choice.
- **A real-provider CI job.** Rejected (Q10).
- **Suspending the march, linking the provider, precomputing** (B–D, §6.4).

### 11. Phases

One implementation pull request under SEP 0, opened as a draft when implementation
starts, with the phases as successive commits and their reports in Part II; Pyroraptor's
changes (an explicit thread count at initialisation, plain-data results, quiet exit) in a
linked companion pull request in its repository. PR #765 stays the proposal's.

| phase | deliverables | verifies |
|---|---|---|
| **0. Baseline** | ADR-077's change merged; `poe size` baseline; the host assembly's time per evaluation with a manufactured binding on the short TDSE test window | the numbers recorded in the phase report |
| **1. Shared source** | `solprop.backends.tdse_source` (a move from `numpy_ref`); `numpy_ref` calls it | §7.1 |
| **2. Provider worker** | the worker, its protocol, lifetime, cancellation and memory rows in the binding; presets, bases and diagnostic sessions through it; `numpy_ref`'s TDSE runs use it; the Pyroraptor companion pull request | §7.2 |
| **3. Native hook** | the C++ hook, its binding and the adapter's hook object; session wiring, metadata, memory rows, resume; the failure boundary; `auto` sends TDSE to native; device builds refuse | §7.3 |
| **4. Qualification and documentation** | the remaining tests; the manual check of §7.4; the ADR; the pages and release notes of §8 | §7.4; the size report against §6.8 |

### 12. Open questions

Decided by the maintainer on 2026-11-12 (Part III, review follow-up):

1. **Q1. The hook's shape.** Three callbacks, `begin(cost)`, `evaluate(...)` and
   `end(accepted)` (§6.2).
2. **Q2. Device builds.** No TDSE on native device builds; the provider gets the GPU or
   the CPU cores (§6.6, P1).
3. **Q3. Where the host TDSE source lives.** `solprop.backends.tdse_source`, called by both
   backends, with the responsibilities of §6.1 and the independent tests of §7.1.
4. **Q9. Delivery.** One implementation pull request; Pyroraptor's changes in a linked
   companion (§11).
5. **Q10. A real provider in CI.** None; by hand as used, reported with its configuration;
   the optional import and revision check kept, and the worker's actual provider
   identity reported (§6.5, §7.4).
6. **Q11. The transport.** Pipes only, arrays unchanged, bounded frames, transport copies
   in the memory ledger, responsive to cancellation (§6.5, §6.7).

Q4–Q8 concerned the reference's reduction and went with it (§10.1).

#### Points for the maintainer

The author's resolutions of points the decisions left open. On 2026-11-12 the maintainer
accepted P1, P2, P3 and P5, decided P4 (the native hook is wanted), and asked for the
cancellation of P3 to work in every case; he accepted P6–P10 the same day.

- **P1. Threads** (refines Q2's wording). The worker gets the run's whole thread budget
  or its device; native's pool keeps its size and waits while the hook blocks; it is not
  shrunk to one thread, since a process-wide pool initialised earlier cannot be resized
  and a one-core budget would leave the provider none. The optical side's work between
  calls is negligible. This keeps Q2's intent (§6.6).
- **P2. Worker start and protocol.** `subprocess` running `sys.executable -m
  solprop.driver._source_worker`, never `multiprocessing`'s spawn or fork; length-prefixed
  frames, arrays as raw bytes with a dtype, shape and byte-order header, no pickle;
  provider exceptions as type, message, classification and traceback text, re-raised as
  the existing provider failure; a handshake with the protocol version and the
  provider's identity, refused on mismatch before any evaluation, which partly offsets
  Q10's manual-only provider testing (§6.5).
- **P3. Cancellation mechanics.** A wait loop with a short timeout, on a reader thread's
  queue so that it works alike on Windows; between polls signal handlers run and the run's
  `StopFlag` is checked. Because native's `advance` runs inside `committing()`, a signal
  during the hook is held rather than raised, so the loop takes the held signal and raises
  it, an extension of `_interrupts`. Every cancellation is a failure before commit. Signals
  work for a run on the main thread; elsewhere only the `StopFlag` does (§6.7).
  *Accepted*, with the maintainer's request that stopping work in every case: it does for
  every way solprop starts a run (scripts, notebooks, the GUI's jobs, scan workers, all on
  their process's main thread), and for an application embedding a run on another thread
  through that run's `StopFlag`, since Python delivers signals to the main thread only.
- **P4. Value and order.** With `numpy_ref` retained, the native hook's benefit is
  uniformity (`auto` picks native for every problem, native's session features for TDSE,
  native's TDSE refusals go) and an unmeasured optical-side speed-up (#247), against about
  1.2k added lines and an exception to "no Python call in a trial step". The worker is
  valuable alone (isolation, a GPU provider beside CPU solprop) and serves `numpy_ref` at
  once. Recommended: phase 0 measures the optical side's share and the host assembly's
  time; the pull request lands the shared source and the worker first, then the hook.
  *Decided:* the hook is wanted. The optical side is certainly negligible, so speed is
  not the reason; native is the executor of nearly every run, and the more reliable
  choice for TDSE too (§2, §6.4, §11).
- **P5. Backends and qualification.** `auto` sends TDSE to native; `numpy_ref` still runs
  TDSE, through the worker; parity between them is the qualification, with the
  `tests/bounds.py` budgets stated as derived for these cases, not as universal
  guarantees, and near-threshold forks reported as class T, not failures (§4, §7.3).
- **P6. ADR-077.** Already amended before merge to keep `numpy_ref` a complete backend;
  the SEP's ADR amends only its "every feature except the TDSE response, for now" (§3.4,
  §8).
- **P7. Signals and the worker's process.** A terminal's Ctrl-C reaches its whole process
  group, and a Slurm time limit or `scancel` signals every process of the job, so the
  worker would otherwise die beside the parent and the parent would read a provider
  failure where the user stopped the run. The worker therefore starts in its own process
  group (session on POSIX, `CREATE_NEW_PROCESS_GROUP` on Windows), ignores SIGINT and
  SIGTERM, and is ended only by its parent; and the parent classifies any worker exit
  after it has begun to cancel as part of the cancellation (§6.5, §6.7).
- **P8. Parent death.** A parent killed outright (out of memory, `kill -9`) cannot stop
  its worker, which would otherwise keep computing, possibly on a GPU, for hours. The
  worker runs a watchdog thread that ends its process at once when its pipe from the
  parent reaches end of file, even during a call (§6.5).
- **P9. No grace period during a call.** All solprop records, the archive included, are
  the parent's, so a worker in a call has nothing to save: on cancellation the parent
  kills it at once rather than waiting a grace period it cannot use (the provider does not
  return mid-call). An orderly shutdown with a short grace period is kept for `close`
  between calls (§6.5, §6.7).
- **P10. A general host source, not a TDSE one.** The native hook and the worker's
  generic layer are written without TDSE or Pyroraptor in them (§6.2, §6.5), at almost no
  extra cost: TDSE is the first hook object and Pyroraptor the first provider adapter,
  with the manufactured provider as the second. That opens a route for other
  host-evaluated sources and providers, and keeps Pyroraptor an adapter module rather
  than a dependency. A *public* way for users to plug in an arbitrary external source is
  not part of this SEP: it needs its own contract (which routes admit it, the field and
  source it exchanges and their units, budgets, resume and its refusals) and is better
  designed with a second real use in hand, as a later SEP.

### 13. References

1. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM, 2002).
   doi:10.1137/1.9780898718027
2. nanobind documentation, "GIL management" and "Exceptions",
   <https://nanobind.readthedocs.io/en/latest/>
3. Python documentation, `multiprocessing`, "Contexts and start methods", and `signal`,
   "Signals and threads", <https://docs.python.org/3/library/>

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
- 2026-11-12: AI review by GPT-6 of all of Part I, with particular attention to the
  verification and development capabilities retained after removing the reference
  executor; R1–R11 below. Only review metadata and this record changed.
- 2026-11-12: Following extensive discussion with the AI reviewer and a code-size and
  responsibility audit against PR #767, John Travers decided to keep `numpy_ref` as
  a backend for now. The review follow-up below records that decision, the revised
  recommendation to consider the native TDSE work separately, and the remaining
  relevant questions. This records a review decision; it does not rewrite Part I.
- 2026-11-12: The maintainer answered the remaining relevant questions in discussion
  with the AI reviewer. The review follow-up records the three callbacks, the commit
  boundary for failures, shared source assembly, one TDSE implementation PR, manual
  real-provider testing, pipe-only transport, and prompt explicit cancellation with
  no automatic evaluation timeout. The proposal text awaits the author's revision.
- 2026-11-12: Part I revised by the drafting agent to the review decisions: scope
  reduced to the TDSE response on native, numpy_ref retained (phase 2 removed), Q1–Q3,
  Q9–Q11, R9–R11 incorporated; the author's resolutions P1–P5 (and the drafting
  agent's P6) listed in §12 for the maintainer.
- 2026-11-12: Author's review of the revision: ADR-077 is already amended to keep
  `numpy_ref` complete (P6); added P7 (the worker in its own process group, ignoring
  SIGINT and SIGTERM, so a group-wide or scheduler signal is a stop, not a provider
  failure), P8 (a watchdog ends the worker when its parent dies) and P9 (no grace period
  during a call). The maintainer accepted P1, P2, P3 and P5 and decided P4 (the hook is
  wanted; the optical side is negligible, and native is the reliable default); phase 0's
  optical-share measurement dropped. P10 proposed: the hook and the worker's generic
  layer are not TDSE- or Pyroraptor-specific; a public external-source API is left to a
  later SEP.
- 2026-11-12: Reviewed by John Travers (human): P6–P10 accepted, so every open question
  and point is decided or deferred. Review status `Reviewed` (GPT-6 (AI), John Travers
  (human)); the status stays `Draft` until accepted.
- 2026-11-12: Accepted by John Travers (#765); implementation to start once ADR-077's
  change (#767) is merged, in one implementation pull request (§11).

### AI review — 2026-11-12

Reviewer: **GPT-6 (AI)**. Reviewed PR #765 at `63201dca4ab69d21b6d4ae0cd4611c4203ff47b1`,
including the worker-process revisions, against the current design, SEP 0, SEP 25,
the reference and native marches, their decision corpora, numerical parity tests,
session tests and contributor tooling. These findings belong to
[#764](https://github.com/LupoLab/solprop/issues/764). Part I is unchanged. The initial
findings R1–R11 precede the maintainer decision recorded in the review follow-up
below; that follow-up supersedes the initial recommendation on reference removal.

**Initial assessment.** The production sessions, groups and registry selection are separable
from the reference's scientific value. Removing those mechanisms can be a good change,
and native can still have strong independent numerical verification without a second
production backend. The current proposal does not yet establish that equivalence:
native-scheduled replay checks execution of a chosen trajectory, and checking decisions
on native's own readings checks policy conditional on those readings. Neither alone
establishes the readings' correctness, adaptive accuracy, or the ability to develop a
new response before native implements it. I recommend a small independently runnable
reference, built on the retained arithmetic and decision functions, alongside the
proposed replay. It need not implement `Backend`, production sessions, output stores,
groups, or the GUI. That alternative deserves comparison with the two extremes in §10.

The TDSE source split and separate provider process are reasonable. R10–R11 concern
execution contracts, without reopening the TDSE model or the maintainer's device
decision. Phase 1 is useful independently; I recommend making phase 2b conditional on
the evidence requested below, rather than treating deletion or a line-count target as
its own success criterion. **High** identifies a verification or correctness gap to
resolve before accepting the affected design; **Medium** identifies a development or
qualification contract that needs an explicit answer.

**Evidence run.** In a Python 3.14.5 environment without solprop-native, the scripted
reference suites `tests/backends/test_adams_traces.py` and
`tests/backends/test_rk_traces.py` passed, **67 tests**. Two small Python probes used
the existing `adams_core.choose_order` and `integrated_weights`; their results appear
in R2 and R5. These are reference-policy and coefficient probes, not native parity or
propagation measurements. Native, GPU and real-provider execution were read, not run;
no physical constant or TDSE model was requalified by this review.

1. **R1 — High: independently verify every decision's numerical inputs
   (§§4, 5.2, 6.4, 7.2, 9.4 R2).** Feeding native's reported norm back into
   `adams_core` or `rk_core` cannot establish that native formed the right defect or
   norm. For example, dividing every reported error by 100 while leaving each trial's
   field update correct can produce an inaccurate adaptive schedule that both the
   accepted-step replay and a policy check on the divided errors accept. The scripted
   corpora deliberately replace field arithmetic with supplied norms, so they do not
   close this gap. The one-trial rung helps, but must specify what is compared and
   must reach the cases the real adaptive run encounters.

   The existing
   [native error-norm tests](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/tests/backends/test_native_error_norm.py)
   explicitly check the norm of native's operands and the norm of independently
   computed trial stages, including masking and nonuniform weights. Preserve these
   two levels. **Requested change:** on the short adaptive ladder cases, compare
   native and reference defects and scaled norms for rejected as well as accepted
   attempts, and all scored Adams orders. Include the state scales, masks, weights,
   non-finite cases and rejection rollback. Then check policy on identical supplied
   readings separately. Demonstrate that an error-scale mutation and a mutation
   confined to a rejected trial fail the replacement checks. Do not describe
   `check_decisions` as covering rejected-trial numerics merely because their reported
   norms are present.

2. **R2 — High: `StepTrace` lacks the information needed to check the advertised
   decisions (§§4, 5.2, 6.4, 6.7).** The current attempt row contains one norm and
   the order used, but
   [`adams_core.choose_order`](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/src/solprop/backends/adams_core.py)
   also reads the lower- and higher-order defect scores. Native's
   [`adams_march.cpp`](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/native/src/adams_march.cpp)
   records only `norms[0]`, although `choose_order` consumes the other scores too.

   **Probe:** with the default `Multistep` policy, no active collar cap, current
   order 2 and `dz=0.001`, scores `{2: 0.5, 1: 1.0, 3: 1.0}` keep order 2 and propose
   approximately `0.00100793684`; changing only the order-3 score to `0.001` chooses
   order 3 and proposes `0.002`. Both have the same recorded attempt norm, accepted
   flag and order. Restarts and landings alone cannot fill in the missing score.

   **Requested change:** specify the trace's complete input and output records before
   the old march is removed: scored orders and norms, stencil/history identities,
   proposal before and after caps, controller state, quiet/crossing readings, landing
   reasons, startup and restart events, and terminal failures that occur before an
   attempt row is emitted. Distinguish trial order from stored and control orders.
   For each checked decision, identify which inputs are independently recomputed
   and which are supplied identically for a policy test; copying a native output
   into the expected state must not count as verifying that output.

3. **R3 — High, maintainer decision: retain autonomous reference integration for
   development and accuracy checks (§§2, 5.2, 7.2, 10.1–10.4).** A new response can
   implement `ref.source` before native supports it, but `ref.replay` then has no
   native trajectory for that response. A developer would have to invent a schedule,
   implement a private adaptive loop, or port to C++ before assessing propagation.
   Prescribed schedules remain useful, but alone make tolerance studies, difficult
   transients and debugging a native regression substantially less convenient.

   There is also an existing independent accuracy claim to retain:
   [`test_a_full_adaptive_run_meets_the_reference_s_accuracy`](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/tests/backends/test_backend_ladder.py)
   compares against a tighter reference integration and checks a declared work band.
   Replaying native's own schedule cannot replace that claim. The analytic P0–P2,
   three-level P3 and cross-scheme checks in [testing.md](../testing.md) must remain
   separately identified class-T or analytic checks, not become matched-replay tests.

   **Recommendation and question:** compare a third option: a small, non-production
   adaptive runner that owns NumPy history, calls the retained trial and policy
   functions, and returns fields and diagnostics, without the production service
   machinery. Give it a documented example of developing a response that native
   explicitly refuses, and of comparing independently chosen trajectories within an
   accuracy envelope. What is its incremental maintenance cost beyond the replay and
   decision-check loops already proposed? If it is rejected, specify the equally
   concrete native-independent workflow for these tasks and the independent accuracy
   oracle replacing the current adaptive comparisons. Removing public selection and
   removing this development capability should be separate decisions.

4. **R4 — High: make independent history and observation arithmetic part of the
   slim contract (§§5.2, 6.4, 6.7, 7.2).** Anchoring from native's field *and history*
   is useful for locating a failing trial, but can conceal an incorrect history
   already committed by native. Free-running replay must keep its own derivative
   rows, prediction/acceptance identities, compensation and carrier-frame data;
   its next step must not be repaired by importing native's history. Specify the
   treatment of rejected attempts, restarts, history retention and RK stage reuse.
   Both forms of comparison should be required, rather than left optional in Q6.

   The retained API also needs to name dense evaluation and diagnostic outputs.
   Section 6.4 groups planes, statistics and monitors under exact bookkeeping, but
   interpolated field values and numerical readings are not bookkeeping. In
   particular, replay must evaluate a plane inside a crossing step with the method's
   continuous extension, not insert an extra landed step. Stage-dependent quadrature
   diagnostics and periodic plasma statistics cannot all be reconstructed by treating
   `stats(problem, state, z, ...)` as a stateless accepted-field query. See the
   [dense-plane session tests](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/tests/sessions/test_conformance.py)
   and the [stash-versus-fresh-scan tests](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/tests/backends/test_native_stash_and_saving.py).

   **Requested change:** define the numerical history and trial-result records and
   their pure commit, restart and dense-evaluation operations. Preserve independent
   checks of the numerical values in dense planes, statistics and boundary readings,
   with their existing rounding or accuracy bounds; use exact expectations for event
   order and accounting. Include mutations of a retained history row, a dense weight
   and a cached statistic, with the ordinary endpoint update left correct.

5. **R5 — High: derive replay bounds for the actual variable-step recurrence
   (§§4, 6.7, 7.1–7.2).** `step_budget` and `propagation_growth` are calculators
   requiring a justified operation count and growth bound; they do not establish
   either for an arbitrary Adams history. In particular,
   [`tests/bounds.py::step_evaluations`](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/tests/bounds.py)
   uses equal-step Adams weights. Variable steps can change their conditioning
   substantially, even when each individual growth factor is permitted.

   **Probe:** `integrated_weights` for five predictor nodes
   `[0, -1, -2, -3, -4]` gives absolute weight sums approximately `12.2444` for the
   predictor and `2.34861` for the corrector through the endpoint. Nodes
   `[0, -0.5, -0.75, -0.875, -0.9375]`, corresponding to successive step doublings,
   give `1305.9714` and `167.3738`. This is a coefficient-conditioning example,
   not a claim that an observed run selected that stencil or became inaccurate.

   **Requested change:** state bounds in terms of each actual stencil's weights or
   moments and operand magnitudes, including cancellation in defects, history
   transport, compensation and coupled-flow conditioning. For free-running replay,
   justify propagation of errors in the augmented field/history recurrence, not
   merely the physical field's continuous sensitivity. Explain when a bound becomes
   uninformative and which short cases then qualify the path. The `16 epsilon` and
   `128 epsilon` witnesses are case-specific evidence, not universal TDSE or
   variable-step guarantees. Reconcile §7.2's blanket ban on changing bounds with
   deriving bounds for these new comparisons; retained tests must keep their claims.

6. **R6 — Medium: distinguish exact decisions from floating-point trace values
   (§§4, 6.4, 7.2).** The statement that decision traces are exact today needs a
   platform qualification. The existing
   [`rk_traces.same_trace`](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/tests/backends/rk_traces.py)
   uses exact comparison on Linux and a numerical tolerance elsewhere because the
   corpus's `pow` values came from glibc. Disabling contraction does not make every
   elementary function portable bit for bit. Nor should independently rounded norms
   be required to produce identical threshold decisions on all adaptive cases.

   **Requested change:** distinguish a same-input policy comparison on a qualified
   implementation from a cross-platform stored corpus and from decisions on
   independently computed numerical readings. Keep exact structural decisions and
   ledgers where justified; give scalar proposals and positions a documented rounding
   treatment where needed, and retain deliberate near-threshold cases whose two
   trajectories satisfy an independent accuracy envelope. Do not turn legitimate
   numerical forks into failures or silently widen the arithmetic checks to avoid them.

7. **R7 — High: preserve claims and the normal execution path during test migration
   (§§6.4, 6.7, 7.2, 11).** Counts of tests, assertions and covered branches are useful
   audit information, but do not show that the replacement detects the same failures.
   An arithmetic mutation alone says little about a lost resume, monitor, event-order
   or stale-statistic check. Moving all observation tests to exact expected rows can
   lose a numerical claim even with the same number of assertions.

   Also, exporting each step for the ladder must not be the only native execution
   configuration tested. Returning after every step could miss state or buffering
   bugs in the normal resident call. The current
   [Adams trace tests](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/tests/backends/test_native_march_traces.py)
   compare one-attempt and production-sized calls; the
   [RK trace tests](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/tests/backends/test_native_rk_traces.py)
   exercise bounds of 1, 7 and 256. That distinction should survive.

   **Requested change:** before phase 2b, provide an old-claim-to-new-test inventory,
   including method/formulation, route, response, varying and coupled operators,
   noise, norms, dense output, statistics, monitors, resume and groups. Separate
   deleted-only behaviour from retained behaviour. Run representative numerical and
   lifecycle mutations against both the old and replacement tests, and keep
   uninstrumented session tests. Establish that enabling trace export changes no
   numerical work, landing, restart or retained history. Keep device qualification
   explicit: CPU success alone must not retire reference-dependent GPU checks.

8. **R8 — Medium: keep a native-free reference development lane (§§6.4, 9.2–9.3,
   10.4).** Removing the selectable backend does not require removing the ability
   to import and exercise its numerical library without native. That ability helps
   develop a new model, diagnose a broken native build and guard the reference's
   independence. The default lane's 100 % coverage does not prove that an accidental
   import of native, or a call into its implementation, has not made the oracle circular.

   `poe fetch-wheel` is not a replacement for every contributor workflow:
   [`difference_from`](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/native/scripts/fetch_wheel.py)
   deliberately refuses an ancestor wheel if `native/` or the solprop modules it
   imports have changed, including uncommitted edits. Those are precisely the edits
   a new feature can require. A published wheel also cannot be presumed to implement
   a changed development seam. The open [#570](https://github.com/LupoLab/solprop/issues/570)
   and [#312](https://github.com/LupoLab/solprop/issues/312) add the deployment limits
   already acknowledged in the SEP.

   **Recommendation and question:** retain a lightweight setup and replace the old
   `no-native` execution lane with a smaller native-free library/translation test
   lane. It should run the reference's analytic, trial and development-runner checks
   and enforce the import boundary. Native qualification still needs a compatible
   native build. Document the two workflows separately, and state what a contributor
   changing the seam can do before CI produces a wheel. If native-free development
   is intentionally removed, identify that as an additional decision beyond removing
   `backend="numpy_ref"`.

9. **R9 — Medium: state what remains independent when code is shared
   (§§3.4, 4, 6.1, 7.1–7.2).** Agreement between two calls into the same moved TDSE
   source implementation verifies wiring and placement of the native finish, but
   cannot detect an error inside that shared source. Likewise, two implementations
   consuming the same translated tables do not independently establish the tables'
   physical correctness. This limitation exists today; slimming should make its
   remaining independent checks explicit, not imply that every ladder rung is an
   independent derivation.

   **Requested change:** document the oracle boundary: which arithmetic is separately
   implemented, which tables or policy constants are shared, and which analytic or
   manufactured tests validate the shared parts. Preserve the manufactured TDSE
   source expectations, signs, density and taper scaling, active-band masking and
   edge-term checks after the production move. Keep a guard against production native
   importing the slim reference, and against the reference invoking native to obtain
   its expected numerical result. A mutation to a shared assembly term should fail
   an independently stated expectation even if both callers agree.

10. **R10 — High: define the hook's failure boundary relative to commit
    (§§6.2, 7.1).** Section 6.2 promises that an exception in *any* hook entry point
    abandons the attempt uncommitted. However, `end(accepted)` calls
    `binding.record_trial`, which can raise `TDSEArchiveError`. In the current
    [host Adams march](https://github.com/LupoLab/solprop/blob/63201dca4ab69d21b6d4ae0cd4611c4203ff47b1/src/solprop/backends/host/adams.py),
    `_attempt` commits the accepted state and history before `_record` calls that
    binding method. Native also commits its engine before recording the accepted
    attempt. Thus preserving the existing call ordering and guaranteeing rollback
    for every `end` failure are different contracts.

    **Requested change:** specify the transaction boundary for `begin`, each
    `evaluate`, acceptance, archive recording and reservation release. Distinguish
    failures before commit from failures after an accepted commit; preserve coherent
    field, position, history, logical/atomic ledgers, archive labels and checkpoint
    eligibility in each case. Release reservations even when `end` raises, retaining
    all attempted-call charges and the original exception. Test archive failure and
    interruption at the boundary as well as provider failure. This asks for precise
    lifecycle semantics, not a change to the chosen source hook or TDSE physics.

11. **R11 — Medium: complete the worker's runtime and cancellation contract
    (§§6.2, 6.5, 6.8, 7.1, 11).** A separate process is a sensible way to isolate the
    provider runtime, but the start mechanism must actually give it that isolation:
    a fork of an already initialised native process would inherit its runtime state.
    The text also refers to a worker exceeding "the run's limit", while the current
    `TDSEExecution` declares memory and attempted-call limits, not a wall-time limit.
    A hung atomic call cannot reach the usual between-attempt stop boundary by itself.

    **Requested change:** specify a fresh interpreter start and an import boundary
    that avoids loading solprop-native in the worker, including notebook, GUI and
    spawned scan use. Name the timeout or cancellation policy and who polls it while
    the hook waits; describe cooperative stop versus forced worker termination,
    attempted-call accounting, exception transport, shared-memory ownership and
    cleanup after partial startup or a dead worker. Include shared transport buffers
    and per-process resources in the memory accounting. Clarify the one-thread-budget
    case, where "every other core" would otherwise assign zero threads, and the
    behaviour when solprop's process-wide pool is already initialised. Qualify these
    contracts with manufactured workers on supported platforms; keep the real-provider
    checks as the separate, explicitly scoped evidence already proposed.

### AI review follow-up — 2026-11-12

**R12 — Revised recommendation and maintainer decision: keep `numpy_ref` for now;
consider TDSE on native separately (§§2, 3.1, 6.4–6.5, 9–12).**

This analysis follows extensive discussion between John Travers and GPT-6 (AI),
including the value of independent complete runs, retaining DOPRI5, Adams and RK4IP,
the actual removable execution machinery, and what native already shares or PR #767
moves into shared code. It revises the initial assessment above; the small-runner
alternative in R3 was considered, not selected.

**Maintainer decision.** John Travers has decided to **keep `numpy_ref` as a
selectable backend for now**, preserving independent runs for development,
debugging and numerical verification. He considers that capability valuable
relative to the modest demonstrated code saving and will continue to consider the
longer-term design. This is not a permanent rejection of every future reduction in
reference scope, nor a decision to remove its sessions, groups or batch support.

**Revised AI recommendation.** Retain the existing reference backend and consider
the native TDSE work on its own merits, separately from removing or replacing the
reference executor. Phase 2's removal should not be a prerequisite for delivering
TDSE on native. The TDSE design still needs the execution and qualification
questions below answered; keeping the reference does not approve those details.

#### What the size audit actually establishes

The baseline is [PR #767](https://github.com/LupoLab/solprop/pull/767) at
`1acdffc4f93e8780f9b6aa4f7fae3eca74cfbb2a`, after its binary64 cleanup and
native/reference decoupling. Counting uses that commit's
[`tools/size_report.py`](https://github.com/LupoLab/solprop/blob/1acdffc4f93e8780f9b6aa4f7fae3eca74cfbb2a/tools/size_report.py)
line classification: code excludes docstrings, comment-only lines and blank lines.
These are source counts and inspection, not measured implementation savings or
runtime benchmarks.

The following covers `src/solprop/backends/host/` and
`src/solprop/backends/numpy_ref/`, excluding the TDSE source assembly that would
move to a shared production location. Shared modules elsewhere are not included.

| Remaining responsibility | Code lines | What retention means |
|---|---:|---|
| `host/session.py`, including the preparation wrapper | 419 | Connects independent single-problem execution to the ordinary run interface, observations, output and failure handling; already composes shared session helpers. |
| `host/group.py`, `host/group_session.py`, `numpy_ref/batch.py` | 798 | Additional group and batch execution; a possible narrower future simplification, not required for independent single-problem runs. |
| `host/driver.py`, `host/adams.py`, `host/low_storage.py` | 1,826 | Mixed numerical integration and production orchestration; keeping an independent runner requires much of the behaviour. |
| The other NumPy reference modules, excluding `batch.py` and `tdse.py` | 4,431 | Numerical passes, the Adams engine, backend assembly and helpers; much remains necessary for verification. |
| **Total** | **7,474** | Approximately 14.8k physical lines, not a deletion budget. |

The four session/group/batch modules therefore offer **1,217 code lines of gross
removal** only if their responsibilities are dropped. The single-run session is
part of what makes a reference comparison convenient; removing it would require a
replacement entry point. Further production plumbing could be removed elsewhere,
but a new runner and trace verification add code back. The discussion's estimate of
net savings in the low thousands of code lines is provisional, not a measured
result. It does not substantiate the proposal's 11.5k source-line saving for a design
that retains independent integration.

Single precision, CF4, comparison steppers and registry simplification are already
handled by the preceding cleanup. Admission checks, modal derivations and coupled
flow factor construction move into shared code in #767. Native still needs the
common session/event contracts, preparation machinery and numerical helpers;
deleting the reference cannot count those as savings. The TDSE source assembly is
also a move, not a deletion. Conversely, native already owns its compiled marches
and sessions, so the reference's production orchestration is a real additional
responsibility; it is simply much smaller than the entire reference implementation.

#### Why retention currently has the stronger case

- **Independent complete runs are useful evidence and a practical debugging tool.**
  Running the same problem through the usual interface with `backend="numpy_ref"`
  exercises history accumulation, rejection, restart, observation placement and
  feature interactions. It permits development before native implements a response
  and diagnosis when a native build is broken. Replay and individual trial checks
  remain valuable, but do not by themselves replace that workflow.
- **Keeping the steppers keeps substantial integration logic.** DOPRI5 and both
  Adams formulations need independent defects, norms, acceptance, step selection,
  history and observation arithmetic; Adams also needs order selection. RK4IP keeps
  its current prescribed-step execution and convergence checks. Adding adaptive
  RK4IP would be a separate numerical change. Rebuilding these capabilities in a
  small runner is partly reorganisation, not elimination of maintenance work.
- **The single-run session is useful; group acceleration is more optional.** The
  reference batch class implements another Runge–Kutta stage path across a member
  axis, whereas native groups compose member sessions. A future narrower change
  could compare native group members with independent single-problem reference
  runs, keeping native lifecycle and failure-contract tests. It would still need
  explicit scope and test migration; no such removal is decided here.
- **The CI saving is not an entire second matrix.** At the audited baseline the main
  lanes run native with reference parity checks, and the dedicated native-free lane
  runs a small selection of modules. Most numerical qualification should remain in
  either design. Fewer test lines are not evidence of equivalent verification.
- **Maintenance cost matters more than the project-wide percentage.** A small,
  stateful module can impose substantial recurring work. The strongest case for
  removal would be concrete costly changes caused by maintaining a second execution
  contract. This review has not established a burden that outweighs the existing
  debugging and verification capability, or a replacement that repays its migration
  and qualification costs.

The independence is execution independence over the prepared problem, not a second
derivation of all physics: both backends share translation, tables and some
helpers. Analytic and manufactured checks remain necessary for those shared parts
(R9). Retention therefore preserves the existing complementary checks; it does not
turn backend agreement alone into proof of physical correctness.

#### Consequences for the proposal and the remaining questions

**Drafting follow-up requested:** when revising the proposal, separate the useful
native TDSE work from reference removal, retain the current reference execution and
native-free development support, and revise the motivation, phases and size claims
accordingly. Reconcile ADR-077's promised slim-reference end state with this
maintainer decision without undoing its binary64 or decoupling changes. Those are
requests to the proposal's author; this review update leaves Part I and Part II
unchanged.

- **Q2 remains decided:** TDSE uses native CPU execution; the provider runs in its
  own worker and gets the device or CPU resources under the recorded policy.
- **Q4–Q8 are deferred with reference removal.** Losing reference execution, a new
  slim API, its replay contract, renaming the module and replacing SEP 0's parity
  rules do not need decisions for the retained-backend TDSE scope. Replay or trace
  improvements may still be assessed on their own merits.
- **Q1 decided by John Travers, 2026-11-12:** keep the three callbacks:
  `begin(cost)` reserves the atomic-call budget before the attempt, `evaluate(...)`
  obtains the response and assembles the source, and `end(accepted)` records the
  accepted or rejected outcome and releases the remaining reservation. This follows
  the reference's existing accounting without returning to the session after every
  attempt solely to manage its reservation and outcome.
- **R10 decided by John Travers, 2026-11-12:** the numerical commit is the failure
  boundary. A failure before commit discards the trial and retains the previous
  accepted field, position and history. A failure after commit, such as archive
  recording in `end`, preserves the newly accepted state, stops the run with the
  original error, reports the archive failure explicitly and makes that state
  available for checkpointing where possible. A rejected attempt does not advance
  the accepted state. Every failure path releases unused reservations while retaining
  charges for atomic calls already attempted; cleanup must preserve the original
  error. Numerical rollback cannot undo completed atomic work or archive writes.
- **Q3/R9 decided by John Travers, 2026-11-12:** place the host TDSE source assembly
  in `solprop.backends.tdse_source`, called by both native and `numpy_ref`, without
  native depending on the reference package. The driver retains provider-worker
  ownership, budgets, archives and resource lifetime; translation prepares the
  parameters and tables; the shared module performs source subtraction, scaling,
  gating, transforms and projection at each evaluation. Both backends retain their
  independent optical marches and history. Keep independent analytic/manufactured
  expectations for the shared source's signs, scaling, gating and projection, plus
  separately tested native wiring: agreement between callers of shared arithmetic
  cannot by itself verify that arithmetic.
- **Q9 decided by John Travers, 2026-11-12:** deliver the TDSE scope in one
  implementation pull request under the normal SEP process, opened as a draft when
  implementation starts. Develop the shared source module, provider worker, native
  hook, tests and documentation in successive commits, and mark it ready once the
  complete TDSE path is qualified. PR #765 remains the proposal/review pull request.
  Required Pyroraptor changes use a linked companion pull request in that repository;
  no exception for separate solprop milestone pull requests is needed.
- **Q10 decided by John Travers, 2026-11-12:** do not add a scheduled or manually
  triggered real-provider CI job. Test Pyroraptor by hand as it is used, on CPU or
  GPU as applicable. Retain manufactured-provider and worker-contract tests in CI.
  The reviewer recommendation for an automated CPU check was not accepted. Public
  release of Pyroraptor is not a prerequisite for the TDSE implementation; it may
  remain private. CI therefore establishes the manufactured integration contracts,
  not compatibility with an installed real provider. Report actual manual evidence
  and its configuration separately, without implying unperformed provider checks.

  **Review note on the existing dependency boundary:** solprop declares no
  Pyroraptor dependency or extra and does not fetch a private source during install.
  The TDSE binding dynamically imports an already installed `pyroraptor`, checks
  its reported native source revision against `_PROVIDER_REVISION`, checks its
  execution space, and records its build identity. Keeping the provider private
  does not obstruct ordinary solprop installs or manufactured tests. A newer provider
  build requires deliberate coordination with the source pin and manual evidence;
  CI alone cannot establish real-provider compatibility. The worker implementation
  should preserve optional loading and report the worker's actual provider identity.
- **Q11 decided by John Travers, 2026-11-12:** use pipe-only transport for control
  messages and arrays; do not implement a shared-memory transport. Atomic evaluations
  take at least minutes in the intended use, so the extra copying is expected to be
  negligible relative to the provider work. Transfer array values unchanged, use
  bounded messages and include transport buffers in memory accounting. Keep the
  communication path responsive to cancellation. Shared memory would need a future
  demonstrated benefit to justify its additional ownership and cleanup machinery;
  transport overhead has not been benchmarked in this review.
- **R11 cancellation/time-limit policy decided by John Travers, 2026-11-12:** no
  automatic per-evaluation timeout, and prompt cancellation when explicitly
  requested. The parent checks for cancellation while waiting for the worker; it
  shuts the worker down, terminating it if necessary, without waiting for a long
  atomic evaluation to finish. Discard the unfinished optical trial and preserve
  the last accepted field, position and history for checkpointing, following R10.
  Keep charges for atomic calls already attempted. A user cancellation is an
  interrupted run, not a provider malfunction. A bounded shutdown grace period
  after an explicit stop is distinct from an automatic evaluation timeout.

#### Remaining worker implementation requirements — R11

The following are reviewer requirements for implementing and testing the agreed
process boundary, not additional maintainer policy decisions:

- **Start a fresh interpreter through an isolated worker entry point.** Do not fork
  an initialised native runtime or re-execute user startup code that imports it.
  The provider worker must not load solprop-native, and the optical process must
  not load Pyroraptor. This includes provider build and preset queries
  (`basis_from_preset`, `hacc_preset`) and diagnostic sessions, which currently use
  `_load_provider`; moving only the main evaluation call is insufficient. Exercise
  the entry point from scripts, notebooks, the GUI and spawned scan workers.
- **Make the parent responsible for worker and pipe lifetime.** Cover preparation
  failure, partial messages, provider exceptions, unexpected worker death and
  cancellation during a blocked transfer or calculation. Reap the worker and close
  both ends appropriately; discard incomplete results and a pipe interrupted by
  forced termination. Preserve the primary exception or cancellation reason if
  cleanup also fails. A dead worker is an observable failure even though a live
  calculation has no automatic time limit.
- **Respect the run's resources in both processes.** Apply Q2's provider-priority
  policy within the declared allocation. A one-core budget must give the provider
  one working thread while the optical side waits, not zero threads or an implicit
  requirement for another core. Account for an already initialised native pool
  without reinitialising its runtime; it should wait passively during provider
  work. Include provider plans, per-process buffers and pipe transfer copies in
  memory admission and reporting.
- **Qualify the boundary with manufactured workers in CI.** Test successful
  exchange, failures before and after commit, explicit cancellation of a hanging
  worker, partial startup/transfer, process exit, import isolation and cleanup.
  Use test-harness deadlines to detect broken cancellation without imposing a
  production evaluation timeout. Real-provider evidence remains manual under Q10.

**Question disposition:** Q1–Q3 and Q9–Q11 have decisions recorded above; Q4–Q8 are
deferred with reference removal. R9–R10 and R11's cancellation/time-limit policy
also have maintainer decisions. The implementation requirements still need to be
incorporated into the proposal and verified; this review record does not mark the
SEP accepted or the implementation qualified.

R1–R8 remain recorded as analysis of the proposed replacement and as possible
verification improvements. They are not automatically prerequisites for a TDSE-only
change that retains the existing reference; relevant numerical and lifecycle claims
must still be preserved in TDSE qualification.

### Response to review — 2026-11-12

By the drafting agent, Opus 5.5, to the review and its follow-up above. Part I is
rewritten to the maintainer's decisions; the review record above is unchanged.

- **R12 and the decision to keep `numpy_ref`:** taken. Phase 2 is removed: the slim API,
  `StepTrace`, replay, the parity ladder, the deletions, the re-pointed modules, the SEP 0
  change and the size savings. The SEP is retitled and its file renamed. The deferred
  design and the audit's figures are §10.1; ADR-077's end state is reconciled in §3.4 and
  §8 (§12, P6). §6.8 states additions only.
- **R1–R4, R7 and R8:** concerned the replacement of independent runs; they go with it
  (§10.1). What they ask that still bears on TDSE is kept: §7.3 item 4 runs the native
  march with a one-attempt and a production call bound (R7), and the `no-native` lane
  stays (§9.2, R8).
- **R5 and R6:** taken for the TDSE comparisons. §4 states the budgets as derived for
  these cases, not universal, and near-threshold forks as class T (§12, P5); §7.3 item 4.
- **R9 and Q3:** §6.1 (the responsibilities and import boundaries), §7 (the oracle
  boundary) and §7.1 (independent expectations for the shared source, with defect checks).
- **R10:** §6.3, tested by §7.3 item 3.
- **R11 and its requirements:** the worker's start, environment, transport, handshake,
  failures, lifetime and memory, §6.5; threads, §6.6 (§12, P1); cancellation with no
  automatic timeout, §6.7 (§12, P3); manufactured-worker qualification with test
  deadlines only, §7.2.
- **Q1, Q2, Q9, Q10, Q11:** §6.2, §6.6, §11, §7.4 and §6.5 with §6.7 respectively, and
  listed as decided in §12. Q4–Q8 are dropped with the reference's reduction.
