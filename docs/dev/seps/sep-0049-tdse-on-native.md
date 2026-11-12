# SEP 49 — The TDSE response on native

| | |
|---|---|
| **SEP** | 49 |
| **Title** | The TDSE response on native |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-12) |
| **Status** | In progress |
| **Review status** | Reviewed |
| **Reviewers** | GPT-6 (AI); John Travers (human) |
| **Type** | Change |
| **Created** | 2026-11-12 |
| **Resolution** | Accepted 2026-11-12 by John Travers, in [#765](https://github.com/LupoLab/solprop/pull/765) |
| **Requires** | [SEP 12](archive/sep-0012-tdse-nonlinear-response.md), [SEP 20](archive/sep-0020-hacc-helium-tdse.md) and [SEP 26](archive/sep-0026-edge-free-tdse-source.md) (the TDSE response and its source); [SEP 39](archive/sep-0039-native-adams-march.md) (the native Adams march); [SEP 47](archive/sep-0047-run-ownership.md) (backend-owned sessions); [SEP 25](archive/sep-0025-rounding-level-reproducibility.md) (agreement to the error bound); ADR-077 and the change it calls for (§3.4) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | In progress: Opus 5.5 (AI), on `feature/sep-0049-tdse-on-native`; phases 0–4, with the manual check of §7.4 done |
| **Manual pages** | [TDSE response](../../guide/media/tdse.md), [Choosing a backend](../../guide/performance/backends.md) |

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
  the worker's process id. The parent refuses a protocol mismatch or an execution space
  other than `TDSEExecution.device` by name, before any evaluation. It pins no provider
  revision: until Pyroraptor has a release, solprop uses whatever provider is installed,
  by default the latest `main`, and records the revision it reports (maintainer,
  2026-11-12). The build record in a run's metadata is this report.
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
2. the handshake refuses a protocol or execution-space mismatch by name, and records
   another provider revision;
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
contracts, not compatibility with an installed provider; the handshake's protocol and
execution-space checks (§6.5) refuse a mismatched provider at run time, and the
provider's revision is recorded with every run.

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
  mismatched protocol or execution space by name, and records the provider's revision;
  manual evidence with its configuration (§7.4).
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
   the optional import kept, no revision pinned (the installed provider is used, by
   default the latest `main`, until Pyroraptor has a release), and the worker's actual
   provider identity recorded (§6.5, §7.4).
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

### Phase 0 — Baseline

- **Scope.** Phase 0, 2026-11-12, Opus 5.5 (AI), branch
  `feature/sep-0049-tdse-on-native`, based on `feature/two-backends-binary64` at
  `1dc64b5f` (ADR-077's change).
- **Delivered.** This record; no code. Commit `d33e8f24`.
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
- **Deviations.** None. The record was kept in a draft beside the SEP until the
  proposal's pull request brought this file to the branch.
- **Handoff.** Phase 1 moves the assembly; the same script then times it again.

### Phase 1 — The shared source

- **Scope.** Phase 1, 2026-11-12, Opus 5.5 (AI), commit `a0a2f7b1`.
- **Delivered.**
  - `src/solprop/backends/tdse_source.py`, moved from `backends/numpy_ref/tdse.py` with
    `git mv`: `atomic_sources`, `atomic_edge_source`, `project_atomic_sources`,
    `FFT_ERROR_CONSTANT` and their private projection, floor and diagnostic helpers,
    unchanged, plus `node_fields` (step 1 of §3.2, moved from
    `NumpyRefBackend._nonlinear_tdse`). The module imports nothing from either backend:
    its four uses of `numpy_ref.passes` (`project`, `to_frequency`, and through
    `node_fields` `to_time` and `synthesise`) are private one-line equivalents for the
    route's one scalar mode on a field grid, and the cubic replacement's polarisation
    is `_cubic_polarisation`, the compiled real Kerr term alone, in the arithmetic and
    order of `passes.kerr_polarisation` and `nonlinear_sources`, and refused by name
    for any other compiled stage.
  - `NumpyRefBackend._nonlinear_tdse` calls `node_fields`, the binding, the two
    assembly functions and the projection; the stage output stays its own.
  - `tests/backends/test_tdse_source.py` (new, 10 tests) and an optional `responses`
    argument of `tests/edge_source_cases.py::make_case`; the four existing modules'
    imports re-pointed, their expectations unchanged.
  - `docs/dev/architecture.md` (the responsibility and the module table) and the
    layers figure (`docs/dev/figures/build.py`, `layers.svg`).
- **Deviations.**
  - The module's entry points keep their moved names (`atomic_sources` with
    `atomic_edge_source`, and `project_atomic_sources`) rather than §5.2's sketched
    `assemble` and `project`, so that the move is pure and the existing tests and their
    expectations run unchanged; `node_fields` is new. Each takes the caller's stage
    scalars (`StageScalars`: density, Kerr density, taper scale) and the projection the
    caller's prefactor, as `numpy_ref` passed them; native's hook will take both from the
    same translation helpers.
  - `numpy_ref` now transforms the modal field to time twice per TDSE evaluation: once
    in `node_fields`, once for its own in-stage absorber term. Re-timed with phase 0's
    script: 0.26 ms (test window, gated), 1.97 ms (production-sized, gated), 1.22 ms
    (production-sized, cubic), no measurable change.
- **Decisions.** A problem whose cubic replacement has no compiled Kerr term (only the
  accumulation stage) keeps a zero polarisation, as `passes.nonlinear_polarisation`
  gave it; manufactured test problems are built that way.
- **Verification** (§7.1). `tests/backends/test_tdse_source.py`,
  `test_tdse_sources.py`, `test_tdse_projection.py`, `test_tdse_edge_free.py`,
  `test_tdse_march.py`, `tests/test_structure_figures.py`,
  `tests/driver/test_tdse_gated_derivative_run.py` and `test_tdse_input_noise.py`:
  485 passed; ruff and basedpyright clean on the changed files; `poe size-check` within
  budget. The new tests state their expectations with NumPy alone: the nodal field of
  one coarse bin, `(2/N) Re(a e^{2 pi i c n / N})` times the synthesis and the taper,
  within `(c_F log2 N + 4) eps` of its norm; the projection's division by the taper
  (`tau = 2` is exact, so bit for bit); the edge array `rho s g' (J - j_1)` within
  `gamma_3`; a source exactly zero off the active band; the complete and remainder
  floors, and a gated source's truncating and edge floors, from the formulas of the
  docstrings and SEP 26 §4.5, within `1e-12` relative (reassociated sums); and an
  import of the module in a fresh interpreter loading neither backend.

  Each named term was then broken, one at a time, in the module, and the five modules
  above run (a one-off script, not committed):

  | defect | first failing test |
  |---|---|
  | `S = J + j_1` | `test_tdse_sources.py::test_each_closure_has_its_own_cubic_and_remainder_definition` |
  | cubic replacement added, not subtracted | the same |
  | density dropped from the current | the same |
  | density dropped from the shadow polarisation | `test_tdse_sources.py::test_cubic_and_remainder_calibrations_are_independent` |
  | Kerr density dropped from the cubic polarisation | `test_tdse_sources.py::test_each_closure_has_its_own_cubic_and_remainder_definition` |
  | temporal collar dropped from the cubic polarisation | the same |
  | projection multiplied by the taper | `test_tdse_source.py::test_the_projection_divides_by_the_taper_scale` |
  | taper dropped from the node synthesis | `test_tdse_source.py::test_node_fields_are_the_taper_times_the_synthesis_of_each_bin` |
  | node synthesis crop off by one bin | the same |
  | gate dropped | `test_tdse_sources.py::test_native_scale_preserves_complete_nonlinearity_after_linear_subtraction` |
  | density dropped from the edge array | `test_tdse_source.py::test_the_edge_array_is_density_scale_and_gate_derivative_of_the_remainder` |
  | `X + E/(i omega)` | `test_tdse_edge_free.py::test_t2a_a_globally_steady_current_cancels_to_the_budget_and_certificate` |
  | `1/(i omega)` of the wrong sign | the same |
  | `1/(i omega)` formed on every bin, DC included | `test_tdse_projection.py::test_periodic_dipole_and_its_current_give_the_same_source` |
  | spectral crop off by one bin | the same |
  | truncating floor halved | `test_tdse_source.py::test_the_complete_floor_is_the_documented_operand_bound` |
  | cancelled operand dropped from the floor | `test_tdse_projection.py::test_cubic_roundoff_floor_counts_the_unscaled_third_order_operand` |
  | third-order operand dropped from the floor | the same |
  | edge transform's constant halved | `test_tdse_source.py::test_the_edge_floors_are_the_documented_section_4_5_bounds` |
  | derivative-error term dropped | the same |
  | division bound dropped | the same |
  | ratio reported below its floor | `test_tdse_projection.py::test_periodic_dipole_and_its_current_give_the_same_source` |

  One variant is invisible and was not made visible: forming `1/(i omega)` on every bin
  but DC. The current factor and the prefactor are exactly zero off the active band by
  construction (`translate.tdse_source.current_numerator`, the prefactor), so the mask
  is defence in depth; `test_the_source_is_exactly_zero_off_the_active_band` holds the
  consequence.
- **Limitations.** Native does not call the module yet (phase 3).
- **Cost.** `tdse_source.py` 937 lines against `numpy_ref/tdse.py`'s 845 (+92:
  `node_fields` and the private transforms); `numpy_ref/backend.py` +3 net; tests
  +308 (new module) and +6 elsewhere.
- **Handoff.** Native's hook calls `node_fields`, the binding, `atomic_sources`,
  `atomic_edge_source` and `project_atomic_sources` with its own stage scalars and
  prefactor, and finishes the stage itself (§4).

### Phase 2 — The provider worker

- **Scope.** Phase 2, 2026-11-12, Opus 5.5 (AI), commits `0f797abb` (`_interrupts`),
  `ea431aa6` (the generic worker), `8f5b64bf` (Pyroraptor in the worker) and
  `63f43686` (the trial's release), then `origin/main` merged (`7f985eb1`, with
  ADR-077's change, #767), and a follow-up commit on the pull request (a fresh
  worker after any failure, the documentation build, small simplifications).
- **Delivered.**
  - `src/solprop/driver/_source_worker.py` (new), the provider-agnostic layer of §6.5:
    `Worker` starts `sys.executable -m solprop.driver._source_worker <adapter> <limit>`
    with `subprocess`, in a session of its own (`CREATE_NEW_PROCESS_GROUP` on Windows),
    with the parent's environment and import path; the worker moves the protocol off
    descriptors 0 and 1, points standard output at standard error, ignores SIGINT and
    SIGTERM, and its reader thread is the watchdog: end of file on the parent's pipe
    exits the process at once (`os._exit`), during a call too. Frames are an 8-byte
    length and a payload; a message is a JSON control frame and one raw frame per array,
    named by dtype (byte order included) and shape; nothing is pickled; every declared
    length is checked against the frame bound before anything is read. Values cross as
    tagged JSON: tuples, mappings, non-finite floats, arrays, and records (a dataclass's
    or a plain object's fields, received as `Record`, with values an adapter computes
    beside them). The handshake carries the protocol version, the worker's process id and
    the adapter's provider identity. Waiting is a reader thread and a queue polled every
    `POLL_INTERVAL` (0.2 s): between polls signal handlers run, a signal a commit holds is
    raised (`_interrupts.raise_held`, new) and the run's `StopFlag` is polled; any
    interruption kills the worker's process group at once. `close` between calls asks
    the worker to exit and kills it after 5 s; every path reaps it and closes both pipe
    ends. `worker_environment` drops the OpenMP wait variables solprop-native set for
    itself (its marker `SOLPROP_NATIVE_WAIT`), keeps a user's own, and adds explicit
    settings.
  - `src/solprop/driver/_pyroraptor.py` (new), Pyroraptor's adapter: `_load_provider`
    (now only ever called in a worker), the build identity, the haCC surface, records
    and objects, the SAE and haCC preparation, plan building and its checks, the preset
    and basis queries, the atomic calls and the consistency diagnostic's temporary plan,
    and the classification of provider exceptions. The manufactured provider of the
    tests is passed to `PyroraptorAdapter(provider)`.
  - `src/solprop/driver/tdse.py`: the binding, `basis_from_preset`, `hacc_preset`,
    `hacc_basis` and `DiagnosticSession` talk to the adapter in a worker
    (`_open_provider`); the queries use a short-lived one. The handshake's identity is
    checked against `_PROVIDER_REVISION` and `TDSEExecution.device` before any request
    (`_build_record`) and is the run's build record, unchanged in form. Budgets,
    reservations, the archive, the validation and normalisation of every result, the
    domain flags and the unqualified-model warning stay in solprop's process, on plain
    records. `prepare_binding` and `DiagnosticSession` take the run's `stop`, which
    `Simulation` passes. A stop, Ctrl-C or SIGTERM during a call leaves the call charged,
    releases the reservation, records the evaluation in the archive as raised with
    classification `interrupted`, and propagates; it is no provider failure. The memory
    ledger gains `transport.solprop` and `transport.worker` (the largest chunk's drive
    twice and its result once, in each process; `tdse_transport.*` in a run's memory
    rows) and the estimate `estimate.transport`, admitted before the worker starts.
  - `src/solprop/backends/host/adams.py`: a trial's reservation is released on every
    exit (`_trial`), a failed release attached to the trial's own error as a note
    (§6.3).
  - Tests: `tests/driver/test_source_worker.py` (21, real workers),
    `tests/driver/test_tdse_worker.py` (8, real workers),
    `tests/source_adapters.py` (the manufactured adapters the worker imports),
    `tests/source_channel.py` (the adapter served in this process through the same
    codec, for the existing tests whose fake providers are built inside the test), two
    failure-boundary tests in `tests/backends/test_tdse_march.py`, one in
    `tests/driver/test_interrupts.py`; the existing binding tests re-pointed from
    `tdse._load_provider` to the in-process channel, their claims unchanged.
  - `docs/guide/media/tdse.md` (the worker, its threads, stopping during a call),
    `docs/dev/architecture.md`, `changelog.d/tdse-provider-worker.changed.md`.
- **Deviations.**
  - Three checks that need the provider's own classes or methods run in the worker, on
    the raw result, and their outcome crosses with it: the result's type is named there
    (`sae.grid_response`, `hacc.total_current_response`, ...), a haCC result's atom,
    basis and gauge are compared there with the model's (the provider's equality), and
    the shadow's `component` is evaluated there for the components the binding reads.
    Everything else of validation and normalisation is the binding's, as §6.5 says. The
    plan checks of preparation and the two memory admissions between model and plans
    (SEP 20, C7) run in the worker too, as part of preparation; a `DriverError` crosses
    by name.
  - `hacc_basis` takes the caller's provider objects, which cannot cross unchanged: they
    travel as records of their dataclass fields, and the worker checks their types by
    qualified name.
  - The archive records an interrupted evaluation as raised with classification
    `interrupted`, its outcome field unchanged (`returned` or `raised`), so the archive
    format does not change. A worker that exits or breaks the protocol is classified
    `worker`, a new value.
  - A stopped TDSE run ends with status `failed` and the message `StopRequested: ...`,
    as every stopped run does (§5.1 says "interrupted or stopped").
- **Decisions.**
  - A worker is never used again after a failure of any kind, as §6.5 says: a
    provider exception the worker reports, a result the binding refuses, a worker exit,
    a protocol error or a cancellation each end it, and the binding's next evaluation
    or diagnostic starts and prepares a fresh one, which must report the same build
    (`_ProviderBinding._ensure_worker`). Decided by the maintainer on 2026-11-12, over
    the implementer's first proposal to keep a worker serving after a reported provider
    exception. A retry after a failed chunk therefore pays a worker start and the
    model's and plans' preparation again; the budget, ledger, archive and records carry
    over. `DiagnosticSession`, which makes one call, does not restart.
  - The thread count is `PYRORAPTOR_THREADS`, the run's `thread_budget().threads`, in
    the worker's environment: Pyroraptor reads it once at import. A device build needs
    no setting.
  - The worker selects its adapter as `module` or `module:factory`, so the tests'
    manufactured adapters are named, not configured by environment.
  - The worker ends with `os._exit` after the adapter has closed its plans and setup,
    skipping interpreter finalisation.
  - The worker runs as `-W ignore::RuntimeWarning:runpy`: `solprop.driver` imports
    the module before `-m` runs it, which runpy warns of; `__main__` delegates to the
    imported module, so the records an adapter receives are its classes.
- **Pyroraptor.** No companion change was needed: its results, monitors, outcomes and
  plan records are frozen dataclasses of arrays and plain values, which cross as
  records; its CPU thread count is already explicit at import (`PYRORAPTOR_THREADS`);
  its atomic calls release the GIL (`nb::gil_scoped_release`), so the watchdog runs
  during a call; and the worker skips interpreter finalisation. No real Pyroraptor
  build was installed here, so no real call was made (§7.4 stays for phase 4).
- **Verification** (§7.2, manufactured providers in real workers unless noted):
  1. evaluations through the worker equal the same manufactured haCC provider served in
     this process, bit for bit: currents, validity, monitors and outcomes, the ledger
     and the whole metadata (`test_evaluations_through_the_worker_equal_the_provider_
     called_directly`); arrays of several dtypes and byte orders cross unchanged;
  2. the handshake refuses a protocol mismatch, a stale revision and another execution
     space by name, the worker reaped;
  3. a provider exception (by type, message, classification and traceback), a worker
     exit (its status and standard error's tail), a closed pipe, a malformed frame, an
     oversized frame, a start-up failure and a preparation failure each end as their
     named failure, the worker reaped and both pipe ends closed; after a provider
     exception the binding's retry runs on a fresh worker
     (`test_after_a_provider_exception_the_retry_gets_a_fresh_worker`); the provider's
     printing on standard output never corrupts a frame;
  4. a hanging call is cancelled by the stop flag, SIGINT, SIGTERM and a SIGINT held by
     `committing()`, each within the poll interval and the kill (a 10 s harness
     deadline); on `numpy_ref` a binding stopped during a call keeps the call charged
     with no failure recorded, and a run stopped during its thirteenth atomic call ends
     with 13 calls attempted, none failed, and a terminal checkpoint at its last
     accepted step (`z > 0`);
  5. a Ctrl-C to the parent's whole process group interrupts the run and spares the
     worker, which the parent then ends; a parent killed with SIGKILL during a hanging
     call leaves no worker (the watchdog);
  6. the worker loads neither `solprop_native` nor `numpy_ref` nor `pyroraptor`, and its
     `__main__` is the worker module, started from a test, from a script that refuses
     to be imported twice and from `python -c`; after a run through the worker neither
     `pyroraptor` nor the manufactured provider's module is in solprop's `sys.modules`;
  7. the transport rows are in the ledger and the estimate, and an admission over
     `memory_limit` is refused before any worker starts;

  and §6.3 on `numpy_ref`: an archive failure in `record_trial` after an accepted
  commit leaves the new state, booked, with nothing reserved; a failed release is a
  note on the trial's own error. Run on the Mac: the new modules, the existing TDSE
  modules of `tests/driver/`, `tests/backends/` and `tests/output/test_tdse_archive.py`,
  `tests/test_structure_figures.py`, `tests/test_seps.py`: 958 passed, 3 skipped, and
  4 failed, all native-only (`test_tdse_march.py`'s native refusals and the shared-RK
  refusal): the editable solprop-native here was built before the merge of `main` and
  refuses to load ("built from other sources than this checkout's native/"); it was not
  rebuilt (phases 0–2 change no native code), so CI decides them.
  `tests/driver/test_interrupts.py`'s native case fails for the same reason. Ruff,
  basedpyright and numpydoc clean on the changed files; `poe size-check` within budget.
- **Limitations.** The isolation from a scan executor's worker process (§7.2 item 6)
  is not tested; Windows paths (`CREATE_NEW_PROCESS_GROUP`, `TerminateProcess` instead
  of a group kill) are written but run only in CI. The worker ignores SIGINT and SIGTERM
  from the start of its `main`, after interpreter start-up; in its own session no
  terminal or group signal reaches it before then. A provider that spawns processes of
  its own outside the worker's process group is not ended by the kill.
- **Cost.** `_source_worker.py` 761 lines, `_pyroraptor.py` 786 (part of it moved from
  `driver/tdse.py`, which went from 2,575 to 2,506); across phase 2, `src/solprop`
  +2,172 −682 and tests +900 −30 (physical lines, `git diff --numstat`, before the
  follow-up commit). By `tools/size_report.py`'s split, the files phases 1 and 2
  touched went from 11,071 to 12,107 code lines (+1,036), 5,613 to 5,950 docstring
  lines and 447 to 493 comment lines; the driver package from 15,296 to 16,290 code
  lines. §6.8 estimated +0.6k source and +0.6k test lines for the worker; the excess is
  the adapter's request handlers and the parent's view of the haCC model beside the
  generic layer, as the binding's validation stayed in solprop's process.
- **Review follow-up** (an independent review of the pull request, 2026-11-12). The
  worker's reader queues a failure for any exception, so a reply it cannot decode (a
  bad tag, nesting too deep, memory) ends the wait as a protocol error instead of
  leaving it polling; the worker's watchdog and its `main` leave through `os._exit` on
  every path, never through interpreter finalisation with the provider loaded; a write
  to a worker that died between calls is the worker's exit, with its status and
  standard error; `close` kills the worker even when a signal interrupts its grace; the
  reader threads hold the queue and stream, not the worker, and a finaliser kills a
  worker its owner drops; a reader still blocked in its pipe leaves the stream
  unclosed rather than blocking `kill`; a stop already asked for owns a worker exit
  seen after it; the worker starts with `-P` and with solprop's BLAS idle defaults
  (`child_environment`); the diagnostic closes its plan only on a live worker.
  `_interrupts.raise_held` acts only inside an `interruptible()` scope on the main
  thread, which native's session will open around its `advance` in phase 3, so any
  other commit, the terminal checkpoint's `committing(deliver=False)` among them,
  keeps or drops a held signal as before. The manufactured hangs are bounded (30 s),
  so a broken cancellation fails a test rather than hanging it; new tests cover the
  undecodable message, the dead worker, the interrupted close, collection, the
  interruptible scope and its absence, another thread, and solprop-native's own wait
  settings (`_wait.py`, read by path).
- **Handoff.** Phase 3 (the native hook) calls `tdse_source` and the same binding; its
  waits are inside `committing()`: the session wraps `advance` in
  `_interrupts.interruptible()` too, and `Worker` raises a held signal there; §7.2 item 4 on native and §7.3 follow with it. The worker's
  `request` raises `StopRequested`, `KeyboardInterrupt` or `RunTerminated` unchanged,
  which the hook stores and the session re-raises (§6.3).

### Phase 3 — The native hook

- **Scope.** Phase 3, 2026-11-12, Opus 5.5 (AI), on the branch after the native
  restructuring (#770, merged in at `42e3b22b`); one commit (below, with this report).
- **Delivered.**
  - C++: `include/solprop_native/host_source.hpp` (new), the generic `HostSource`
    with `begin(cost)`, `evaluate(field, z, purpose, source)` and `end(outcome)`, and
    `HostSourceError`; nothing in it names TDSE or Pyroraptor (P10). The march
    (`adams_march.cpp`) calls `Engine::begin_attempt` after its budget check and
    `end_attempt` after each decision -- after the commit for an accepted attempt --
    through an `Attempt` guard that ends an attempt abandoned when it leaves without a
    decision (a failure, a floor, a diverged replay); a host failure is
    `Category::host`, naming its stage, `none` for `begin`. `ResidentEngine` forwards
    both to the bound source and names each evaluation's position and purpose (the
    startup at `z`, the predictor at `z + dz`). `GeneralState::bind_host_source`
    allocates the hook's two rows once, the field and the source, listed as
    `host_source` in the memory ledger; the stage's entry kernel
    (`StageTerms::combine`) writes the lab-frame field into the field row, unscaled
    and with any noise added, the workspace's stage core forms the in-stage absorber
    as every route does and calls the source from the calling thread outside every
    parallel region, and the stage finish (`StageTerms::finish`) takes the source with
    the collar alone, skipping the engine's own response pass. Host builds only: a device build throws, and its
    adapter refuses the problem first.
  - The binding layer (`bindings.cpp`): `PythonHostSource` calls a Python object's
    three methods under the interpreter lock the march released, handing `evaluate`
    the two rows in place (the field read-only), catches `nb::python_error` there and
    reports it to the march as `HostSourceError`; the state holds the object
    (`State.bind_host_source`), the engine borrows it.
  - Python adapter: `solprop_native/_host_source.py` (new), `TDSEHostSource`: `begin`
    reserves, `evaluate` is `node_fields`, the binding's evaluation and the shared
    assembly and projection with the reference's own scalars (`scalars(problem, z,
    z)`, plain helpers) and prefactor (`prefactor_at` or the combined z table), `end`
    records the outcome and releases in a `finally`; every exception is kept, the
    first as the error and later ones as notes, and a new attempt starts with none.
    `NativeBackend` takes `tdse_binding`, binds the hook for its life (closing the
    state drops it, a leased one too), refuses a TDSE problem without
    its binding (and the reverse) and the statistics the TDSE response does not define,
    records `atomic_calls`, `tdse_execution`, `tdse_last_evaluation` and `tdse_source`
    (now `tdse_source.source_record`, shared with the reference), lists the binding's
    rows as `tdse_*`, and restores an atomic ledger into the binding. The session opens
    `_interrupts.interruptible()` beside its commit around `advance` when a source is
    bound, and reports a host failure as the kept exception, a reservation the budget
    refused rewrapped with the last accepted step's position and field as the reference
    does; a kept exception the march did not stop for, an abandoned attempt's failed
    release, rides on the run's error as a note, as the reference's `_trial` attaches
    it.
  - solprop: the registry hands a TDSE binding to either backend; the driver's
    `numpy_ref`-only rules for a TDSE run (`_admit_tdse_execution`,
    `_check_tdse_execution`) are gone, so `backend="auto"` runs it on native; the
    capability rule `tdse-numpy-ref` is removed.
  - Tests: `tests/backends/test_tdse_native.py` (23); the registry, entry, defaults
    and checkpoint tests that held the old refusals rewritten.
  - Documentation: `docs/dev/kernel-contract.md` (§1's `TDSESource`, the source
    reservations, §6's host source hook), `docs/dev/backends.md`, `AGENTS.md` and
    `native/AGENTS.md` (the hook's exception to "no Python call in a trial step"),
    the TDSE manual and *Choosing a backend*, `changelog.d/tdse-on-native.changed.md`.
- **Deviations.**
  - The march's `end` takes an outcome, `rejected`, `accepted` or `abandoned`, not a
    boolean: an attempt that leaves without a decision must still release its
    reservation, which the reference does in a `finally`, and is not recorded, as the
    reference records none. Three entry points remain.
  - A host failure is the march's `error` reason with a new category, `host`, and its
    stage, rather than a reason of its own: the session already books a failed batch
    and raises its error, which is now the kept exception.
  - A signal taken inside the hook's wait ends the native session as its terminal
    error, the batch booked and the state eligible for its checkpoint, where the
    reference's session lets the signal through to the run loop; the run loop raises
    it either way.
  - §7.3 item 1 reads each engine's history row zero after the first attempt, not
    the stage output alone: native has no stage output outside a run. Without the
    in-stage absorber the initial step is accepted on both, and the row, the startup
    output carried to the step's end in the interaction formulation, is equal bit for
    bit; with it, the collar rejects the initial step, the row is the startup output,
    and it is within one transform's bound, peak-normalised.
  - Item 4 is two runs: eight steps of 2^-27 m without the in-stage absorber (cap and
    length binary-exact, so no final step is split), within the eight-step 128 eps; and, with the absorber, whose collar holds the steps to about a
    two-hundredth of the run, 0.2 um in some four hundred attempts within the
    attempts times the one-step 16 eps. Each checks that the source moves its field a
    millionfold beyond the budget.
  - The memory row is the generic `host_source`, not §6.2's `tdse_host_source`: the
    engine knows no TDSE (P10). It holds §6.2's two `n_omega` rows.
  - The reference records an accepted replayed step after releasing its reservation;
    native's `end` records, then releases, for every decided attempt. The hook is not
    told that a step is replayed, and the two orders differ only when the record or
    the release fails.
  - Item 5 is met through the driver: the TDSE run tests that ask for `auto`
    (`test_tdse_hacc_run.py`, `test_tdse_archive_run.py`, `test_tdse_entry.py`, the
    gated-derivative and natural-output runs) now run on native, unchanged; those
    that name `numpy_ref` keep it. No module was duplicated over both backends.
- **Decisions.** The Runge-Kutta and low-storage paths are not guarded against a
  bound host source in C++: the translation admits a TDSE response with the adaptive
  `Multistep()` only, and a zero-length Runge-Kutta trial with a source bound does not
  give the reference's stage (above), so a later guard should refuse it there.
- **Verification.** Native CPU build (Apple clang, editable, the `estimate` planning),
  Apple M5 Max. `tests/backends/test_tdse_native.py`, its manufactured remainder
  1e-18 A m, strong enough that `F` moves every compared field seven orders or
  more beyond its budget: through the C++ gather and the binding, in both formulations, on
  the gated and cubic descriptors, with and without input noise, the field the hook
  receives and `F` bit for bit, the history's first row as above; the call sequence
  (every reserve, evaluation purpose, record and release, in order), the atomic ledger
  and the `atomic_calls` and `tdse_source` records equal to the reference's on a whole
  run in both formulations, the evaluations' positions within 1e-9 relative and the
  final fields within the per-attempt budget; a budget refusal, a provider failure, a
  SIGTERM before the commit (raised as an exception, and a real signal the commit
  holds, raised by the evaluation's `raise_held` inside the session's
  `interruptible`), and an archive failure after it, each leaving the stated state
  with nothing reserved and the reference's calls, the archive failure's terminal
  state carrying the committed step's proposal; an abandoned attempt's failed release
  as the reference's note on the floor's error; the eight-step and collar-held runs
  within their budgets, with one attempt per call and the production bound (measured
  5e-5 eps against 128 eps, and 5 to 6 eps against 6,480); a device build and a
  missing binding refused by name; the memory rows; closing a backend whose state a
  session leases, after which the binding, held in a cycle through the kept
  exception's traceback, is collected (the state's own reference, which the collector
  cannot see, is dropped once the state is closed). The TDSE, worker and interrupt modules of `tests/backends` and
  `tests/driver`, the compiled-multistep, resume-ledger, capability, structure and
  SEP modules: 1,411 passed, 3 skipped; `native/tests/test_backend.py` and
  `test_session_failure.py`: 61 passed. Non-TDSE runs, before (the #770
  build) and after, `SOLPROP_FFTW_PLAN=estimate SOLPROP_CALIBRATION=off`, one thread:
  the final fields of `gnlse_soliton` (default), `rdw_one_mode` (both Adams formulations), `argon_blueshift`, `radial_filament_256` and `dense_output`, each over 0.5–2 % of its length, are identical bit for bit. `poe fmt-native`; `poe tidy-native` on the changed C++ files:
  no finding. No CUDA or HIP compiler here: the device builds are left to CI's
  device-compile lanes.
- **Limitations.** TDSE on a native device build is refused by name (§12, Q2). The
  hook is not exercised by a real provider (§7.4, phase 4).
- **Open questions.** Q12, closed by the review below: the rows had been read at
  different points (the reference's after an acceptance had carried it, native's
  after a rejection), and at matched points the reviewer found them 1.3e-11 apart. The
  mismatch hid the true one: native's `F` never reached its history.
- **Cost** (phase 3, `git diff --numstat` from `42e3b22b`): C++ +332 −11, native Python
  +251 −12, `src/solprop` +49 −58, tests +464 −30. §6.8 estimated +0.3k C++, +0.4k
  native Python and +0.5k tests for the hook. The review's fixes: C++ +51 −35,
  native Python +25 −4, tests +333 −79.
- **Review** (independent, of `478b3808`; fixed in the following commit).
  - Critical: the binding handed `evaluate` ownerless NumPy views, which nanobind
    copies when it passes them to Python, so `out[:] = source` wrote into a copy and
    native's TDSE stages were the window times zero plus the absorber. The views now
    have an owner, a capsule that frees nothing; the field is read-only.
  - Critical: the stage core gathered the field from the fine spectrum, which holds
    it divided by `n_to` (and doubled on an analytic grid's interior), so the hook saw
    the field divided by 384. The entry kernel now writes the unscaled stage input,
    noise included, into the hook's field row; the stage core reads that row.
  - High: the tests could not see either defect. Item 1 compared the Python adapter
    with itself; the runs' remainder of 1e-27 A m moved the field by 581 eps against
    a budget of 6,480 eps. The tests above replace them.
  - Medium: `NativeBackend.close` unbound the hook first, which a leased state
    refuses; it now only closes the state.
  - Low: a kept exception the march did not stop for was neither taken nor reported,
    and stayed for the next attempt; it is now a note on the run's error and cleared
    at each attempt's start. A failed `end` after an accepted commit returned before
    the step's proposal, count and replay cursor were kept; they now are.
  - Each fix was checked by putting its defect back and running the module: the
    ownerless views fail 14 tests (item 1 in every case, both run budgets, both call
    sequences and the floor's note); the scaled field fails item 1 in all 8 cases
    (the manufactured remainder does not depend on the field, so the runs cannot see
    a scale); the unbind fails the close test; the missing note fails the note test;
    the session without `interruptible` fails the held-signal test; and the lost
    proposal fails the archive test. Non-TDSE runs stay bit for bit as before.
- **Handoff.** Phase 4: the manual check of §7.4 through the worker on both backends,
  the ADR, the remaining pages, and the size comparison.

### Phase 4 — Qualification and documentation

- **Scope.** Phase 4, 2026-11-12, Opus 5.5 (AI): the review's two follow-ups to phase 3,
  the ADR, the remaining pages of §8, the size report against §6.8, and the manual
  check of §7.4, its procedure and its results. The SEP stays `In progress`: it
  becomes `Final` after CI and an A/B.
- **Delivered.**
  - Phase 3 follow-ups:
    - `NativeBackend.close` now drops the host source once the state is closed
      (`bind_host_source(None)`, which a closed state allows). Before, the state kept a
      reference that the garbage collector cannot see. A cycle through it (a kept
      exception's traceback reaching the session) was never freed, and the held-signal
      test leaked eight nanobind instances.
    - That test now checks, with a weak reference after `gc.collect()`, that the
      binding is freed, and it fails without the fix. A session that could not close
      keeps the source it still uses.
    - Item 4's eight-step run actually took nine attempts, all accepted: 8e-8 m in
      10 nm steps left a last step one ulp over the cap, which the march split in two.
      The cap and length are now binary-exact (2^-27 m and 2^-24 m), so the run is
      eight accepted steps. The comment no longer counts the startup as an attempt.
  - ADR-078 (`decisions.md`) covers:
    - the host source hook, its scope and the device refusal;
    - the provider worker;
    - the failure boundary;
    - `auto`.

    It amends ADR-077's "every feature except the TDSE response, for now", and
    ADR-077's status now says so. The maintainer accepted it on 2026-11-12.
  - No provider revision is pinned (maintainer, 2026-11-12): until Pyroraptor has a
    release, solprop uses whatever provider is installed, by default the latest `main`.
    `_PROVIDER_REVISION` and `pinned_revision` are gone, and the handshake no longer
    refuses another revision. It still refuses another protocol version, and an
    execution space other than `TDSEExecution.device`. The build record keeps the
    revision, build and execution space the worker reports (no `source_pin`); a
    preset's or a basis's `source` names the reported revision, the consistency cache
    keys on it (`tdse.provider_revision`), and the archives' contexts drop
    `provider_pin`. Tests now check that another revision is recorded, not refused;
    the installed-provider test runs with any revision; a release note says so.
  - `testing.md` gains *Shared code and out-of-process providers*: the oracle boundary
    for shared code and the worker-contract tests. `backends.md` and
    `kernel-contract.md` now cite ADR-078.
  - The other pages of §8 were done in phases 2 and 3:
    - the TDSE manual and *Choosing a backend*;
    - `backends.md`, `kernel-contract.md` §6 and `architecture.md`;
    - `AGENTS.md` and `native/AGENTS.md`;
    - the release notes: `tdse-on-native.changed.md` for `auto`, and
      `tdse-provider-worker.changed.md` for the worker and cancellation during an
      atomic call.
- **Size** against §6.8. Lines are `git diff --numstat` from the tip of #770
  (`1f00b8f7`, which contains `main`), so #770's own changes are not counted. Code lines
  are from `tools/size_report.py`.

  | part | `src/solprop` | native Python | C++ | tests |
  |---|---|---|---|---|
  | estimate, total | +0.6k | +0.4k | +0.3k | +1.3k |
  | lines | +2,448 −784 | +280 −12 | +349 −11 | +2,160 −66 |
  | code lines | +1,052 | +166 | — | — |

  By part:
  - The shared source is a move: +34 code lines with `node_fields`, and its tests +0.3k.
  - The worker and adapter are `_source_worker.py` (522 code lines) and `_pyroraptor.py`
    (605), less 136 from `driver/tdse.py`: about +1.0k code lines, which is the
    `src/solprop` overrun. Their tests are +1.0k.
  - The hook is +166 native Python code lines and +349 C++ lines, with tests of +0.8k.

  The worker is larger than §6.8 estimated. The binding's validation and normalisation
  stay in solprop's process, as §6.5 asks, while the requests the parent's view of the
  haCC model makes must be served in the worker. Phase 2's report records what was
  examined for removal.
- **The manual check of §7.4: procedure.** With Pyroraptor's latest `main` (no revision
  is pinned), installed into solprop's environment.

  1. Set up the environment on the CPU. On macOS this needs Homebrew LLVM, `libomp` and
     FFTW; on Linux, `CXX=clang++`, as Pyroraptor's installation guide says.

     ```sh
     cd <solprop checkout>   # this branch
     uv sync                 # solprop and its native CPU build
     git -C <pyroraptor> switch main && git -C <pyroraptor> pull
     uv pip install --python .venv/bin/python <pyroraptor>   # its CPU (OpenMP) build
     uv run --no-sync python -c "import pyroraptor as p; print(p.build_info())"
     ```

     From here on, use `uv run --no-sync`: a plain `uv run` or `uv sync` removes
     Pyroraptor.
  2. On a GPU machine, use the same solprop environment with its native CPU build (a
     native device build refuses TDSE). Install Pyroraptor's device build into it in
     place of step 1's, configured as its `scripts/device_build.sh` does:

     ```sh
     cd <pyroraptor>
     kokkos="$PWD/build/sources/kokkos-5.2.1"
     test -f "$kokkos/CMakeLists.txt" || git clone --depth 1 --branch 5.2.1 \
         https://github.com/kokkos/kokkos.git "$kokkos"
     # CUDA
     arch=$(scripts/device_build.sh cuda arch auto)
     NVCC_WRAPPER_DEFAULT_COMPILER=g++ CMAKE_ARGS="-DPYRORAPTOR_EXECUTION_SPACE=CUDA \
         -DPYRORAPTOR_DEVICE_ARCH=$arch -DCMAKE_CXX_COMPILER=$kokkos/bin/nvcc_wrapper \
         -DFETCHCONTENT_SOURCE_DIR_KOKKOS=$kokkos" \
         uv pip install --python <solprop checkout>/.venv/bin/python --reinstall .
     # or HIP
     arch=$(scripts/device_build.sh hip arch auto)
     CMAKE_ARGS="-DPYRORAPTOR_EXECUTION_SPACE=HIP -DPYRORAPTOR_DEVICE_ARCH=$arch \
         -DCMAKE_CXX_COMPILER=$(command -v hipcc) \
         -DFETCHCONTENT_SOURCE_DIR_KOKKOS=$kokkos" \
         uv pip install --python <solprop checkout>/.venv/bin/python --reinstall .
     ```

  3. Save this script as `sep49_check.py` in an empty directory outside the checkout:

     ```python
     """SEP 49 section 7.4: a real Pyroraptor through the worker, native against numpy_ref.

     python sep49_check.py compare DEVICE   one short haCC(1) fast segment on both backends
     python sep49_check.py cancel DEVICE    native, Ctrl-C (SIGINT) during an atomic call
     python sep49_check.py refuse DEVICE    native, a device the build does not have: refused

     DEVICE is cpu, cuda or hip, matching the installed Pyroraptor build (for refuse, one
     it does not match).
     """

     import json
     import os
     import signal
     import subprocess
     import sys
     import threading
     import time
     import warnings
     from pathlib import Path

     import numpy as np

     from solprop.driver import Simulation, tdse
     from solprop.driver.tdse import hacc_preset
     from solprop.exceptions import BackendError, ResponseBudgetError
     from solprop.nonlinear.tdse import (
         TDSEConsistency,
         TDSEExecution,
         TDSEGasResponse,
         TDSESampling,
     )
     from solprop.output import open_file

     LENGTH = 2.0**-24  # one step of 60 nm: a startup and a predictor call, then a retry
     BUDGET = 3  # atomic calls, at most (SEP 49 section 7.4)
     RTOL = 0.1  # loose: the one step is accepted, a check of the plumbing, not accuracy
     DELAY = float(os.environ.get("SEP49_DELAY", "5"))  # seconds into the first call


     def simulation(backend, device, budget, length):
         preset = hacc_preset(1, "fast")
         response = TDSEGasResponse(
             "helium",
             preset.basis,
             preset.max_atomic_step,
             "tdse_native",
             1.0,
             preset.calibration(),
             TDSESampling(4),
             model="hacc",
             # No weak-field consistency call: the budget is the march's alone.
             consistency=TDSEConsistency(enforce=False),
         )
         return Simulation.from_capillary(
             256e-6,
             length,
             "He",
             0.1,
             wavelength=800e-9,
             wavelength_limits=(400e-9, 1.6e-6),
             trange=40e-15,
             fwhm=6e-15,
             energy=20e-6,
             modes=1,
             radial_rule="field_scale",
             n_r=2,
             noise=None,
             n_save=2,
             rtol=RTOL,
             dz_init=LENGTH,
             dz_max=LENGTH,
             backend=backend,
             tdse=response,
             tdse_execution=TDSEExecution(device=device, max_atomic_calls=budget),
         )


     def run(backend, device, budget=BUDGET, length=LENGTH):
         path = Path(f"sep49-{backend}-{device}.h5")
         path.unlink(missing_ok=True)
         started = time.monotonic()
         with warnings.catch_warnings():
             warnings.simplefilter("ignore")
             try:
                 simulation(backend, device, budget, length).run(path, progress="none")
             except ResponseBudgetError as error:  # the budget ends the segment
                 print(f"{backend}: {error}")
             finally:
                 print(f"{backend}: {time.monotonic() - started:.1f} s", flush=True)
         return path


     def summary(path):
         with open_file(path) as output:
             outcome = output.outcome
             plain = lambda value: json.loads(json.dumps(value, default=str))  # noqa: E731
             return {
                 "status": outcome.status,
                 "z": float(outcome.z),
                 "atomic_calls": plain(outcome.atomic_calls),
                 "logical_calls": plain(outcome.logical_calls),
                 "provider": plain(outcome.execution["tdse"]["build"]),
                 "fields": np.asarray(output.fields(convention="fft")),
             }


     def workers():
         """This process's provider workers and their CPU seconds."""
         table = subprocess.run(
             ["ps", "-A", "-o", "pid=,ppid=,time=,command="],
             capture_output=True,
             text=True,
             check=True,
         ).stdout
         found = {}
         for line in table.splitlines():
             pid, ppid, cpu, command = line.split(None, 3)
             if int(ppid) == os.getpid() and "solprop.driver._source_worker" in command:
                 parts = [float(part) for part in cpu.replace("-", ":").split(":")]
                 found[int(pid)] = sum(v * 60**i for i, v in enumerate(reversed(parts)))
         return found


     def compare(device):
         results = {name: summary(run(name, device)) for name in ("numpy_ref", "native")}
         reference, ours = results["numpy_ref"], results["native"]
         for name, result in results.items():
             print(
                 name,
                 result["status"],
                 result["z"],
                 result["atomic_calls"],
                 result["logical_calls"],
             )
         print("provider:", reference["provider"])
         if reference["logical_calls"]["accepted_steps"] == 0:
             print("no step was accepted, so nothing to compare: lower LENGTH")
             return 1
         first, last = reference["fields"][0], reference["fields"][-1]
         gap = np.linalg.norm(ours["fields"][-1] - last)
         difference = gap / np.linalg.norm(last)
         print(f"final field, native against numpy_ref: {difference:.3e} relative")
         print(
             f"... and against the segment's change: {gap / np.linalg.norm(last - first):.3e}"
         )
         keys = ("status", "z", "atomic_calls", "logical_calls")
         same = all(ours[key] == reference[key] for key in keys)
         print("same status, position and call ledgers:", same)
         return 0 if same and difference <= 1e-9 else 1


     def cancel(device):
         sent = {}
         output = Path(f"sep49-native-{device}.h5")
         output.unlink(missing_ok=True)

         # The binding's evaluation in this process is where an atomic call is in flight
         # (a private name: this script checks solprop, it is not an API).
         calling = threading.Event()
         evaluate = tdse._ProviderBinding.evaluate

         def marked(self, *args, **kwargs):
             calling.set()
             return evaluate(self, *args, **kwargs)

         tdse._ProviderBinding.evaluate = marked

         def watch():
             calling.wait()
             time.sleep(DELAY)
             sent.update(worker=next(iter(workers()), None), at=time.monotonic())
             os.kill(os.getpid(), signal.SIGINT)

         threading.Thread(target=watch, daemon=True).start()
         try:
             # Many steps, so that a call is in flight whenever the signal comes.
             run("native", device, budget=1000, length=1000 * LENGTH)
         except KeyboardInterrupt:
             stopped = time.monotonic() - sent["at"]
             print(f"KeyboardInterrupt {stopped:.2f} s after SIGINT, during a call")
         else:
             print("the run finished before a call was interrupted: raise LENGTH")
             return 1
         alive = sent["worker"] in workers()
         print("worker", sent["worker"], "still running" if alive else "gone")
         result = summary(output)
         print("provider:", result["provider"])
         print(
             result["status"], result["z"], result["atomic_calls"], result["logical_calls"]
         )
         charged = result["atomic_calls"].get("attempted", 0) >= 1
         reserved = result["atomic_calls"].get("reserved")
         return 0 if not alive and charged and reserved == 0 else 1


     def refuse(device):
         try:
             run("native", device)
         except BackendError as error:
             print("refused:", error)
             return 0 if "device=" in str(error) else 1
         print(f"device={device!r} was not refused")
         return 1


     if __name__ == "__main__":
         mode, device = sys.argv[1], sys.argv[2]
         sys.exit({"compare": compare, "cancel": cancel, "refuse": refuse}[mode](device))
     ```

  4. Run it from that directory. On the CPU build, `compare cpu` and `cancel cpu`; on a
     GPU build, `refuse cpu` and `cancel cuda` (or `hip`), and `compare` only on a GPU
     whose FP64 rate makes an atomic call short (not an A40).

     ```sh
     uv run --no-sync --project <solprop checkout> python sep49_check.py compare cpu
     uv run --no-sync --project <solprop checkout> python sep49_check.py cancel cpu
     ```

     `compare` runs one 60 nm step of haCC(1) fast at about 3e13 W/cm², with two radial
     nodes, through the worker: first on `numpy_ref`, then on native. Each run makes at
     most three atomic calls: a startup, a predictor and one retry.
     - It passes (exit status 0) when both runs end with the same status, position and
       call ledgers, and native's final field is within 1e-9 of `numpy_ref`'s,
       relative.
     - It also prints the difference against the segment's own change.

     `cancel` runs native over a longer segment and sends itself SIGINT 5 s
     (`SEP49_DELAY`) into its first atomic call. It passes when:
     - the run raises `KeyboardInterrupt` within about a second;
     - the worker is gone and nothing is reserved;
     - the file records the call as charged and the last accepted step's position.

     `refuse` asks for a device the installed build does not have (`cpu` with a CUDA
     build, say) and passes when the run is refused by name.
  5. Record here, for each run:
     - the provider revision and build (the `provider:` line);
     - the machine;
     - the threads (`SOLPROP_THREADS` or the machine's) or the device;
     - both modes' output and exit status.

  Both modes were dry-run on an Apple M5 Max. The worker served the manufactured haCC
  provider of `tests/hacc_cases.py` in place of Pyroraptor
  (`solprop.driver.tdse._ADAPTER = "source_adapters:manufactured_tdse"`).
  - `compare` took one accepted step on each backend with two atomic calls. The ledgers
    were the same, and the final fields were 1.3e-13 apart, relative.
  - `cancel` was run with every call after the second hanging
    (`MANUFACTURED_HANG_AFTER=2`). It raised `KeyboardInterrupt` 0.11 s after the
    signal: the worker was gone, three calls were charged, none were reserved, and the
    run stood at its one accepted step.

  This checks the script and the plumbing, not Pyroraptor.
- **The manual check of §7.4: results** (2026-11-12, run by Opus 5.5 at the
  maintainer's request, on a shared HPC cluster through Slurm). Done with this
  procedure's script, and with `refuse` on the GPU.
  - **Code.** solprop `7a5ca1f9a7e25c28007c8ee4768f834b6ffa4d51` (this branch).
    Pyroraptor `7658b615ef99e169d30ff0b12526d2a8d8b638a8` (the latest `main`; reports
    `source_revision='7658b61'`, version `0.0.1.dev26+g7658b615e`).
  - **CPU builds.**
    - solprop's native CPU wheel: LOCAL profile, Release, GCC 16.2, FFTW 3.3.11 with
      OpenMP threads, OpenMP OpenBLAS, `-march=znver3`.
    - Pyroraptor's OpenMP build: GCC 16.2, LOCAL, `x86-64-v3`, Kokkos 5.2.1. It was
      installed into the same environment, so the worker ran the same interpreter.
  - **CPU machine.** A node with two AMD EPYC 7763; a Slurm job with 32 cores
    (`-c 32 --hint=nomultithread`). The worker reported `requested_threads=32`, the
    run's whole budget.
  - **`compare cpu`**: exit 0, wall 1,422 s.
    - Both backends ended `complete` at z = 5.96e-8 m (2^-24 m). Each took one
      accepted order-1 step: a startup call and a predictor call, 2 atomic calls,
      none reserved. The call ledgers were equal.
    - Native's final field was 4.1e-13 from `numpy_ref`'s, relative, and 1.4e-8 of
      the segment's own change.
    - Each backend's run took 707 s (`numpy_ref`) and 712 s (native). About 8.4 min of
      that was the provider's model preparation in the worker; each atomic call took
      about 100 s at 32 cores, below the 10 min bound set beforehand. The problem
      therefore did not need reducing.
  - **`cancel cpu`**: exit 0, wall 550 s.
    - SIGINT came 5 s into the first atomic call (after the same preparation), and
      `KeyboardInterrupt` was raised 0.13 s later.
    - The worker was gone. The file records status `failed` at z = 0, with the startup
      call charged (`attempted` 1) and nothing reserved.
  - **GPU builds.**
    - Pyroraptor's CUDA wheel: `PYRORAPTOR_EXECUTION_SPACE=CUDA`, `AMPERE86`, CUDA
      12.8.1 with `nvcc_wrapper` over system GCC 11.5, Kokkos 5.2.1. It was compiled on
      the login node with 4 jobs, in 26 min.
    - It was installed into a second environment beside the same solprop native CPU
      wheel, whose `execution_space` is `OpenMP`.
  - **GPU machine.** A node with NVIDIA A40s; a Slurm job with one A40, 16 cores and a
    20 min cap. The worker reported `execution_space='Cuda'`.
  - **`refuse cpu`**: exit 0, wall 9.4 s. The run was refused by name ("TDSE requested
    device='cpu', but the native build uses 'Cuda'"), and no worker remained.
  - **`cancel cuda`**: exit 0, wall 681 s.
    - The provider's model preparation on the A40 took about 11 min. SIGINT came 5 s
      into the first atomic call, and `KeyboardInterrupt` was raised 0.47 s later.
    - The worker was gone, and the device released: `nvidia-smi` listed no compute
      process and 0 MiB used afterwards.
    - The file records the startup call charged (`attempted` 1) and nothing reserved.
  - **No GPU `compare`.** The A40's FP64 rate is about 1/64 of its FP32, so an A40
    atomic call could take hours, and SEP 49 does not change Pyroraptor's GPU
    numerics. The GPU numerical comparison of native against `numpy_ref` is to be done
    by hand when the provider is next used on a production GPU (Q10).
  - **Logs.** The build logs, and both runs' logs with every command's output, are
    kept with the maintainer's campaign records.
- **The wheel's strict-FMA check.** On the merge with `main`, CI's Linux wheel failed
  `native/ci/check_strict_fma.py` with 36 fused multiply-adds "in" `march::reached`.
  The disassembly of that wheel shows `reached` itself (0x5d bytes, its symbol's size)
  with none. The 36 lie in 9 KB of unlabelled code that the link-time optimisation
  placed after it: Kokkos kernels of the dispersive response, which call
  `GeneralDispersive::respond`, `sin`, `tanh`, `pow` and `exp`. The wheel is stripped,
  so `objdump` labels only exported functions and counted that code as `reached`'s.
  No strict file includes a contracting header, and no strict code changed. The check
  now bounds each strict function by its symbol's size from `objdump -T` and `-t`,
  with a test. It passes on that wheel (333 strict functions, none fused). The
  non-TDSE runs stay bit for bit as before.
- **Verification.** `tests/backends/test_tdse_native.py`: 23 passed. The held-signal test
  no longer produces nanobind's leak report, and with the unbind removed it fails its
  weak-reference check. The non-TDSE bitwise check is unaffected, because phase 4
  changes the native extension's comments only.
- **Handoff.** Next are CI and an A/B; then `Final`. The GPU numerical comparison stays
  manual, for the provider's next production GPU use (Q10).

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
- 2026-11-12: In progress: Opus 5.5 (AI) on `feature/sep-0049-tdse-on-native`, based on
  #767's branch; phases 0 and 1 reported in Part II.
- 2026-11-12: Phase 2 (the provider worker) reported in Part II; implementation pull
  request opened as a draft.
- 2026-11-12: Phase 3 (the native hook) reported in Part II.
- 2026-11-12: Phase 4 reported in Part II, with the manual check of §7.4 on the CPU
  (compare and cancel) and on an A40 (device refusal and cancel). ADR-078 accepted by
  John Travers. The
  maintainer decided that solprop pins no Pyroraptor revision until Pyroraptor has a
  release: the installed provider is used, by default the latest `main`, and its
  reported revision recorded; §6.5, §7.2, §7.4, §9.3 and Q10 revised to match.

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
