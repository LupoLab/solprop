# Working on SolpropJulia

Read the root `AGENTS.md`, `docs/dev/architecture.md`, `docs/dev/conventions.md`,
and SEP 48 before changing this package. The accepted SEP defines the scope; these
rules apply to every implementing and reviewing agent in this directory.

## Ownership and interfaces

- Julia receives translated arrays and immutable run policy, never physics objects.
  The engine owns the complete session: stepping, controller, history, monitors,
  statistics, snapshots and export. Python lowers inputs and services events.
- The PythonCall bridge is a package extension. The engine has no Python dependency
  and tests independently. No host loop drives Julia trials or finishes statistics.
- Follow SEP 47's merged preparation/session contract. Do not inherit `StepDriver`,
  import native lowering, duplicate policy on the problem, or add legacy dispatch.
- Admission refuses unsupported routes, responses, methods, precision and options by
  name before allocating state. Internal failures use specific exception types; the
  adapter preserves their cause in `BackendError` or `ConvergenceError`.

## Julia code

- Use explicit imports, concrete or parametric struct fields, and small functions.
  Kernel bodies capture only values and arrays and are generic over the admitted
  scalar and array types. Keep CPU and device behavior in one numerical implementation.
- Use Runic without configuration. Wrap lines at 92 characters manually; Runic does
  not enforce that limit. Aqua and JET error analysis are blocking. `JET.report_opt`
  is advisory; fix straightforward instabilities in changed hot code.
- Document only the public surface (declared `public`, exported, or bridge entry
  points): signature, imperative summary, arguments/keywords, returns and exceptions
  as applicable, with shapes, SI units and valid ranges. Internal helpers use comments
  only when a mathematical choice, protection or measured behavior needs explaining.
- Every code file has the repository's SPDX copyright and GPL-3.0-or-later header.
  Every dependency has a compat bound. GPU packages and PythonCall are weak
  dependencies with extensions. Generate manifests; do not hand-edit them.

## Numerics and memory

- Follow SEPs 25 and 33. Contraction is permitted where intended; no umbrella
  `@fastmath`, approximate-function flags or flush-to-zero. Protect error-free DF32
  primitives from contraction and reassociation except their explicitly required FMA.
- Derive and cite each numerical bound. Distinguish matched-history rounding,
  adaptive accuracy, and exact decisions on identical recorded readings. Preserve
  structural zeros, lossless copies, integer accounting and transport invariants.
- Input noise is binary64 only. All binary32 paths refuse noise; Metal also refuses
  binary64. A DF32 phase guard includes actual replay steps and retained history.
- Preparation may allocate; warmed CPU advancement allocates nothing. Preallocate
  event storage and work buffers, bound their capacity, and list all retained storage
  in the memory ledger. No Python objects, callbacks or logging in numerical kernels.
- Use `@inbounds` only where measured, on the narrowest expression, with a reason.
  Run tests with bounds checking enabled. Test kernel device safety on CPU as well.
- Stop/snapshot requests and exports obey lifetime ownership. Stop access is atomic;
  detach listeners before releasing the owner. A failed advance poisons the session.

## Development and evidence

- Respect the exclusive ownership recorded in `IMPLEMENTATION.md`. Make changes in
  a dedicated worktree; integrate coherent commits into the single SEP 48 branch.
  Do not edit another agent's files without a recorded ownership handoff.
- During iteration run Runic, `GROUP=Core` Julia tests, and directly affected Python
  tests and static checks. Full gates, slow tests and coverage run in GitHub CI.
- Qualify about ten steps or seconds-long windows. Record each benchmark's window,
  hardware, revision, execution identity, thread count and preparation-inclusive time.
  Never infer one vendor's tuning from another's result.
- An independent reviewer checks each implementation package before integration.
  Append actual checks, limitations and next actions to the implementation record;
  complete phase reports belong in SEP Part II. A generated implementation is not
  qualified until integration and the required tests pass.
