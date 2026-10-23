# Scans and batches

Scans are implemented above the backend seam. A point has a permanent zero-based index in
C order. Executors change where that point runs; they do not change its physics, seed,
backend or place in the output. Seeds derive from a `base_seed`, retained fields are
explicitly selected planes, and different output shapes have separate layouts. Lockstep
batching is first class for `numpy_ref`, `numba_cpu` and `native`. Scans preserve the
selected backend and reject unsupported contracts explicitly.

The executors (`solprop.driver.executors`) are `Local` (sequentially, in the calling process), `Pool`
(spawned worker processes), `Batch` (fixed-schedule lockstep groups on one backend,
[backends.md §1](backends.md)), and the durable file `Queue` with its `Slurm` emitter,
which runs queue or chunk modes as array tasks.

## Why the standard library

The comparison weighs, in order, additional dependencies, resource control, fresh
interpreters, Slurm arrays, macOS development and deterministic collection. This is an
architectural selection, not a benchmark of libraries that are not installed.

| Candidate | Fit and cost | Decision |
|---|---|---|
| Standard-library spawned processes | Explicit process lifetimes, import-time environment, per-worker initialisation and ordinary pickle; resource probing and file-queue ownership remain our responsibility. Global indices make completion order irrelevant to storage. | Use. The required scheduling graph consists of independent points. |
| loky / joblib | Reusable processes and cloudpickle broaden callable support, particularly interactive functions. Neither removes the need for our resource policy, durable point records or shared-filesystem recovery. | Additional callable convenience does not justify a runtime dependency. |
| Dask Distributed | A scheduler, workers and memory management support distributed task graphs and shared intermediate results. Propagation allocations are largely unmanaged native memory, and points have no graph dependencies. | Reconsider for distributed workflows requiring data locality or a scheduler beyond the file queue. |
| Ray | Distributed tasks and actors have logical resource scheduling; resource declarations are not hard memory or CPU isolation. Adds a runtime and deployment layer. | Independent simulations do not require actors or its object store. |
| Parsl | Execution providers integrate workflow tasks with cluster allocations. Useful for multi-stage scientific workflows. | Slurm arrays around independent point controllers are sufficient here. |

Primary references: [Python processes](https://docs.python.org/3.14/library/multiprocessing.html),
[futures](https://docs.python.org/3.14/library/concurrent.futures.html),
[loky](https://loky.readthedocs.io/en/stable/),
[Dask memory](https://distributed.dask.org/en/stable/worker-memory.html),
[Ray resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html),
and [Parsl](https://parsl.readthedocs.io/en/stable/).

No dependency is added. Pool builders and reducers must be importable, pickleable
callables; a module-level function with a lazily cached base Simulation is the
recommended pattern. Live Simulation objects and native backend instances are not
sent to workers. Local execution also permits closures.

## Execution and memory

The layers are Slurm array, node controller, worker, and optionally a backend batch.
The controller owns the collected HDF5 file. A worker receives point descriptions
and returns statistics and reductions; retained fields travel only when requested.
Every spawned interpreter inherits its thread environment before importing NumPy,
FFTW or a backend. Linux affinity respects the allocation's allowed CPU set.

The first-point probe records process peak RSS after backend construction and the
backend's named allocation ledger. RSS already includes resident backend buffers:
the worker estimate uses the larger of RSS and baseline RSS plus ledger bytes,
rather than adding the whole ledger to post-construction RSS a second time.
Twenty percent of the effective memory limit is reserved, together with the
controller's RSS. The effective limit is the smallest declared machine, cgroup,
Slurm and user limit. This is admission control, not a portable hard RSS limit.

## Recovery and storage

Common records are content-addressed and stored once. Point records reference those
records and retain variable metadata, execution provenance and the resolved seed.
Different output layouts have their own arrays and global-index mappings. A point's
results are staged and checked before its status becomes done; failed and unfinished
points are masked by readers. Periodic statistics use exact CSR values and offsets.

Queue ownership follows the refreshed per-point leases of
[Luna PR 441](https://github.com/LupoLab/Luna.jl/pull/441): elapsed run duration alone
does not establish abandonment. An ownership token prevents a late worker from
publishing over a replacement claim. The completion marker precedes queue removal.

## Configuration hashes

Every point keeps the exact single-run configuration key, so a scan point and the same
run started alone are the same run. Common metadata and its content digests are
deduplicated in storage, but the complete per-point SHA-256 is still computed and
checked, and its cost is part of scan timing. Hashing the common part once cannot save
that cost: a SHA-256 over a common suffix cannot be reused once the preceding variable
bytes change, and no second identity contract is introduced to work around it.

## Durable executor implementation

The queue manifest is immutable. A persistent `flock` file serialises state updates;
its inode is never removed. Active point rows contain a controller UUID, host, PID,
process creation identity, claim token and refreshed wall-clock lease. Local liveness
wins over elapsed age; remote takeover requires lease expiry. The controller heartbeat
runs independently of the workers. Artifacts are written to unique private files,
closed and fsynced, then atomically renamed only after the claim token is rechecked.
The state records their SHA-256 and exact single-run configuration hash. Completion
is durable before active state removal. A crash between rename and state publication
can cause recomputation; it cannot authorise a superseded worker's result.

Collection has an exclusive output lock, with shared locks for scan readers. Each
artifact is validated under the queue publication lock against the captured row;
concurrently changed claims are left for the next collection. Out-of-order results
enter the recoverable staging/CSR path. Repeated collection of unchanged
points and controller reports avoids rewriting layouts or appending duplicate audit
rows. Queue files contain portable data; only the locally supplied builder uses
ordinary process serialisation.

The Slurm emitter exports resource settings before scientific imports, quotes paths
and directive values, and divides CPU affinity and memory across task instances.
Actual device IDs come from the scheduler's assignment. Local task-body and golden
script tests exercise queue/chunk modes without submitting jobs. The two-controller
filesystem test is qualified on a local ext4 filesystem, and local array runs pass in
queue and chunk modes; a remote NFS mount and cross-host clock behaviour are not
qualified.
