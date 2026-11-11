# Shared device march integration

This internal slice attaches execution-owned storage to the existing binary64
linear flow and RK march. Public device selection stays closed. Its source and
host tests establish orchestration behavior; actual CUDA tests, unfiltered vendor
quality and complete physical sessions remain separate requirements.

## Execution and retained storage

`LinearFlow` keeps the CPU exponent/failure layout unchanged. A device binary64
flow validates its input owner before allocating its exponent, per-column failure
scratch and optional factor tables through the prepared execution. Its failure
scratch is a `DeviceReadback{...}`: numerical work writes one integer per column,
then the host inspects its persistent mirror in the original first-column,
first-row failure order. No numerical array is downloaded for that check.
Zero-distance table hits use the context-owned copy operation too.

`RungeKuttaWork(state,tableau)` remains the CPU constructor. Its new overload
`RungeKuttaWork(execution,state,tableau)` uses that same path for CPU execution.
Device binary64 preparation owns every stage/field buffer through that execution
and retains a `DeviceReadback` for the existing `(4,n_columns)` norm partials.
Device binary32 work refuses before scratch allocation; its scaled tolerance
construction requires separate integration and qualification.

The binary64 reduction graph is unchanged: each column first computes scaled
old/candidate sums of squares, the host merges the completed partials and forms
the common relative scale, then columns compute defect partials for a second
host merge. There is no vendor sum, field-sized host oracle or Python trial loop.
After candidate services, the same partial storage reports whether each column
is finite, preserving the existing finished-candidate failure before commit.
The prepared execution identity is checked before any partial writes.

The recursive ledger sees each device array and its persistent host mirror.
For two-stage RK on binary64 state with `N=n_omega*n_columns`, the trial payload
is `8N*sizeof(ComplexF64) + 8*n_columns*sizeof(Float64)`: six field buffers, two
stage planes and two copies of four norm rows. This excludes the independently
owned accepted field, source, flow, preparation and vendor/runtime caches. CPU
keeps only one set of norm partials, with no additional mirror. The warmed CPU
advance allocation contract remains unchanged.

## Numerical and lifecycle witnesses

Host tests use a deliberately labelled execution model to count transfers and
allocation owners. They cover compact failure ordering, nonfinite entries outside
the norm mask, mismatched execution before writes, the device binary32 refusal,
cached/zero-distance flows and exact ledger bytes. They do not qualify GPU code.
Existing CPU RK, scaled binary32 norm, source, flow-table and session tests remain
in the affected regression set, including warmed advancement allocations.

The explicit CUDA test disables scalar indexing in its own process. It advances
one and eight fixed Heun steps under both delivery limits and both table policies,
using the Julia scheduler. The source is the analytic scalar equation
`y' = (-1/4 + i/2)y + (i/8)y`, over `1/32 m`. This isolates execution and control;
it is not a modal FFT/Kerr propagation qualification. The exact discrete map is

```math
y_{n+1}=e^{h(-1/4+i/2)}
       \left(1+ih/8-\frac{(h/8)^2}{2}\right)y_n.
```

MPFR checks every nonzero exponential argument used by this binary step graph
against a predeclared `8*eps(Float64)` componentwise relative allowance. The CUDA
documentation reports observed elementary-function accuracy rather than a formal
all-input theorem; these concrete witnesses establish only their tested inputs.
There are fewer than 256 rounded scalar operations and elementary-error units per
cell and step in this two-stage graph, and its amplification over the window is
less than two. Comparing both implementations therefore uses
`2*gamma(512*n_steps)*norm(initial)` with binary64 unit roundoff. The same bound
also checks the independently evaluated discrete map. This is a matched-history
rounding witness, not an adaptive global-error estimate.

Source and norm failure rows, exact work counts, receipt traces, context/stream
restoration and accepted-field preservation accompany the numerical checks.
The standalone witness reads its final field through an explicit owned copy.
The production physical export transaction is still additional work.

The frozen host milestone passes 56 orchestration assertions and 808 directly
affected existing CPU assertions, including zero warmed advancement allocation.
CPU Aqua/JET passes 16 checks with all 2,242 definitions analyzed and none cached.
The explicit CUDA witness is prepared but has not yet run on a device; these host
results do not qualify its compilation, arithmetic, context or transfer behavior.

## Remaining integration

The modal source still needs context-owned allocation, full-map chunk offsets,
device coefficient staging and compact plasma failures. Dense observations,
statistics/monitors, splitting, input noise, spatial sources and the remaining
march families must use the same ownership/readback boundary. Host profiles and
scalar control policy remain host records; numerical fields remain on the device.
Actual A40 qualification and independent review are required before integration
is described as a tested GPU march. Other vendors require their own evidence.

The primary arithmetic reference checked during development is NVIDIA's
[floating-point computation appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).
Its observed-error caveat is retained; precise compilation alone is not an error proof.
