# Binary64 CUDA low-storage replay

This internal method slice keeps the fixed midpoint RK4 interaction-picture method
and its public Julia signatures. CPU binary32 and binary64 use the previous operation
graph and allocation policy. A non-CPU execution requires binary64 before workspace
allocation. This does not open public CUDA selection or qualify a complete material
source, observation service, export bridge, or vendor runtime.

## Numerical and lifecycle boundary

`LowStorageWork(execution, state, RK4IPTableau(); norm_mask)` owns the three existing
work matrices. `low_storage_trial!` still invokes four complete source evaluations
and four forward linear flows, with the same register assignments, coefficients,
positions and protected scale/sum-of-squares recurrence. The diagonal-flow derivation
and references remain SEP 24 sections 4.2–4.3 and the `RK4IPTableau` docstring.

Device preparation adds two `DeviceReadback` records, with the exact preparing
execution retained. `norm_readback.device` is the existing `norm_parts` matrix and
`finite_readback.device` is the existing `finite_columns` vector. The column kernel
uses device arrays; the existing binary64 host merge receives only these summaries.
The finite flags inspect every stored field entry, including rows outside the norm
mask. A wrong execution, incompatible workspace, retained alias or unsupported
writable output layout refuses before numerical entry.

`LowStorageSession` reuses the common scheduler. Launch, fixed replay edges, hard
masks, snapshots, stop requests and acknowledgment generations retain their original
ordering. The state is marked in flight before the first source call. Source, flow,
norm or candidate-finishing failure poisons that state; accepted-step service failure
occurs after a valid field and attempt record commit. Attempted source calls are
recorded before entry and committed source counts increase only after finishing.
There is no rollback buffer, adaptive error estimate, new controller or dense output.

## Storage and transfers

For `r` spectral rows and `c` columns, the explicit binary64 workspace array data is:

| Owner | Shape and element type | Bytes |
|---|---|---:|
| Three work registers | three `(r,c)` ComplexF64 arrays | `48*r*c` |
| Norm mask | `(r,)` Bool | `r` |
| Device norm partials | `(2,c)` Float64 | `16*c` |
| Device finite flags | `(c,)` Bool | `c` |
| Persistent host partials | `(2,c)` Float64 | `16*c` |
| Persistent host flags | `(c,)` Bool | `c` |

The device total is `48*r*c + r + 34*c`. CPU binary64 retains
`48*r*c + r + 17*c`, without mirrors. These counts exclude the separately owned
session field, masks in `MarchNorm`, linear-flow/source storage and bounded attempt
records. Shared owner-aware ledger traversal deduplicates the readback device aliases.
GPU allocator reservations and vendor process state are not array payload.

Each norm copies exactly `17*c` bytes to its persistent host mirrors. Linear-flow
and finished-candidate status use the existing, separately accounted per-column
integer failure readback. Advancement does not download a complete field. Test
readouts deliberately do so only at an observation boundary after advancement.
The warmed CPU path remains allocation-free; this statement is not an assertion
that vendor launch machinery allocates no Julia or driver objects.

## Independent short-step witness and bound

The committed witness has the scalar commuting equation `u' = D*u + g*u`,
`D = -1/4 + i/2`, `g = i/8`, and one or eight equal steps over `L = 1/32 m`.
The initial real and imaginary components are small dyadic fractions. Step lengths,
midpoints and exponent inputs are exactly representable. A structural mask sets one
row to exact zero. Eliminating the midpoint interaction-picture variables gives the
independent discrete map

```text
u_n = [exp(D*h) * P4(g*h)]^n * u_0
P4(x) = 1 + x + x^2/2 + x^3/6 + x^4/24.
```

The reference evaluates this expression with 256-bit MPFR arithmetic, then rounds
once for a binary64 readout. It does not repeat the implementation's register loop.
This is a matched-discrete-method witness, so it measures rounding rather than the
RK4 truncation error relative to the continuous equation.

The bound assumes round-to-nearest binary64 arithmetic with no overflow or underflow
for these fixture operands, and a componentwise relative error at most `16*u` for
the two nonzero components of `exp(D*h/2)`, where `u = 2^-53`. The CUDA test checks
that primitive hypothesis against MPFR at each actual half-step input before using
the bound. It is not a universal claim about the vendor complex exponential. The
tabulated and direct flow paths use the same factors; cached reuse adds no operation.

Per cell and step, four complex source products and four complex flow products use
at most 48 scalar operations. The first, two middle and final register updates use
at most 28. Four non-dyadic divisions form the sixth/third coefficients. Reserving
eight more operations for phase products and 32 rounding equivalents for each of
the four qualified complex exponential factors gives at most 216. The bound uses
256 per step, leaving slack for the coefficient and final-reference rounding. FMA
contraction may remove rounding steps and does not need a larger count here.

Replacing every real/imaginary arithmetic contribution by its absolute value bounds
the expanded scalar register graph by
`M = sum(abs(real(u0))) + sum(abs(imag(u0)))` times
`exp((abs(real(D))+abs(imag(D))+abs(real(g))+abs(imag(g)))*L)`.
This follows from the positive-coefficient RK polynomial being bounded by its
exponential series. No reduction affects the fixed step sequence. A conservative
bound for two implementations, also covering the rounded independent readout, is

```text
gamma(512*n) * M * exp((7/8)*L),  gamma(k) = k*u/(1-k*u).
```

The same component one-norm envelope bounds the reported Euclidean difference.
The test evaluates the bound at 256 bits and converts outward to binary64. All
quantities are finite and far from the normal/subnormal boundary in this fixture.
This argument does not qualify nonlinear material sources, different histories,
adaptive schedules, arbitrary exponential operands or binary32 device work.

The norm witness separately compares the scaled sum-of-squares result with an MPFR
norm. Its conservative count `32*r*c + 16*c` covers division/squaring/addition,
column rescaling, the serial host merge and final square root on the bounded fixture.
It preserves the mask and independently tests finiteness outside that mask.

## Qualification status

The source-free host model counts every preparation allocation and norm readback,
checks exact payloads, foreign-offset aliases and repeated-index output refusal,
and exercises source and finishing failures through actual session hooks. It is
orchestration evidence, not device execution evidence. Existing CPU low-storage,
session and precision tests retain their unchanged arithmetic and warmed allocation
checks. The focused author run passed 2,322 distinct CPU assertions at two threads,
including timing-probe and preserved binary32 consumers. One inherited RK test
fixture needed its existing readback field appended in the private harness; the
low-storage implementation preserves its previous positional workspace constructor.
The new 115 host checks also passed at one thread. CPU Aqua/JET passed 16 checks
with all 2,248 definitions analyzed, plus eight concrete call analyses.

`test/cuda_low_storage.jl` is an explicitly selected device leaf. It requires a
functional CUDA device, disables scalar indexing, and checks one/eight steps,
direct/tabulated flow, bounded yields, exact source accounting, compact storage,
failure poisoning, snapshots, stop receipts and context/stream restoration. Actual
device execution and unfiltered vendor quality remain separate mandatory evidence;
the initial host handoff does not claim that either has passed. The previously
recorded CUDA vendor-analysis failures remain visible under issue #741.
