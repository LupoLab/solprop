# Binary64 device field export

The device export follows the same retained-generation contract and physical recipe
as CPU export: optional modal square-root scaling, then optional optics conjugation
and alternating frequency sign. Globally unit weights skip the complex product;
mixed weights retain its explicit zero terms for every cell. Spatial exports keep
their existing rule of ignoring modal power conversion.

Each device session owns a complex binary64 staging field, an equally shaped host
mirror and two integer facts per column on both sides. The first fact identifies
invalid weights or a nonfinite staged result; the second identifies nonunit weights.
Weight tables and expansions remain on the execution device. Only validity/identity
facts cross to the host before the complete physical field is downloaded.

The caller's destination must be writable detached host storage of the exact field
shape. Validate the receipt, retained-storage independence, execution and all values
before copying into it. A failed numerical recipe or partial device download changes
only session-owned scratch; the pending generation remains available for retry. The
engine does not consume that generation until acknowledgement.

For `r` frequency rows and `c` columns, transfer scratch owns `32*r*c + 32*c` bytes
on a 64-bit host: both complex fields and both two-row integer summaries. Existing
power-weight values, coefficients and prepared data are additional owners. The usual
memory ledger includes every buffer and deduplicates shared owners. CPU export keeps
its existing two-pass recipe and allocates no new transfer scratch.

`test/device_export.jl` exercises host orchestration, exact signed zeros, retained
storage, generation failures, compact transfer sizes, partial-copy failures and retry.
`test/cuda_export.jl` repeats the numerical and address-space checks with scalar
indexing disabled on actual CUDA. Host-model results do not establish CUDA support;
this internal leaf does not change public device admission.

The author host run passes 169 new orchestration assertions plus 61 existing
power-weight assertions. The affected CPU session, transfer and precision consumers
pass 2,469 assertions across 67 testsets, including their warmed zero-allocation
checks. CPU Aqua/JET passes 16 checks with all 2,253 definitions analyzed and none
cached. These checks use bounds-enabled Julia 1.13.3 with two threads. An initial
constructor-dispatch collision was corrected before this final run by explicitly
typing the execution-aware table overload. Actual CUDA execution remains pending.
Independent review additionally exposed an explicit-weight header overlapping the
staging field. Such weights now refuse before any summary or staging write; genuine
foreign offset-header tests verify unchanged output, scratch, weights and generation,
followed by successful retry with detached weights. The CUDA witness includes the
same retained-address boundary and awaits hardware execution.
