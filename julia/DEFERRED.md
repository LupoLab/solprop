# Deferred Julia scope

The maintainer selected binary64 CPU and CUDA as the SEP 48 implementation baseline
on 11 November 2026. These records preserve unfinished work without admitting it or
making it a hidden requirement of that baseline. Public unsupported selections keep
their named refusals. No failed qualification has become a pass.

## Binary32 CPU with binary64 wide arithmetic

The modal/spatial kernels, three march families, observations and services have an
internal implementation. All 59 distinct short composition cases have passing latest
results, including the corrected 32-case physical spectral-band subset. The separate
six-case Adams-versus-RK physical campaign completes all cases: four pass, while both
helium cases meet every absolute envelope but fail the unchanged twice-RK criterion.
Independent 16/32-step binary64 references settle to at most `2.514e-15` across scored
metrics. Identical NumPy binary32 helium controls pass. See
[CPU_SINGLE_ADAMS_QUALIFICATION.md](CPU_SINGLE_ADAMS_QUALIFICATION.md) and issue 753.

The reduced-log plasma correction and population ownership guard are independently
reviewed at `e690ef30` and `cd409ee3`. ADR-080 states the CPU elementary-function
boundary; it does not establish an all-Float32 implementation inside vendor libraries.

Remaining: resolve the physical comparison, complete the precision/restart/admission
matrix and x86 measurements, review the final composition, then enable public admission
only after its required evidence passes. Binary32 input noise remains refused.

Run the retained managed qualification explicitly with
`uv run --no-sync python tools/check_julia.py precision` after publishing a matching
setup and exporting its identity as for the baseline Julia lane. This command runs
all seven deferred numerical modules and treats every skip, failure or error as a
failure. The baseline `python` phase excludes those modules explicitly; it retains
binary64 sessions, public refusals, pure precision lowering and shared conformance.
The Julia engine Core regressions continue to exercise the retained arithmetic.

## Binary32 CPU with DF32 wide arithmetic

DF32 replaces selected wide intermediates; the evolving state and FFTs remain
binary32. It is not a separate whole-state precision or backend.

Certified constant, cumulative-table and expansion flows are attached to internal
RK, low-storage and Adams sessions. Portable same-precision Adams history is tested
across the two wide types and supported backend crossings. The reviewed Adams leaf
has 26,755 independent assertions; its integrated bridge/history milestone is
`ed3a2fcb`, with 51 managed cases plus two group and six boundary cases passing.
The arithmetic remains conditional on the documented RN-even/FMA/gradual-underflow
hypotheses and actual per-operation certificates.

RK dense output is independently reviewed at `990871fb` (integrated as `ad5ab855`):
2,423 independent assertions pass, including MPFR factors, query preservation,
ownership and zero warmed allocations. Its engine attachment is integrated at
`98f6af3e`; 2,419 directly affected bounds-enabled assertions pass.
Python dense attachment remains refused and deferred.

Adams dense output is an explicitly unqualified isolated draft at `973fc351`; no
numerical test completed before it was stopped. It is not part of the integrated
implementation. Remaining work includes that dense leaf, separable spatial assembly,
tabulated execution, final restart/admission coverage and performance measurements.
The reviewed mathematical notes retain their exact domain and conservative refusals.

## Metal

The integrated execution/FFT foundation at `035ee5ac` passes 308 actual Apple M5 Max
checks, including independent ownership, queue and transform witnesses. Aqua passes;
unfiltered JET remains failed (745). Precise Float32 arithmetic loses subnormal values
in 20 of 136 checks, including a normal-input product's FMA residual (739). The current
DF32 proof therefore does not qualify this device. Native-cache accounting and
unsupported transform shapes remain recorded in 743 and 744. Complete propagation,
arithmetic replacement and device qualification are deferred; public admission stays
closed. No further Metal-specific work is required for the selected baseline.

## ROCm/HIP

The foundation is preserved outside the integration branch at `1a4d8aea`, followed
by the reviewed aligned-stride correction `ef4ee80e` and the capture draft/fix
`3c00734d`. The stride correction has 151 independent host assertions. The private
capture witness was resolved as address erasure during Julia type canonicalization,
rather than a production capture defect. No corrected HIP qualification or AMD
hardware test followed deferral, and no AMD rental has occurred.
Unfiltered JET remains failed (750). Full propagation and AMD qualification are
deferred; public admission stays closed. Common/CUDA capture corrections are
integrated separately and do not require these ROCm commits.

## Existing runtime limitations

Issues 610, 620 and 669 retain the accepted main-thread and signal boundaries and
remaining runtime failures. The earlier instruction to defer rare runtime diagnosis
until core implementation is complete still applies. Required baseline checks must
report their actual results; no deferred record silently waives a failing check.
