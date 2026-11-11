# Device execution for RK dense observations

CPU and device execution use the same `retain_runge_kutta_observation!` and
`observe_runge_kutta!` routines. Coefficient formation, the stiff-bin decision,
the nested polynomial, represented phase, endpoint selection and hard masking
have one implementation. The device helper file supplies storage admission,
compact finite checks and buffer accounting; it defines no alternate RK method
or continuous extension. The existing CPU binary32 and DF32 paths are unchanged.

The device constructor admits binary64 modal diagonal, represented and expansion
operators and validates their selected context before allocating work. Its five
coefficient fields, endpoint, observation, exponent, stiff flags and expansion
weights use the selected execution. The original CPU integer failure vector becomes
a device integer vector plus a persistent host mirror only for device execution.
Each finite check downloads one machine integer per column. Retention checks the
initial field, endpoint, stage array, represented exponent and formed coefficients;
it never downloads their numerical payloads. Memory reporting counts both mirrors.

Retention and observation keep their existing argument and alias checks. A device
work object accepts only its prepared execution. Shape, storage and ownership admission
failures preserve the retained window; numerical or readback failure during retention
leaves that window unavailable.
The caller's initial field, stages and finished endpoint are read-only. Device copies
and zeroing use the common execution primitives. Exact endpoints and accepted positive
landing gaps copy the endpoint; interior numerical failure is checked before masking.
The session uses its existing dense-plane scheduling and delivery rules without
another controller, source invocation or field commit.

The serial test model refuses host scalar access to numerical arrays, records compact
download sizes, and injects a partial readback failure. It covers different execution
owners, interior foreign headers, nonfinite masked bins, canonical endpoint gaps,
buffer accounting and a two-step session delivering interior planes. These are
orchestration checks, not CUDA arithmetic evidence. Actual CUDA tests disable scalar
indexing and check context/stream restoration, retained input bytes, compact storage,
the same numerical comparisons and warmed device allocation.

Comparisons use an absolute envelope formed before cancellation from the actual
solution and continuous-extension weights. For each element, define the positive
coefficient envelopes `B2=h*sum(abs(b_i)*norm1(k_i))`,
`B3=h*norm1(k_1)+B2`, `B4=B2+h*norm1(k_N)+B3` and
`B5=h*sum(abs(d_i)*norm1(k_i))`. The nested evaluation envelope is
`norm1(y)+abs(t)*(B2+abs(s)*(B3+abs(t)*(B4+abs(s)*B5)))`, using the same recorded
binary64 `t` and `s` as both executions. The test computes this in MPFR and applies
`4*gamma_512`: two scalar graphs each have fewer than 128 operations, the remaining
allowance covers phase formation and the finite elementary-function hypothesis,
and the largest tested real phase is 1.2, whose gain is below four. This is conditional
on the binary64 elementary-function accuracy assumed by device qualification; it
does not prove arbitrary-range vendor transcendental accuracy. The inherited dense
tests independently check the polynomial with MPFR and its published midpoint
constraints. No observed discrepancy determines a tolerance.

Initial validation passes 5,730 affected CPU/spatial/DF32 assertions. The final
source repeats 3,508 directly affected CPU/spatial/DF32 assertions and 334 model/RK
assertions, including 148 new transport/session checks. Final CPU Aqua/JET passes
16 checks with all 2,310 definitions analyzed and none cached. Independent review
is recorded at integration. Actual CUDA execution remains pending, and public
admission is unchanged.
