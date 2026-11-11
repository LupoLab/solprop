# Binary32 march arithmetic

The CPU RK and low-storage leaves retain host binary64 positions and translated
tableau records. State, slopes, stage combinations, defects and norm products use
binary32. Each scalar stage weight and step multiplier rounds before entering its
pointwise kernel. Nonzero multipliers must fit the binary32 normal range; preparation
checks tableau coefficients and trial admission checks the actual step before writes.
This range boundary does not infer an admitted physical phase domain.

`LinearFlow` forms and evaluates its represented exponent in binary64. Its exponential
rounds once to the state precision before multiplying the state. The stable linear
increment follows the same boundary, using the existing protected complex `expm1`
formula. Zero exponents preserve exact copies or zero increments. DF32-wide flows
require their separate whole-expression certificate and are not inferred here.

## Protected norms

For each column, first find the largest absolute real or imaginary component `m`.
For nonzero `m`, sum the squares of components divided by `m`, using the protected
Sum2s accumulator described in [observation arithmetic](OBSERVATION_PRECISION.md).
Normalized components are at most one, so finite inputs cannot overflow in their
products or sum. There are two terms per spectral row; workspace preparation checks
the strict `2*n_omega < 2^24` theorem hypothesis before allocation. A zero column has
zero scale and zero sum. Any nonfinite state entry rejects the norm, even outside its
mask. The small column merge and final square root use host binary64.

For a weighted defect norm, the existing global magnitude supplies
`d_i = rtol * weight_i * magnitude + atol`, in stored state units. Construct these
scalars in host binary64, then retain `d_i = m_i*2^e_i` as a rounded binary32 mantissa
and an integer exponent per frequency. Defect ratios use binary32 `ldexp` and division;
no physical inverse-square weight is formed in binary32. A zero denominator and zero
defect contribute exact zero. A nonzero defect with zero denominator, a nonfinite
denominator, or an unrepresentable ratio yields a failed scalar norm. All buffers are
reused and appear in the ordinary retained-array ledger.

The error budget includes normalization divisions, squared products, the rounded
denominator, each column's Sum2s bound and the host merge. Applying only a merge bound
to rounded column sums would omit their earlier errors. Gradual underflow contributes
absolute terms bounded by half the smallest binary32 subnormal per inexact operation;
it is not flush to zero. Tests with columns differing by 260 binary exponents check
the protected reduction against binary64 reference readings and explicit bounds.

## Qualification scope

Short eight-step RK and low-storage witnesses compare with the analytic solution of
a split linear equation, with separate rounding and truncation terms. Existing
binary64 leaf tests retain their original bounds and convergence checks. Concrete
JET calls and warmed allocation checks cover binary32 RK trials, error norms and
low-storage trials. Full session scaling, Adams, portable history, DF32 and devices
remain separate integration work; this leaf does not enable public single precision.
