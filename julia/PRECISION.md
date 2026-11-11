# Binary32 integration contract

This is the shared implementation boundary for SEP 48 phase 3, following SEP 33.
It records interfaces and obligations; unfinished paths remain refused until their
numerical and session qualification passes.

`SourceArithmetic{T}` fixes `Float64` or `Float32` and two explicit integer powers:
`state_exponent` relates stored state to physical modal amplitude, and
`field_exponent` relates stored nodal samples to physical electric amplitude. With
synthesis-map exponents folded at upload they coincide. The exact binary64 accessors
are for host use. The record does not prove that a coefficient or intermediate fits
binary32. Binary64 has unit scales and retains its existing arithmetic.

Source and observation preparation receive `arithmetic=SourceArithmetic(Float64)`
as a keyword. `ModalSourceGrid`, `ObservationPreparation` and spatial observation
preparation retain `.arithmetic`. Python transfers the existing immutable binary64
descriptors; Julia owns their precision-specific preparation. The master derives the
state exponent from the original translated launch using the shared precision helpers.
It is unchanged by restart or replacement of the current field.

`ArithmeticTransfer(name, state_exponent, field_exponent)` carries that record from
Python. A physical starting field converts component by component with `ldexp`,
then rounds once to state precision; it never forms a reciprocal scale. The same
power scales the host absolute tolerance. Preparation refuses nonfinite inputs and
overflow in the scaled state or tolerance. State entries retain gradual underflow;
the separate table-buffer normal-range rule is not imposed on individual samples.

Field export widens the stored state before restoring its physical scale and applying
the existing power/optics recipe. Portable Adams history follows a different existing
contract: the accepted raw scaled state widens to complex128, source and sampled-state
rows retain the state's dtype, and sampled rates remain complex128. It does not apply
physical scaling. This matches the NumPy engine and HDF5 schema. Restore requires the
same scientific precision and exact matching row dtypes; the original launch fixes
the common scale. Binary32 rates are never narrowed during history transport.

Every narrowed source/observation buffer has a named block exponent and range fact.
An exponent may be folded at upload only when the result retains the oracle's admitted
range; otherwise its pass carries the exponent. Raman kernels retain normalization
explicitly. Source profiles, coordinates and combined stage coefficients remain host
binary64. Products of density, material factors and powers are assembled as mantissa
and integer exponent before rounding once, avoiding tiny or overflowing intermediate
coefficients. The source evaluates physical plasma rates with `field_exponent`, cubic
terms with its squared scale and plasma loss with its inverse square. Population-only
leaves receive `field_exponent::Int=0` explicitly.

State-sized source and observation work, transforms and pointwise arithmetic use `T`.
Long binary32 sums and running integrals use justified protected accumulation; sums of
squares scale before squaring. Final packed observations and control scalars are
binary64. Statistics restore their own physical homogeneity there: amplitudes degree
one; powers, energy, intensity and fluence degree two; widths, positions, ratios and
populations degree zero. Monitor records and decisions retain shared physical units.
All binary32 noise remains refused.

The master owns session state/controller scaling, export and portable history, Python
records, wide-type selection and public admission. Source and service agents own
disjoint numerical files. Changes to these shared records or scale conventions require
master agreement before dependent work. CPU binary64-wide execution is qualified first;
DF32 then attaches the reviewed whole-expression certificates and stored-rate provenance.
Device arithmetic and performance are separate qualifications.
