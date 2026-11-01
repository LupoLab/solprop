# Step-index fibres

A circular step-index fibre has a uniform core and a uniform exterior. Its bound vector
modes retain the boundary conditions and longitudinal fields that a scalar weak-guidance
approximation omits. Use a single mode as the dispersion and area of a [GNLSE
calculation](gnlse.md), or use several channels in a [general modal
calculation](solid-fibres.md).


{class}`~solprop.modes.stepindex.StepIndexMode` is a mode of a solid circular core in
a homogeneous cladding — a conventional fibre, or a tapered strand or nanowire in air.
It solves the lossless vector eigenproblem exactly, with no weak-guidance approximation,
so it holds for high index contrast. A glass's real part is what it solves with: an
optmat index is held as its recorded transparent approximation, which warns once per run
where the absorption it drops exceeds 100 m⁻¹
([materials](../media/material-selection.md#materials-from-optmat)), and a loss is declared with `loss=`.
The family and indices follow Snyder and Love
(`kind="HE"`, `"EH"`, `"TE"` or `"TM"`, azimuthal `n`, radial `m` counted from the
highest effective index, and `parity="even"` or `"odd"` for a hybrid). Declare the
wavelength that fixes its transverse profile:

```{code-block} python
import numpy as np

from solprop.constants import C
from solprop.modes import StepIndexMode
from solprop.modes.base import beta2, zdw

fibre = StepIndexMode(1.25e-6, profile_wavelength=835e-9)
dispersion = beta2(fibre, np.array([2 * np.pi * C / 835e-9]))
zero = zdw(fibre, wavelength_limits=(400e-9, 1400e-9))
```

The default core is fused silica -- optmat's `fit/SiO2`, taken
through its transparent approximation, so two channels built separately share one
material (adapters of one model compare equal) -- and the default cladding is the
constant index 1 (a strand in air). Pass a `GlassIndex` or a positive constant for either material.
{meth}`~solprop.modes.stepindex.StepIndexMode.from_numerical_aperture` instead sets the
core index from a numerical aperture by the weak-guidance convention
`n_co = n_cl + NA**2/(2*n_cl)`, with a fused-silica cladding by default, and
differentiates that expression through the cladding's material slope.

**How it propagates.** A single step-index mode propagates through the mode-averaged
GNLSE: `prop_gnlse(None, length, fibre=fibre, n2=..., ...)` or
{meth}`~solprop.driver.simulation.Simulation.from_gnlse` with the same keyword. It is
not a gas-filled waveguide, so `prop_waveguide` and `prop_capillary(modes=...)` refuse
it and say which entry to use. Several step-index channels propagate together on the
[general modal route](general-modal.md), through the full-vector provider
described in [full electric and magnetic mode fields](vector-fields.md#full-electric-and-magnetic-mode-fields), which builds a common
exterior domain and uses a power-orthogonal (Poynting) metric. The ordinary multimode
route cannot take them: exact vector modes are not orthogonal under its transverse
electric-field inner product (the overlaps are measured at the end of this chapter).

**The eigenvalue equation.** For core radius $a$, $k=\omega/c$, core/cladding indices
$n_c,n_s$ and effective index
$n_e$, define $u=ak\sqrt{n_c^2-n_e^2}$, $w=ak\sqrt{n_e^2-n_s^2}$,
$V^2=u^2+w^2$ and $r^2=n_s^2/n_c^2$. With
$A=J_n'(u)/(uJ_n(u))$ and $B=K_n'(w)/(wK_n(w))$, the vector equation is

$$ (A+B)(A+r^2B)=n^2(1/u^2+1/w^2)(1/u^2+r^2/w^2). $$

The solver clears the denominators. Writing $P=uw^2J_n'K_n$ and
$Q=u^2wJ_nK_n'$, its pole-free product is

$$G=(P+Q)(P+r^2Q)-n^2V^2(w^2+r^2u^2)J_n^2K_n^2=0.$$

Positive factors can be divided out without changing the roots. Bessel recurrences
cancel the endpoint terms algebraically before evaluation; this avoids spurious sign
changes from subtracting nearly equal products at $u=0$ and $w=0$.
For azimuthal order zero the separate equations are explicitly

$$\mathrm{TE}:\quad wK_0J_1+uJ_0K_1=0,\qquad
\mathrm{TM}:\quad n_c^2wK_0J_1+n_s^2uJ_0K_1=0.$$

Written as characteristic equations these are easy to swap: the TE equation is the one
without the index ratio. Their cleared form carries one power
of $J_0(u)$, which vanishes at the zeros of $J_0$ and so conditions
$dn_e/d\omega$ less well as a TE₀ₘ or TM₀ₘ mode approaches its own cutoff; it is
never zero *at* a root.

**Finding the roots.** The scan has at least 64 nodes and spacing at most 0.1 in $u$,
augmented with
Bessel zeros. Every sign-change bracket is refined with Brent's method; hybrid
roots are classified by the vanishing branch and radial order counts downward from
the highest effective index. The scan is then held to the family's exact cutoffs,
because two roots can share one interval: $J_{-1}=-J_1$ puts HE$_{1,m+1}$ and
EH$_{1,m}$ at the same cutoff $j_{1,m}$, as roots of the same $n=1$ product, and
their two sign changes cancel. While a count is short of the cutoff count, the
equal-sign intervals meeting the region the cutoffs localise the missing root to
are sampled twice as finely, up to six halvings; the $i$-th root must then lie
between the $i$-th and $(i+1)$-th cutoff, and a root whose radial order cannot be
assigned raises `ModeError` rather than being returned. A missing
guided root raises `ModeError` with its $V$, cutoff and mode. No pole filter or
dispersion spline is used. Because the denominators are cleared before the scan, the
poles of the uncleared equation at $J_n(u)=0$ are double zeros of a non-negative
factor and never appear as sign changes, so they can never be reported as roots.

**Derivative and loss.** The real frequency derivative is analytic:
$dn_e/d\omega=-(\partial G/\partial\omega)/(\partial G/\partial n_e)$,
including both material slopes. A 40-digit mpmath reference checks this derivative
independently, and Richardson differences check the material chain between its points.
`loss` is a non-negative absolute power attenuation in m⁻¹, constant or callable,
with $\operatorname{Im}n_e=c\alpha/(2\omega)$. A callable's `derivative` attribute
supplies its analytic slope; only this loss term is differenced when that attribute is
absent. The material's imaginary index is not converted to mode loss: material
absorption is outside the model, and the eigenproblem uses real material indices.

**Fields and normalisation.** Hybrid profiles have even and odd parity; their fields
use Snyder and Love's
Table 12-3, rotated into Cartesian components. TE and TM use their separate
azimuthally symmetric limiting solutions. Tangential electric field and radial
displacement are continuous at the core boundary; radial electric field jumps.
The exact boundary point returns the core value. Normalisation and effective area
use analytic azimuthal moments and adaptive radial integration split at the core.
The integration radius is $\max(2a,a[1+\ln(10^8)/w])$, which follows the mode's own
cladding tail down to an amplitude of $10^{-8}$ and so retains the long tail near cutoff.
A cladding decay parameter $w$ below 0.1, where that tail would reach 185 core radii, is
refused. The profile is evaluated at the declared `profile_wavelength` for both
parities, and used unchanged at every frequency; there is no hidden default wavelength.
A varying radius recomputes the eigenpair along $z$ at that wavelength, so such a taper
is not scale invariant.

**Cutoffs.** `cutoff_parameter()` uses the material indices at the profile
wavelength. For
$\mathrm{HE}_{nm}$ with $n\ge2$ the **exact** cutoff solves
$J_{n-2}(V_c)+(n_c^2-n_s^2)/(n_c^2+n_s^2)J_n(V_c)=0$.
The often quoted Bessel zeros apply in the weak-guidance limit. For 1.47/1.45 indices,
HE₂₁ cuts off at 2.416293236, not the weak-guidance 2.405; for 1.45/1.0 it cuts off at
2.760804856 — ten significant figures, which is what `brentq`'s four-ulp `xtol`
supports, and both are pinned in `tests/modes/test_stepindex.py`. TE/TM cutoffs are zeros
of $J_0$, EH cutoffs zeros of $J_n$, and HE₁₁ has no cutoff. Higher HE₁ₘ cutoffs
are the preceding zeros of $J_1$.
These conditions and the separate TE/TM fields are also given in Appendix C of
[Le Kien et al. (2018)](https://doi.org/10.1103/PhysRevA.97.013821).

**Quadrature and orthogonality.** A transverse rule reads `radial_breaks(z)` and maps
a Gauss or Gauss–Kronrod rule onto each smooth radial panel. `n_r` is the node count
**per panel**;
`TransverseQuadrature.n_r` records the total first-coordinate count, and `breaks`
records the panel boundaries divided by the outer radius. With no breaks the rule is
the single-panel one of [the transverse quadrature](mode-sets.md#the-transverse-quadrature). For the
1.25 µm silica strand at 835 nm, 24 nodes per panel give a parity-pair Gram defect of
6.66e-16; an unsplit 24-node rule misses by 0.00308. This is a
discontinuity-resolution issue, not a normalisation fit.

Exact vector modes are not orthogonal under the transverse electric-field inner product
the ordinary multimode route uses. At $V = 6$ and core index 1.45, adaptive
electric-field overlaps of $\mathrm{HE}_{11}$ with $\mathrm{HE}_{12}$ are 2.54e-4 for
$\Delta = 0.005$ and 3.03e-3 for $\Delta = 0.05$, where
$\Delta = (n_c^2 - n_s^2)/(2n_c^2)$; both exceed the driver's 1e-6 orthonormality
threshold, so the driver refuses such a set. (Overlaps with the two $\mathrm{HE}_{21}$
orientations vanish by angular symmetry. The test integrates on a shared disc, splits
at the core and scales the radial coordinate so the adaptive absolute tolerance
resolves the integral.) The scope of `StepIndexMode` on its own is therefore
single-mode propagation; several channels need the general modal route and its
power-orthogonal metric
([full electric and magnetic mode fields](vector-fields.md#full-electric-and-magnetic-mode-fields)).

**Large V is not supported yet.** The characteristic equation and its derivatives use
the unscaled modified Bessel functions `kv` and `kvp`, which both underflow to zero at
large cladding arguments (about 1600 and above) although their ratio is finite. Such a
fibre — a 300 µm core of NA 0.22 at 250 nm, for example — fails at construction: the
division warns as invalid and the root scan raises a
{class}`~solprop.exceptions.ModeError`, since the radial order of its roots cannot be
assigned.
