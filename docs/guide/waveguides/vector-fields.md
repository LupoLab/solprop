# Electric and magnetic mode fields

Full vector modes supply electric and magnetic fields, with their phases and
longitudinal components, rather than only a scalar transverse profile. These fields
determine both physical power and nonlinear coupling in a [general modal
calculation](general-modal.md). The [solid-fibre workflow](solid-fibres.md) shows how to
construct a guide before using this reference.

## Full electric and magnetic mode fields

This chapter serves the [general modal route](general-modal.md),
which propagates several solid-guide channels with their full vector fields and a
power-orthogonal metric. That route needs a mode's complete vector field, not only its
transverse profile. `StepIndexMode.electromagnetic_field(xy, omega, z)` returns complex Cartesian
`(E_x, E_y, E_z)` and `(H_x, H_y, H_z)` at explicit Cartesian positions in metres.
Both fields are normalised to unit full-domain Poynting power, independently of those
points, in V/m and A/m per sqrt(W). These fields use the **internal FFT convention**,
`exp(i omega t - i beta z)`. They are separate from `Mode.field`, which is a real,
frozen-frequency, two-component transverse profile used by the mode-averaged and
multimode routes; both use the same exact bound roots and family and cutoff
counting.

In each homogeneous region, with `q² = epsilon_r (omega/c)² - beta²`, eliminating
the transverse components from Maxwell's curl equations gives

```{math}
\mathbf E_t=-\frac{i}{q^2}\left[\beta\nabla_t E_z
 -\omega\mu_0\hat{\mathbf z}\times\nabla_t H_z\right],\qquad
\mathbf H_t=-\frac{i}{q^2}\left[\beta\nabla_t H_z
 +\omega\epsilon_0\epsilon_r\hat{\mathbf z}\times\nabla_t E_z\right].
```

The core uses regular Bessel functions and the exterior uses decaying modified
Bessel functions. Analytic axis limits avoid division by zero. The implementation
uses exponentially scaled exterior functions. Both hybrid parities and TE/TM
sectors have all three components, including components which vanish by symmetry.
Attenuating/leaky fields are refused: a prescribed attenuation coefficient does
not establish a complex eigenfield, its reciprocal adjoint, or its exterior norm.
The cutoff guard of the transverse profile, `PROFILE_MIN_W`, applies here too.

Use a provider to obtain true-power-normalised fields:

```{code-block} python
import numpy as np
from solprop.constants import C
from solprop.modes import IndexMaterial, ModalGuide, StepIndexMode

omega = 2 * np.pi * C / 835e-9
mode = StepIndexMode(250e-9, core=1.45, cladding=1.0, profile_wavelength=835e-9)
guide = ModalGuide.from_step_index(
    [mode], core=IndexMaterial(1.45), cladding=IndexMaterial(1.0)
)
provider = guide.field_provider
nodes = provider.node_rule(omega)
sample = guide.solve(omega, 0.0, nodes=nodes)
```

A material law is a `PermittivityLaw`: anything with `permittivity(omega)`, an
`omega_domain` and an `identity`, of which `IndexMaterial` is the adapter for a
glass law or a constant index. It is named for the law, not for a material,
because `solprop.optics.Material` in the flat `solprop.sol` namespace is a length
of a real medium and the two must not be confused. A `MaterialRegion` gives one
law an annular region, whose radii may follow declared paths.

`ModalNodeRule` owns Cartesian positions, positive area weights and an integer
material-region index at every node. Its panel machinery is that of the
[transverse quadrature](mode-sets.md#the-transverse-quadrature). The step-index builder splits at the core
interface and integrates the omitted cladding power separately. Increasing radial
and angular resolution tests quadrature error independently of the tail.

Every provider builds its rule through one signature,
`node_rule(omega, z=0.0, *, n_r=..., n_theta=None)`. The frequency is required
even where the mesh does not depend on it, because the power a rule discards
beyond its outer radius does: both providers put that exactly integrated fraction
in the rule's `identity["tail_power_bound"]`. `n_theta=None` asks the provider for
the angular count its channels' azimuthal orders need, `4(n + 1) + 1`, which is
exact for their quartic products; the radial provider uses at least nine.

`ModalSample.electric` and `.magnetic` have shape `(n_nodes, 3, n_modes)` and units
V/m and A/m per sqrt(W). Each physical parity partner is one channel. The
Hermitian Poynting product is

```{math}
W_{mn}=\frac14\int(\mathbf e_n\times\mathbf h_m^*
                  +\mathbf e_m^*\times\mathbf h_n)_z\,dA.
```

Every provider refuses an unsupported channel by name -- too near cutoff, an
uncertifiable tail, a boundary residual above its bound -- rather than masking
it, so `ModalSample.valid` is all `True` in any sample a provider returns. The
field is there so that a future provider which can honestly return a partial set
has somewhere to say so, and tracking refuses a sample whose mask is not all
`True`.

The fields are normalised over the full radial domain, separately from the
caller's node rule. Their sampled flux matrix therefore exposes missing power in
a truncated or inaccurate rule. It is not forced to be the identity. Reciprocal
backward fields satisfy `e_minus=e.conj()`, `h_minus=-h.conj()` for this bound,
lossless model, giving `C=-4W`. The internal source functional is
`-i omega/4 * W^-1 integral(e.conj() dot P_NL) dA`. It includes the longitudinal
electric field and differs from an electric-only coefficient-recovery inverse.
The full-vector Kerr contraction is discussed by
[Afshar and Monro](https://arxiv.org/abs/0812.1824). solprop derives it independently
from Maxwell's equations, because the paper's printed equations are not self-consistent
in places: the linear electric term of its Eq. (5) lacks a factor $\omega$, the integral
in its Eq. (8) and the definition of $N$ that follows differ by a factor two, and the
displayed sign of the nonlinear source in its Eq. (9) differs from direct substitution
of its curl equations. The sign used here is fixed by the positive-$n_2$ plane-wave
limit.

Putting the instantaneous isotropic Kerr polarisation of an analytic signal,
`P = eps0 chi3 (2 |E|^2 E + (E.E) E*) / 4`, into that functional at unit
amplitude and unit flux leaves `-i gamma` with

```{math}
\gamma=\frac{\omega\varepsilon_0}{16}\int\chi^{(3)}
\left(2|\mathbf E|^4+|\mathbf E\cdot\mathbf E|^2\right)dA,
```

which is `solprop.nonlinear.normalisation.modal_kerr_coefficient(electric,
weights, chi3, omega)`. The susceptibility is per node, so a linear cladding is
excluded by being zero there and no region ownership is assumed. Both the
longitudinal field and the `|E.E|^2` term are in it, and neither is in the scalar
`n2 omega / (c A_eff)`: on the 250 nm strand the vector coefficient is **25 %
larger** (0.541771892 against 0.432424108 W⁻¹m⁻¹), while at weak guidance the two
agree to better than 0.2 %.

The displayed $\gamma$ above is the **unit-flux** one, which is what a provider's
normalisation declares over the whole physical domain. A channel whose sampled flux
$W_{mm}$ is not one divides by it, and `metric=` is how that is said: a positive float
is read as this channel's own $W_{mm}$, while a square Hermitian
$(n_\text{modes}, n_\text{modes})$ array with `channel=` gives the
coefficient $[W^{-1}]_{mm}$ — which is what the general modal route's own projection map
contracts, and which equals $1/W_{mm}$ only while $W$ is diagonal. On a sampled basis
the number a run executes is therefore the unit-flux $\gamma$ times $[W^{-1}]_{mm}$, and
the difference between the two is the basis's measured departure from orthonormal.
