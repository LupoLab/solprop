# General modal guides that vary along the propagation

When a guide changes shape, its local modes and their normalisation change too. The
connection between those bases transfers amplitude between channels. This chapter
derives that connection and explains its sampling and error controls. Begin with
[mode conversion in a capillary taper](taper-coupling.md) for the physical idea, then
use the [moving solid-fibre
workflow](solid-fibres.md#a-solid-guide-that-moves-along-z).

## The connection of a basis that moves along z

A guide whose transverse profile moves is written in a **moving basis**, and a moving
basis costs a term. Expanding the field on the local modes at every position, $\mathbf
E(\mathbf r, z) = \sum_n c_n(z)\,\boldsymbol\psi_n(\mathbf r; z)$, the coefficients
obey

```{math}
\frac{dc}{dz} = (\text{the local generator})\,c - \Gamma c,
\qquad \Gamma = W^{-1}S,
\qquad S_{mn} = \langle\boldsymbol\psi_m\mid\partial_z\boldsymbol\psi_n\rangle,
```

where the pairing is **this package's own flux metric** — the Hermitian Poynting
product {py:func}`~solprop.modes.vector.flux_product` integrates — and $W$ is its Gram
matrix, the same one the source projection $Q$ is built from. One pairing for the dual,
the projection and the connection is what makes the three consistent.

**The flux self-terms are kept.** Differentiating the Gram gives the identity the whole
certificate rests on,

```{math}
W' = \langle\partial_z\psi_m\mid\psi_n\rangle
   + \langle\psi_m\mid\partial_z\psi_n\rangle = S + S^{\mathsf H},
\qquad\text{so}\qquad W\Gamma + \Gamma^{\mathsf H}W = W',
```

and with $dc/dz = -\Gamma c$ the flux norm obeys $d(c^{\mathsf H}Wc)/dz = 0$ exactly:
**the connection alone moves no power**. For an orthogonal basis with diagonal $W$, the real diagonal part is
$\operatorname{Re}\Gamma_{nn}=W'_{nn}/(2W_{nn})$, reducing to
$\tfrac12 W'_{nn}$ where $W_{nn}=1$. It describes the change of power
normalisation. The imaginary part describes the gauge's phase rate. For a
dense metric, use the full matrix identity rather than this scalar formula. Neither is zero in general. Zeroing the diagonal, as
the capillary's closed form correctly does for *its* basis, breaks the identity; for diagonal $W$ the resulting norm defect is
$\sum_n|c_n|^2W'_{nn}$, which is why that formula is **not** transplanted to a
solid guide.

{py:func}`~solprop.modes.connection.modal_connection` measures it from a provider:

```{code-block} python
from solprop.modes import modal_connection

link = modal_connection(provider, omega, z)
link.matrix  # Gamma in 1/m
link.self_terms  # its diagonal: the flux self-term and the gauge phase rate
link.antisymmetry  # the certificate, in 1/m
```

$\partial_z\boldsymbol\psi_n$ is a three-point Lagrange derivative of the provider's
own solves, tracked into one gauge, on **one shared node set whose panels are split at
the union of every stencil position's declared edges**. That is not a detail. A rule
built at one position and reused at another leaves its panel edge off the moving
interface, and a Poynting integrand with a jump there is then integrated by a Gauss rule
that does not see it: the error is first order in the displacement, so it is first order
in the step and survives the difference undiminished. On a step-index taper a frozen
rule reads a self-term of 0.98 1/m and a metric rate of 2.8 1/m where the true one is
below 1e-8. Splitting the panels aligns the quadrature with each solve's smooth pieces;
finite quadrature and mode-solve errors remain — and it must be the union of
the *declared edges*, not of the material radii, because a graded-index sector is a
finite-element field whose slope breaks at every element edge and whose whole mesh scales
with the core radius.

The reported `antisymmetry` is the residual of the metric identity, with $W'$
differenced independently of the pairing, so it is a statement about the sampling and
not about the arithmetic. It is reported in 1/m; relative to $\max|\Gamma|$ it reads
7.4e-11 on a step-index taper and 5.7e-9 on a graded-index one at the default step,
against the 1e-8 the tests admit, and
{py:func}`~solprop.modes.connection.modal_connection` warns when it exceeds
{py:data}`~solprop.modes.connection.CERTIFICATE_WARNING_RTOL` (1e-4), where the
connection is itself wrong at about a part in ten thousand. Refine the mode solves,
quadrature and longitudinal derivative separately when assessing it. The [qualification
record](../../dev/accuracy.md#11-capillary-modes-raman-and-the-moving-basis)
holds comparisons against the high-$V$ capillary limit and direct projection
of a frozen physical field. Those checks support the sign and finite-basis
interpretation; they do not establish convergence for every taper.

**A declared change of coordinates that moves.** For $c = T(z)d$ the declared basis
carries $-T^{-1}T'$ beside the transformed generator, and for a unitary $T$ the added connection is
$T^{\mathsf H}T'$ (subtracted from the generator). It needs no inverse and is anti-Hermitian
({py:func}`~solprop.modes.connection.basis_connection`). With it, a z-dependent unitary change of coordinates preserves the same physical
field. Changes of the coordinate amplitudes alone must not be interpreted as
physical mode conversion; compare fields in one declared physical basis.

**What the retained set cannot do.** Within the set the connection carries a frozen
physical field exactly. What lies outside it does not vanish at a material interface
while every retained basis function's expansion of it converges only in $L^2$, so the
propagated coefficients depart from the direct projection at second order in the total
contraction. That is a bound on how non-adiabatic a taper the basis describes, not an
error of the derivation.

**A specified fibre has none.** A {py:class}`~solprop.nonlinear.gnlse.SpecifiedMode`
built from a moving guide carries a dispersion path and an area path and **no transverse
field**: its profile is a surrogate disc for the nonlinear coefficient and is not the
mode, so $\langle\psi_m\mid\partial_z\psi_n\rangle$ cannot be formed from it. A
single moving channel needs none — a one-by-one anti-Hermitian matrix is a pure gauge
phase — and a multi-channel moving set on the mode-set route propagates with the term
absent, which is that route's behaviour for every moving multimode guide it has no
connection for and is not specific to a specified fibre. It is not a small omission: a
capillary taper from 50 to 40 µm over a fifth of a millimetre converts 7 % of the energy
out of $\mathrm{HE}_{11}$ when the connection is carried — which the capillary's own
contiguous $\mathrm{HE}_{1m}$ set does by default, with or without a wall (see
[the taper coupling](taper-coupling.md#the-taper-coupling-without-a-wall)) — and leaves
$10^{-18}$ there when it is not. Read any other multimode run on a moving guide on the
mode-set route as an **adiabatic** one; such a run warns once, by name
({py:class}`~solprop.exceptions.AdiabaticTaperWarning`). For a solid guide whose modes
move, the general route of the next section carries the connection.

## A guide whose geometry moves along z

A tapered step-index fibre, a graded-index multimode taper, a photonic lantern: a
**solid** guide whose geometry moves changes the *shape* of its modes along z, because
its normalised frequency moves with the core. A capillary's modes are
$J_0(u r/a)/a$, a pure function of $r/a$, so a tapered capillary is its entrance
transform times one scalar; a solid guide's synthesis and projection maps $S(z)$ and
$Q(z)$ genuinely move, every nonlinear overlap follows them, and the connection of the
previous section couples the channels. The general route propagates such a guide with
its connection carried.

**Declaring one is declaring the provider's radius path.** A provider whose region radii
follow a {py:class}`~solprop.profiles.ZProfile` that is not constant, or that declares a
kink, is sampled along its path by
{meth}`ModalGuide.from_provider <solprop.modes.guide.ModalGuide.from_provider>` as well as
in frequency, and runs through `Simulation.from_waveguide(provider=...)` like any other
guide:

```{code-block} python
import numpy as np
from solprop.driver import Simulation
from solprop.grid import FieldGrid
from solprop.modes import GradedIndexProvider, PowerLawPermittivity
from solprop.modes.guide import ModalGuide
from solprop.modes.material import IndexMaterial, MaterialKerr
from solprop.profiles import Linear

grid = FieldGrid.from_limits(1e-3, 1.03e-6, (1.0e-6, 1.065e-6), 0.4e-12)
taper = GradedIndexProvider(
    permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
    exterior=IndexMaterial(1.45 * np.sqrt(0.99)),
    radius=Linear(25e-6, 22.5e-6, 1e-3),  # the path: 25 to 22.5 um over a millimetre
    family="LP",
    count=2,
    elements=32,
    extent=2.0,
    core_responses=(MaterialKerr(chi3=2.2e-22),),
)
active = grid.omega[grid.active_mask]
knots = np.linspace(0.99 * active[0], 1.01 * active[-1], 25)
guide = ModalGuide.from_provider(
    taper, knots, nodes=taper.node_rule(knots[0], 0.0, n_r=2)
)
guide.sampling.path  # the sections, the degree its accuracy demanded, what it measured
run = Simulation.from_waveguide(
    (),
    1e-3,
    provider=guide,
    grid=grid,
    wavelength=1.03e-6,
    fwhm=60e-15,
    energy=1e-12,
    save_planes=[1e-3],
    backend="numpy_ref",
).run()
```

**How the path is sampled.** The provider's z domain is cut into **sections** at every
declared kink -- where the slope jumps and the connection with it -- and each kink
interval may be divided further with `sections=`. Each section is solved at the
Chebyshev-Lobatto points of one degree, starting at four and doubled until the expansion
reproduces a **fresh** solve at the points the doubling adds to within `path_rtol`, which
by default is the frequency interpolation's own measured profile error: sampling in z is
then never the coarser of the two
({py:class}`~solprop.modes.guide.ModalPath`,
{py:func}`~solprop.translate.modal_path.sample_modal_path`). On the graded-index taper
above the first degree, four, already reproduces the held-out solves to 7.6e-8 against a
budget of 2.9e-6, and the whole sampling takes a few seconds; the doubling costs no repeated solve, because the Chebyshev-Lobatto points nest.

**Each position is solved on the provider's own rule there.** The rule is asked for
again at every position exactly as the entrance rule was, so its panel edges sit on the
material interfaces *there* and the quadrature resolves each smooth region separately (a rule frozen at one
position is wrong at first order in the interface's displacement). Node $q$ is
then one point $x_q(z)$ moving smoothly with the geometry, in one material region all
along -- the graded-index mesh scales with the core, so its exit rule is its entrance
rule scaled by 0.9 exactly -- which is what lets a map stored at node $q$ be expanded in
z at all: on a frozen rule a node the interface sweeps over changes material half way
along, and its field has a kink or a jump there that no expansion resolves. The
connection keeps the union-split rule of the previous section, which is what its
Eulerian derivative needs; the two describe the same modes. Each position is kept in
**the provider's own gauge**, because the connection is measured in it, and a provider
whose own gauge is not continuous along the path -- a sign that flips, a phase that
drifts -- is refused by name.

**What a run carries.** The maps are one Chebyshev expansion per section, combined at
every stage with the stage position's interpolation weights. The linear
flow is the frozen-residual **coupled flow** of a varying guide: $\Phi$ carries the first
channel's **entrance** dispersion, and the flow carries every channel's departure from
it -- the first channel's own motion included -- and the connection $\Gamma$, frozen at
each section's midpoint with the rest in the explicit residual the Runge-Kutta source
carries -- on the bins the error norm scores only; outside them the frozen flow alone
carries the state (the reason is given under "Only where the error norm looks" in
[a guide that varies along z](coherent-wall.md#a-guide-that-varies-along-z)).
The connection is stored as measured: the provider's fields are
internal-convention and the synthesis stores them unconjugated, so the kernel's amplitude
is their coefficient and $\Gamma$ enters without a conjugate (tested with a circular
basis, where the two conventions differ). It needs no power conversion: the Poynting
amplitude is the true-power amplitude on this route, so the flux self-term
$\mathrm{Re}\,\Gamma_{nn} = \tfrac12 W'_{nn}$ is the whole of each channel's own
normalisation, which a capillary's flow carries in its diagonal instead.

**What it does, measured.** Against an independent DOP853 solve of the coupled-mode
equations -- the modes' own propagation constants at every position and a connection
sampled separately at its own resolution -- an $\mathrm{HE}_{11}$/$\mathrm{HE}_{12}$ pair
on a 10 % step-index taper over a millimetre agrees to 6e-9 in $b_2/b_1$ (a converted
amplitude of 3e-3) at 80 steps, falling faster than 16 per halving from 20. The same
radius change over 0.25, 0.5 and 1 mm converts 4.3, 2.1 and 1.1 % of the amplitude of a
graded-index taper: the conversion falls as $1/L$, which is the adiabatic limit, and the
launched channel's phase is its own $-\int(\beta(z) - \beta(0))\,dz$. The effective
nonlinear coefficient follows the mode area: on a weakly guiding step-index taper the
self-phase against the constant entrance guide is the path average of
$\gamma(z)/\gamma(0)$, and agrees with the scalar
{py:class}`~solprop.nonlinear.gnlse.SpecifiedMode` of the same fibre to the vector and
scalar areas' own departure.

**What it costs.** The path's solves and the connection's at every sampled position:
seconds on the graded-index sector and about twenty on a two-channel step-index fibre.
The maps hold `sections x (degree + 1)` coefficients each rather than one, and a stage
evaluates one of each, which {py:func}`~solprop.backends.memory.estimate_memory` counts
to the byte.

**Refused, by name**: a snapshot position other than the entrance, `sections=` or
`path_rtol=` beside a guide whose geometry does not move, an entrance rule the provider
would not build again, a run longer than the declared path, the `modified` noise model
(the `input` draw is part of the launch and is admitted), a lockstep batch, a fixed
schedule that straddles a kink, a CF4 flow or no flow at all beside moving maps. Every backend executes the moving maps — `numpy_ref`, `numba_cpu` and `native` on
the CPU and on a GPU. A declared circular basis is honoured: the metric then stays
the identity, the basis being unitary and the measured Gram departure unitarily
invariant.

**Reading a plane back.** A saved plane's amplitudes are coefficients of that plane's
own modes, on the rule that has moved there with the interfaces, so the field accessors
of {py:mod}`solprop.processing.modal` solve the guide again at the plane's own position
exactly as its path was sampled -- the provider's rule asked for as the entrance rule
was, in the provider's own gauge, carried to the run's bins by the same cubic in
frequency -- and reconstruct E and H there, on that position's nodes and area weights,
which {py:func}`~solprop.processing.modal.plane_nodes` returns for any saved plane.
The expansion the run steps with carries no H, which is why the solve is repeated
rather than read; it is the field the run's synthesis stores there to the path's
held-out accuracy. On the kinked graded-index taper the plane 0.6 mm in reconstructs, to
rounding, what a constant guide of the radius there reconstructs from the same
amplitudes (the entrance modes differ from it by 3.7 %), and the time integral of its
reconstructed flux is the run's `physical_energy` to 1.6e-8, inside the plane's own
Gram departure of 6.4e-5. `channel_amplitudes`, by contrast, recovers amplitudes on the
entrance reference only.

The [tapered graded-index worked example](../../auto_examples/waveguides/plot_05_moving_solid_guide.rst)
runs the route at full size: the $\mathrm{LP}_{01}\to\mathrm{LP}_{02}$ conversion of
eight tapers from 50 µm to 10 mm against a first-order coupled-mode estimate built from
fresh solves, and self-phase modulation in a taper's waist against the scalar area path.
