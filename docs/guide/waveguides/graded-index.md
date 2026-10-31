# Radial graded-index fibres

A graded-index fibre varies continuously across its core. The radial provider solves
selected bound sectors and their exterior continuation. It is appropriate when radial
symmetry and the declared sector approximation describe the guide; it is not a general
two-dimensional vector eigensolver. See the [solid-fibre workflow](solid-fibres.md) for
propagation.

## Radial graded-index providers

`PowerLawPermittivity` describes a core law
`epsilon(r)=epsilon_axis*(1-2*Delta*(r/a)**p)`, with `0<Delta<1/2` and `p>=2`.
The axis material and the exterior material are explicit. Neither material nor
radial law takes a `z` argument. `GradedIndexProvider.radius` may be a `ZProfile`;
its `parameter_paths` declares that geometry path separately, under the name
`core_radius` that every other parameterised mode uses, and `at_parameters`
accepts any subset of it.

```{code-block} python
import numpy as np

from solprop.constants import C
from solprop.modes import GradedIndexProvider, IndexMaterial, PowerLawPermittivity

grin = GradedIndexProvider(
    permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
    exterior=IndexMaterial(1.45 * np.sqrt(0.99)),
    radius=25e-6,
    family="TE",
    count=2,
    elements=64,
    extent=2.0,
)
rule = grin.node_rule(2 * np.pi * C / 1.03e-6)
fields = grin.solve(2 * np.pi * C / 1.03e-6, 0.0, nodes=rule)
tail = grin.tail_bound(2 * np.pi * C / 1.03e-6, outer_radius=50e-6)
```

The solver uses continuous linear finite elements for the radial TE₀ or TM₀
Maxwell sector and matches to a decaying Bessel exterior. It solves the resulting
frequency-dependent boundary eigenproblem; it does not multiply transfer matrices
through thick evanescent layers. `elements` and `extent` are explicit numerical
choices. Refine element count, matching radius and quadrature independently.
The analytic exterior integral certifies the omitted tail of the discretised
mode; it does not certify the finite-element error. `boundary_residual` includes
the one-sided Maxwell trace error at the material interface. Linear elements
have first-order derivative traces even when their eigenvalues converge at second
order.

The same thing holds pointwise inside an element. The strong-form curl residual
of a P1 field does not converge at all -- it stays at about
$|\beta^2-k_0^2\epsilon|/(k_0^2\epsilon)$, measured 2.06e-3 relative from 32 to
256 elements -- because a linear element has no second derivative. Integrated
functionals built from the same field do converge at second order (a full-vector
Kerr integral refines by factors 3.95 and 3.99 over the same sequence), so
pointwise $E_z$ near the interface carries about 1 % error at 96 elements while
the integrals over it are second-order accurate. Refine for anything that reads
the longitudinal field at a point rather than under an integral.

A channel above cutoff is not necessarily bound *on this domain*. If more than
`MAX_EXTERIOR_TAIL_FRACTION` (1e-4) of a channel's power lies beyond the matching
radius, the exact Bessel continuation rather than the discretised core is
carrying the mode, and the provider refuses that channel by name. The floor 1e-4 is a
judgement that has not been verified: the reference cases reach 6.6e-6, a factor 15
below it, and a channel that should be refused reads 3.2e-4, a factor 3 above, but no
reference yet ties the fraction to the error of an observable. A refusal near the floor
is a prompt to widen the matching radius. The fraction is computed exactly, from the
analytic modified-Bessel tail integral over the same
full-domain norm the fields are normalised by; it is written as a power fraction
rather than as the dimensionless exterior decay because the decay also depends on
the numerical choice of `extent`. For reference, a five-channel TE case at 48
elements keeps 5.97e-8 of its last channel outside a 25 um core's matching radius
and 3.24e-4 outside a 22.98 um one.

`family="LP"` selects a **weak-guidance scalar approximation**, with one transverse
linear polarisation and no longitudinal field. It requires relative permittivity
contrast at most 0.02. The supplied tests compare its propagation constants with
vector TE/TM sectors and the ideal parabolic ladder; this is not a general bound
on nonlinear overlaps or longitudinal-field errors. **Hybrid HE/EH GRIN is refused
by name.** TE₀/TM₀ evidence does not qualify it.

### A guide from a constructed provider

A channel's propagation constant is the generator's diagonal, so the bridge
refuses a generator that couples its channels (`MAX_GENERATOR_COUPLING`): a
coupled basis has no per-channel constant, and its flow is a coupled flow
([general modal propagation](modal-bases.md#a-declared-basis-of-a-block-that-is-only-nearly-degenerate)),
not a table of indices. The off-diagonal is measured against the **smallest
splitting between the channels**, $\min|\beta_i-\beta_j|$, and not against the size
of $\beta$: two channels mix over a length set by the ratio of their coupling to
their splitting, so the relevant small parameter is the coupling divided by the splitting.
For example, a lossless two-channel model with coupling $\kappa=10^{-3}$ m$^{-1}$
and splitting $\Delta\beta=1$ m$^{-1}$ has maximum power transfer
$4\kappa^2/(\Delta\beta^2+4\kappa^2)\simeq4\times10^{-6}$, obtained by
diagonalising its $2\times2$ propagation matrix. Comparing the same coupling
with a carrier $\beta=10^7$ m$^{-1}$ would conceal that finite transfer. The splitting is floored at $64\,\epsilon\,|\beta|_{\max}$
so that a single channel and an exactly degenerate pair still have a scale.

A provider's generator is in the **internal** convention
([conventions §2](../../dev/conventions.md)), $G_{cc} = -i(\omega/c)\,\overline{n_{\rm
eff}}$, so its imaginary part is $-\beta$ and its real part is $-\alpha/2$: a passive
channel has $\mathrm{Re}\,G \le 0$, and the bridge refuses a positive real part as
gain by name. An `IndexTable` is a record in the user-facing *optics* convention, where
$\mathrm{Im}\,n_{\rm eff} > 0$ is loss, so the bridge writes
$n_{\rm eff} = \overline{i G c/\omega}$. That conjugation is undone when a run builds
its linear operator from the table, the same round trip any optical-convention table
makes.

`ModalGuide.from_provider(provider, omega)` pairs a provider that solves its own
fields with the ordered modes the rest of the package needs. The provider owns
its sector, channel count, elements, extent and radius path, exactly as
`from_step_index` takes constructed modes: nothing in the constructor invents a
radial resolution, and there is no second root solver behind a `Mode` interface.

```{code-block} python
from solprop.modes import ModalGuide

omega0 = 2 * np.pi * C / 1.03e-6  # the centre of the band the guide will be used over
omega = np.linspace(0.98 * omega0, 1.02 * omega0, 9)
guide = ModalGuide.from_provider(grin, omega)
```

With no rule supplied the constructor asks the provider for one at the **lowest**
frequency of the band, where a bound field reaches furthest and a truncating rule
discards most power, so that the rule's certified `tail_power_bound` is a bound over
the whole band. Both providers name the frequency the certificate was taken at in
the rule's identity, as `tail_power_bound_omega`.

`degenerate_blocks` names the channels whose degeneracy the provider declares — the
two parities of an `HE_1m` channel, say, as `((0, 1),)` in the provider's own channel
order. Tracking then rotates freely *inside* a block and matches one-to-one outside
it, and the generator is rotated with the fields, so a degenerate pair stays
physically equivalent instead of being matched by an ordering its equal eigenvalues
cannot resolve. It is empty by default, because an arbitrary provider declares no
degeneracy, and the blocks are kept on the table so that the propagation-constant
checks below align a fresh solve the same way.

```{code-block} python
# strand: a step-index provider, ModalGuide.from_step_index(...).field_provider
pair = ModalGuide.from_provider(strand, omega, degenerate_blocks=((0, 1),))
```

Each declared channel becomes a `ProviderChannelMode`: a dispersion carrier whose
propagation constant is the provider's own generator at the knots, read off the
tracked diagonal so the channel is followed by physical overlap rather than by
eigenvalue order, and whose `beta_1` and `beta_2` are **the interpolant's**, the
dense table's cubic in frequency differentiated, because a provider declares no
analytic frequency derivative. The constructor solves the whole band once, into a
single `sample_modal_table`, and reads every channel out of that one table;
`solprop.translate.modal_tables.sample_provider_table(provider, channel, omega)` is
the same bridge for one channel on its own, the sibling of `sample_index_table` for
a provider.

The constructor then asks the provider again, at the entrance frequency and midway
across the first, the middle and the last gap between knots (the ends of a band are
where the cubic interpolant is worst), and refuses a table whose
propagation constant disagrees by more than `beta_rtol` (default
`PROVIDER_BETA_RTOL`, 1e-6). A provider that names no channels cannot be paired at
all. A guide whose geometry does not move is a snapshot at one position, and every
`ProviderChannelMode` reports `z_constant`. A provider whose geometry **moves** — a
radius path that is not constant, or any longitudinal breakpoint — is sampled along its
path as well: its z domain is cut into sections at every declared kink, each divided
`sections=` times, and every section is solved at the Chebyshev–Lobatto points of the
degree its own held-out check demands, on the provider's own rule at each position;
each channel then carries its index and area along the path. The checks above are taken
at the entrance, and a moving guide's departure along z is held to `path_rtol`
(by default the frequency interpolation's own measured error, with a floor).
Propagating such a guide is described under
[a guide whose geometry moves along z](moving-modal.md#a-guide-whose-geometry-moves-along-z).

**That check certifies `beta` and nothing else.** `validate_modal_table` measures a
profile, a generator and a flux defect; it measures no frequency derivative, and
the guide checks none. `beta_1` is the interpolant's: measured held out against
five-point differences of fresh solves on the reference parabolic sector, over a
±2 % band, it is 4.4e-13 relative at nine knots. `beta_2` is **diagnostic and
uncertified**: on the same sector and the same held-out frequencies it is 2.8e-2
relative at nine knots, 3.2e-3 at seventeen and 2.1e-3 at thirty-three, and worse
than unity over a ±20 % band at every knot count measured. A cubic through the
knots reproduces a nearly constant group delay to twelve digits and still carries
almost no curvature. Whatever reads a second derivative from such a channel —
`modes.base.beta2` itself, `modes.base.zdw`, `tools.dispersion_length`,
`tools.soliton_order` — gets an order of magnitude, not a certified number.

`modes.base.beta2` differences `beta1` at 0.3 % of the frequency and its Richardson
step doubles that, so it reaches 0.6 % either side of the point asked about. A
channel table never extrapolates past its knots, so `beta_2` nearer than that to
the edge of the sampled band is refused by name rather than guessed.

A `ProviderChannelMode` has no transverse profile: `field` refuses by name,
because the physical fields of the channel are the provider's three-component E
and H on a node rule. Its `aeff` is the scalar surrogate
`(int |E|^2 dA)^2 / int |E|^4 dA` summed over the three components, which is what
the mode-averaged interfaces mean by an area and **not** the nonlinear
coefficient of a vector channel; that is the full Kerr contraction
(`modal_kerr_coefficient`) over the regions that respond.

That surrogate is one number for the whole band, and it is taken at the **median
knot**, the middle of the band the guide was sampled over. The channel table records
which frequency that was, as `source_identity["channel_area_omega"]`. It matters:
on the reference parabolic sector the lowest knot's area is 2.0 % above the median
knot's over a ±2 % band, 25 % over ±20 % and 66 % over ±40 %. A frequency-dependent
area does not reach the scalar route at all; the table carries the area at every
knot, and a consumer that wants it as a function of frequency reads
`effective_area(omega)` from the table rather than `aeff()` from the mode.

To have the area at one chosen frequency -- a scalar model's pump -- ask the channel
itself: `solved_area(omega, z)` solves the provider again there on the guide's own
rule (for a moving guide, the rule the provider builds at `z`, as its path was
sampled), and `area_at(omega)` does so at every sampled position of a moving guide's
path and returns the area as a path along z. Every channel a
`ModalGuide.from_provider` builds carries its provider for this (a `ChannelSource`,
left out of the channel's serialised record). `SpecifiedMode.from_mode(channel,
wavelength=...)` uses it, and `prop_gnlse` passes its `wavelength`
([GNLSE page](gnlse.md)). Without it `from_mode` falls back to the median-knot
surrogate, which on a graded-index guide can be tens of per cent off the area at the
pump.
