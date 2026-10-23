# Channel labels and modal tables

Channel labels identify physical modes as frequency or geometry changes. A modal table
samples their fields and propagation constants so a run can reconstruct a
frequency-dependent basis. Resolution, overlap tracking, cutoff and omitted exterior
power all affect whether that basis remains accurate.

## Channel labels and dense reference tables

Every mode carries a `ChannelLabel`: its family, azimuthal and radial order,
parity, handedness and a stable display label. It is a member of the `Mode`
protocol, so the label belongs to the mode rather than to its place in a set, and
`ModeCollection.channel_labels` is simply its members' labels in the order of the
mode axis. A `GradedIndexProvider` produces one label per requested radial
channel, and both providers publish them in `identity["channels"]`, which is what
`ModalGuide` checks its mode collection against. A label a family does not have is
`None`, never a string standing for one: an axially symmetric TE0m or TM0m
channel has no parity partner, so its `parity` is `None`. `handedness` is `None`
on every real basis, and is **filled** where a guide declares a complex one: the
circular partners of {py:func}`solprop.modes.guide.circular_pair` carry `left` and
`right` in place of `even` and `odd`, by the convention
{doc}`general-modal` derives and {doc}`../setup/polarisation` fixes the sign of.

**Parity means one thing across every family**, and `ChannelLabel`'s docstring is
where it is defined: `"even"` is the member whose transverse pattern is the
$\cos((n-1)\theta)$ one about the **x** axis — for $n=1$ that is the x-polarised
member — and `"odd"` is its quarter-turn partner. Each family reaches it its own
way. An exact step-index $HE_{nm}$ builds the pair from $\cos(n\theta)$ and
$\sin(n\theta)$ on the *radial* component, and its "even" member is x-polarised on
the axis. A Marcatili $HE_{nm}$ is a rotation $\phi$ of one pattern, and its "even"
member is the unrotated $\phi = 0$ — so `MarcatiliMode(125e-6, "He", 1.0)`, which is
x-polarised, is `HE1,1:even`, and its `orthogonal_polarisation`, the quarter turn
$\phi = \pi/(2n)$, is `HE1,1:odd`. Any other rotation is a gauge and not
a parity: its `parity` is `None` and the angle goes in the display label. Note that
the label's `azimuthal_order` is the family index $n$ — unlike the mode property of the
same name, which is the field's highest harmonic — while the parity names the
$(n-1)$-fold pattern the transverse vector field has.

Repeating a step-index parity or
adding a second TE/TM parity is refused. A common physical step-index guide may
have different mode-specific cladding tail radii; its collection uses a common
enclosing domain.

`sample_index_table(mode, omega, z=..., include_area=True)` is the dense dispersion
bridge beside `IndexTable`. It retains the source's frequency derivative in a
Hermite interpolant and records the snapshot position and source identity.
Step-index areas are sampled at each frequency. Other modes retain their declared
frozen-profile area. The resulting table does not supply electric or magnetic
fields, and never extrapolates beyond its knots. It is an `IndexTable` subclass,
so it also carries `derivative_values` and `source_identity`, and it is built
with `loss_interpolation="linear"` rather than the table default `"log"`: a
lossless source has an imaginary index of exactly zero, which has no logarithm
and which the logarithmic policy refuses outright.

`solprop.translate.modal_tables.sample_modal_table` retains full E/H samples on
common physical nodes. It assigns channels by physical overlap, aligns their
phases and optionally rotates declared degenerate subspaces. It rotates the
complete generator and metric with the fields, including a near-degenerate block.
The blocks a table was tracked with are retained on it, so a held-out comparison
aligns a fresh solve in the same declared subspaces. A table interpolates over one
z snapshot or at least four z knots, because SciPy's not-a-knot cubic degrades
silently to a parabola on three positions and to a straight line on two. An
interpolated sample carries the largest `boundary_residual` of its bracketing
knots. No compression is performed.

Construction **gates the sampled Gram**. A provider normalises over its whole
physical domain and deliberately does not renormalise a caller's rule, so a rule
that truncates or under-resolves the fields returns a basis whose sampled
$W$ is not the identity while every sample is still `valid`. A table refuses a
knot whose $\lVert W-I\rVert_2$ exceeds `gram_atol`, whose default
`DEFAULT_GRAM_ATOL` is 1e-4 -- the loosest defect a provider here can
legitimately produce, since the radial provider admits up to
`MAX_EXTERIOR_TAIL_FRACTION` of a channel's power beyond its matching radius and
the sampled Gram is short by exactly the power the rule discards. The largest
defect over the knots stays on the record as `gram_defect`. On the 250 nm strand
it is 0.31, 2.41e-4, 1.24e-5, 2.92e-8 and 2.03e-14 at radial orders 3, 12, 16, 24
and 48; the six-channel 1.5 um guide is at rounding (1.33e-15) from order 24, and
a parabolic GRIN sector is at 4.2e-14 on its own element rule.

The **cutoff policy of a table** is the providers' own, applied when the table is
built: a band in which a channel is below cutoff, or too near it for the frozen
profile to be resolved, is refused by name at construction, and a table never
extrapolates past its knots into a band it does not carry. Nothing is masked and
no channel is silently dropped.

Tracking follows the **physical channel through a crossing**: the assignment is
by the overlap of the fields, so two channels whose propagation constants exchange
order across the band keep their identities, where sorting by eigenvalue would
swap them.

`modal_transform_tables(table, omega, profile_error=...)` builds
`ModalTransformTables`, the flat record a run's kernel problem carries as its
`transform.modal_tables` leaf. It evaluates the table on the frequency grid a run will
use -- which need not be the table's own knots, and in general is not -- and stores the
fields, the reciprocal partners, the coefficient dual, the two execution maps (the
synthesis and projection of [general modal
propagation](modal-projection.md#the-synthesis-and-projection-maps))
and the node geometry as arrays, with frequency last and contiguous. Outside the
table's band the record is explicitly zero rather than extrapolated, and a run whose
active band leaves the guide's support is refused when the problem is assembled. The
execution maps are built for a snapshot table only; a moving guide's maps come from the
path sampling described above.

`coordinate_dual` recovers amplitudes from a physical field: contracted against
the six components `(E, H)` at every node it returns the amplitude vector that
synthesised them, because it is `W^-1` times the same Hermitian Poynting pairing
the flux metric is built from. Its first three columns are in sqrt(W) m/V and its
last three in sqrt(W) m/A, and its two longitudinal columns are zero, since only
the transverse components carry longitudinal flux. It is for amplitude recovery,
not for nonlinear projection, which carries the separate `-i omega` of
`source_frequency_factor`.

`profile_error` is declared by the caller, per interval and coarse frequency:
nothing in the constructor infers the accuracy of its own representation.
A snapshot declares one interval; a path declares one per pair of stored
positions.

`validate_modal_table` compares held-out solves
against budgets supplied before the comparison; frequency derivatives can be
checked through `generator_derivative`. Physical kinks require separate sections.
Besides the weighted field and generator defects it reports the flux defect
$\lVert W-I\rVert_2$ of the interpolated basis, which is what says how far the
interpolant has stopped being an orthonormal set in true power. That column falls
with the frequency knots like the field defect (2.25e-8, 1.13e-9 and 5.96e-13 for
5, 9 and 17 knots on the 250 nm strand), but at a held-out *position* it is a
property of the provider rather than of the interpolation: the node rule is
frozen at the entrance radius, so along a taper the panel edge no longer sits on
the material interface and the defect grows linearly with the displacement
(4.84e-7, 9.68e-7 and 1.94e-6 at z = 0.25, 0.5 and 1.0 m for a 1e-5 relative
radius change). Budget that floor, or rebuild the rule along the path, as
`ModalGuide.from_provider` does for a moving guide.

A guide built by `ModalGuide.from_provider` is something to **run**, not only to
inspect: `Simulation.from_waveguide(..., provider=guide)` and
`prop_waveguide(..., provider=guide)` propagate its three-component fields on
`numpy_ref`, `numba_cpu` and `native` (on the CPU and on the qualified GPU builds), and
[the general modal page](general-modal.md) is the whole description. The guide carries
its own {class}`~solprop.modes.guide.ModalSampling`
-- the knots, the position, the node rule, the declared degenerate blocks and the
measured profile interpolation error -- and a run rebuilds its dense reference
from that; a guide built by `from_step_index` carries none, because nothing there
declares a transverse resolution, and a run refuses it by name. A region's
nonlinear response is declared on the region, with
{class}`~solprop.modes.material.MaterialKerr`, by `chi3` or by `n2`. A
{class}`~solprop.modes.graded_index.GradedIndexProvider` builds its own two regions
and so takes its core's responses as `core_responses=(...)`; its exterior is always
linear, because a bound radial sector's exterior is the exact Bessel continuation and
holds at most {py:data}`~solprop.modes.graded_index.MAX_EXTERIOR_TAIL_FRACTION` of the
power, at nodes the rule barely resolves.
