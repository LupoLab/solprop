# Circular and coupled modal bases

A change of basis can express the same guided field using different channels, such as
circular instead of linear polarisation. It must transform the fields, power metric and
linear propagation together. A rotated pair is independently propagating only when its
propagation constants are exactly degenerate.

## Declaring a circular basis

A guide's channels arrive in whatever basis its provider solved them in, which for an
azimuthally degenerate family is the real parity pair: the $\cos$ and $\sin$ patterns
{py:class}`~solprop.modes.protocol.ChannelLabel` defines. Any **unitary** combination of
an *exactly degenerate* block is as good a basis, and one of them — the circular pair —
diagonalises the isotropic Kerr tensor's polarisation dependence: for a field that is
circular at every node $\mathbf E\cdot\mathbf E$ vanishes, so $(\mathbf E\cdot\mathbf
E)\mathbf E^*$ drops out of the cubic and the self coefficient falls to two thirds of the
linear one.

{py:func}`~solprop.modes.guide.circular_pair` declares it, and
{py:meth}`~solprop.modes.guide.ModalGuide.from_provider` takes it as `basis=`:

```{code-block} python
from solprop.modes.guide import ModalGuide, circular_pair, provider_channels

# provider: a field provider whose channels 0 and 1 are an exactly degenerate parity
# pair; omega: the knots; rule: its node rule, all as in the first example.
basis = circular_pair(provider_channels(provider), (0, 1))
guide = ModalGuide.from_provider(
    provider, omega, nodes=rule, degenerate_blocks=((0, 1),), basis=basis
)
# The channels are now HE1,1:left and HE1,1:right, with handedness and no parity.
```

The transform is applied to every knot **after** the tracking gauge, in that fixed
order, and the composed identity is `<tracking gauge>+circular(0,1)`. Everything
downstream follows from the one rule $\mathbf e' = \mathbf e\,U$: the maps become
$S' = S\,U$ and $Q' = U^{\mathsf H} Q$, the metric $W' = U^{\mathsf H} W U$, and a
physical field's amplitudes $\mathbf d = U^{\mathsf H}\mathbf c$.

**The convention.** The even partner is the $\cos((n-1)\theta)$ pattern about $x$ —
for $n = 1$ the x-polarised member — and the odd one its quarter-turn partner. A
provider's fields are in the **internal** convention, whose plane wave is
$e^{i(\omega t - \beta z)}$, so the channel $(\mathbf e_{\rm even} - i\,\mathbf
e_{\rm odd})/\sqrt2$ has the physical transverse field $(\cos\omega t, \sin\omega t)$:
it turns from $x$ towards $y$, which an observer looking towards the source sees turn
anticlockwise. That is positive helicity, which optics calls **left** circular, and its
optics-convention Jones vector $(1, i)/\sqrt2$ is the one {doc}`../setup/polarisation` records
as positive Stokes $V$. So `block[0]` becomes the **left** channel and `block[1]` the
**right** one. The order is not free: `block[0]` must be the **even** member and
`block[1]` the odd one, and a block given the other way round is refused by name.

**Three gates, each refusing by name.** The declaration itself must be unitary to
`BASIS_UNITARY_ATOL` (1e-12); each block it acts inside must be one the guide declared
degenerate; and
the transform must not change the basis's measured departure from orthonormal, which the
spectral norm's unitary invariance says a unitary one cannot — to the declaration's own
`BASIS_UNITARY_ATOL`, because a transform unitary only to that budget moves the measured
norm by its own unitary defect, so a tighter gate here would refuse what the declaration
has just admitted. The route and the backend then gate the stored metric against
`MAX_DENSE_METRIC_DEPARTURE` (1e-3), which is the loosest of the three and the one a
*backend* can apply, since it sees the block and not the declaration.

**The metric is then genuinely dense.** $W' = I + U^{\mathsf H}\delta U$ with $\delta$
the sampled Gram defect: the exact identity is invariant but the measured object is not,
and the off-diagonal of $U^{\mathsf H}\delta U$ is not $\delta$'s. So
`linear.modal_metric` becomes `kind="dense"`, built from the table's own sampled flux
metric per frequency rather than from a literal, and the identity outside the guide's
band, where nothing lives and a singular block would certify nothing.

The declaration is opt-in: a guide built without `basis=` stays in the provider's own
coordinates, bit for bit.

## A declared basis of a block that is only nearly degenerate

$U^{\mathsf H}\operatorname{diag}(\beta)U$ is diagonal **only** where the block's
propagation constants are equal. Declare a basis over a block whose members are split —
and every "degenerate" group of a real guide is split by something: the vector
correction, a stress, a bend — and the rotated channels genuinely **beat**. That is not
an error, it is the physics of the coordinates asked for, and the run carries it in
`linear.coupled_flow`.

```{code-block} python
from solprop.modes.guide import ChannelBasis, ModalGuide

root = 1.0 / np.sqrt(2.0)
basis = ChannelBasis(
    np.array([[root, -root], [root, root]], dtype=complex),  # a quarter turn
    ((0, 1),),  # the block it acts in
    rotated_labels,  # a ChannelLabel for each channel after the rotation
    "lp11(0,1)",
)
guide = ModalGuide.from_provider(
    provider, omega, nodes=rule, degenerate_blocks=((0, 1),), basis=basis
)
```

**What the route does with it.** The transformed generator is
*evaluated* rather than assumed diagonal — for an exactly degenerate block it is carried
over bit for bit and no coupled flow is built — and
{py:func}`~solprop.translate.general.coupled_block` reads it at the run's own bins
relative to one **reference channel**, the first. The assembly then builds the
cumulative $\Phi$ from that channel alone, replicated, so that $\Phi$ is a multiple of
the identity in the channel axis and commutes with the block, and puts every channel's
departure from it — its own dispersion included — in the leaf beside it. The leaf is one
`"constant_schur"` section, because the geometry does not move: only the coordinates are
rotated, so the block is the same matrix at every position.

**What it costs.** The whole split is in the leaf's off-diagonal, and over a length $L$
a launch into one rotated channel arrives with $\sin^2(|G_{01}|L)$ of its power in the
other — 3.8e-3 over a millimetre on the `TE_01`/`TM_01` pair of a strand of 6 µm core
radius, whose vector splitting is 120 1/m at 835 nm, 1.1e-5 of its own propagation
constants (on a 3 µm core it is 744 1/m and 6.9e-5). The same launch in the
provider's own coordinates, carried through $U$, reproduces it: measured 8.4e-6 in each
channel's modulus and 2.1e-5 rad in the phase between them, which is what says the flow
is the same physics in other coordinates and not a second model. Both are measured inside
the **declared window**, where the linear operator is the guide's own; outside it the grid
carries the absorbing collar, which two guides with different index tables carry
differently, and the same comparison reads 1e-3.

**Where it runs, and what it refuses by name.** Every backend executes the leaf —
`numpy_ref`, `numba_cpu` and `native`, on the CPU and on a GPU — except a CF4 section,
which runs on `numpy_ref` alone and which the entry point refuses elsewhere before
anything is translated. A **noise** model beside it is refused, because the shot-noise
draw ({doc}`../media/noise`) is transported by the unitary phase of a *diagonal* flow and its
transport through a coupled one has not been measured. The **lockstep batch** refuses a
member carrying one, because it evaluates its own stage arithmetic and applies neither
the flow action nor its residual. And a block whose exponent over the largest declared
step exceeds `MAX_GENERAL_FLOW_EXPONENT` (1e3) is refused: the block carries every
channel's departure from the reference, so a basis declared over channels that were
never a degenerate group would put their whole dispersion difference into a matrix
exponential taken by scaling and squaring.

The run records what it propagated. `modal.flow` joins the schema-7 identity with the
section methods, the degree, the reference channel, the three certificate rows and the
largest coupling, so a resume against a changed block is refused by the block that
moved. The key is written **only** where a flow exists; a run without one has no
`modal.flow` entry at all. Its `model` reads `"declared channel basis"`
and its `gauge_id` ends in the declaration's own name (`+lp11(0,1)` above); the entries it
shares with the capillary's coherent wall mean the same things there, and are tabulated
under {ref}`reading the record <coherent-wall-record>`.

**What a rotated channel reports, and what it withholds.** A channel of a split block
has no propagation constant of its own — it beats — so the per-channel index the guide
publishes for it is the block's **mean**, read off the diagonal of the coupled
generator: both rotated `TE_01`/`TM_01` channels above report
$n_{\rm eff} = 1.447776101449$ at 835 nm, where the provider's own channels report
1.447784077518 and 1.447768125380. That mean is what the surrogate area, the dimension
limits and the labels need, and the run itself reads only the reference channel's.
`ProviderChannelMode.neff` on such a channel returns this mean: **do not read it as a
propagation constant**, because the channel has none. The one statistic built from per-channel
propagation constants, `zdw`, is **withheld** on a coupled block — it is left out of the
record's list of admitted accessors, and asked for by name it is refused at the driver,
the backend and the file — because a zero-dispersion wavelength of a mean propagates
nothing.
