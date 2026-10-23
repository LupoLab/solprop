# Raman data and validity

The response strengths, line positions, damping and collision partners must describe the chosen material and state. Built-in gases use optmat data with recorded sources and qualification ranges. This chapter explains how those data enter a run; the [model chapter](raman-models.md) gives the equations and the [workflow](raman.md) shows how to select a response.

## Raman responses from optmat

Every built-in gas's Raman response is optmat's
([SEP 3](../../dev/seps/archive/sep-0003-optmat-materials.md),
[materials from optmat](material-selection.md#materials-from-optmat)). optmat's gas Raman
responses are its per-line models of N₂, O₂, H₂, D₂, CH₄, SF₆ and N₂O, and of their
mixtures, with the available gain and revival evidence recorded per response.
The coverage is not uniform: N₂O retains provisional bands and requires opt-in. They are the response of
every capillary and waveguide fill of built-in gases; the [molecular models](raman-models.md) are a
supplied species' route.

## Selecting one

A run takes optmat's response when its populations are built-in gases and its
{class}`~solprop.physdata.selection.MaterialSelection` — `MaterialSelection()` unless
given — takes its Raman family from optmat: on
{py:func}`~solprop.driver.interface.prop_capillary` and
{py:func}`~solprop.driver.interface.prop_waveguide`, and in
{func}`~solprop.nonlinear.mixtures.responses_for`. The `raman` keyword keeps its vocabulary (`True`,
labels, a mapping to `MolecularOptions`). For one gas outside a mixture,
`response_for(gas, selection=..., density=...)` builds it
({func}`~solprop.nonlinear.optmat_raman.gas_raman`). The selection's `raman` field
chooses the response:

| `MaterialSelection.raman` | response |
|---|---|
| `"recommended"` (default) | optmat's curated response for the run's composition, temperature and carrier |
| `"provisional"` | the same, and admits N₂O's provisional vibrational bands, with a one-time {class}`~solprop.exceptions.MaterialStatusWarning`; a capillary's `raman=None` then turns N₂O's response on. Without it N₂O is refused by name, and `raman=None` leaves it off with a warning |
| an optmat response object | taken as it is, for instance `raman_response("N2", 293.15, wavelength=800e-9).with_parameters(width_scale=0.8)`; it must be for the run's composition, reference temperature and carrier, and is refused otherwise |

optmat's overrides (`width_scale`, `amplitude_scale`, and `width_law`, a law
$(c, d, \beta, \Delta\nu_D)$ per band) are the one-line path for a modified width or
strength; the response then has the status `user` and keeps the digest of the curated
response it started from. A {class}`~solprop.physdata.mixtures.Component`'s
`dephasing` is mapped onto `width_law` for every band of its species. A supplied law
with a Dicke term is refused, because optmat caps that term at a Doppler width the
legacy law does not state; pass the four numbers through `width_law` instead.

## The same convention, mapped onto terms

optmat's convention is this page's: for a medium of total number density $N$,

$$
P_{R,x}(t) = \varepsilon_0 N E_x(t) \int_0^\infty h(\tau; N)\, E_x^2(t-\tau)\,
\mathrm{d}\tau,
\qquad
h(\tau; N) = \Theta(\tau) \sum_k A_k \sin(\Omega_k \tau)\, e^{-\Gamma_k(N)\tau},
$$

with $A_k$ in m⁵ V⁻² s⁻¹ and $\Gamma_k = \pi\Delta\nu_k$, the full width at half
maximum $\Delta\nu_k$ in Hz: the amplitude decay $e^{-\Gamma t}$ has a Lorentzian half
width $\Gamma$ in angular frequency, so a full width $\Gamma/\pi$ in Hz. The widths are
$\Delta\nu = c + \min(d/n, \Delta\nu_D) + \beta n$ with $n$ in amagat: a constant, a
Dicke-narrowing term capped at the forward-scattering Doppler width, and collisional
broadening. One response is built for one composition, one temperature and one pump
wavelength, and its amplitudes already carry each constituent's mole fraction, so a
mixture's response is driven by its **total** density.

Lines that share a width law and a shift form a **damping group**. Each group $g$ and
tensor family $\kappa$ becomes one existing Raman term,

$$
u_{g\kappa}(t) = w(t) \sum_{k\in g} A_{k\kappa} \sin(\Omega_k t),
\qquad
\gamma_g(z) = \pi\,\Delta\nu_g\big(n(z)\big),
$$

with $w$ the grid's causal taper ([sampling the kernel](raman-numerics.md#sampling-the-kernel-and-the-density-it-depends-on)):
an undamped row and a decay-rate path, exactly the two arrays the kernel contract
already carries. The mapping is **exact** for every gas whose lines do not shift, at
any density path: the rows do not depend on the density, and the decay rate is optmat's
own group law, evaluated by optmat. An all-zero row is dropped and groups are never
merged. The counts at 293.15 K and 800 nm:

| response | lines | damping groups | terms (rows) |
|---|---:|---:|---:|
| N₂ | 106 | 43 | 72 |
| O₂ | 64 | 39 | 54 |
| H₂ | 20 | 3 | 4 |
| D₂ | 30 | 12 | 17 |
| CH₄, SF₆ | 1 each | 1 | 1 |
| dry air (N₂, O₂, Ar) | 170 | 82 | 126 |

against two terms for solprop's own N₂ model.

**Tensor form.** optmat gives each line an isotropic and an anisotropic amplitude, in
Hellwarth's decomposition
$R_{ijkl} = a\,\delta_{ij}\delta_{kl} + \tfrac12 b(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})$
with $a + b$ = iso + aniso, $a$ = iso − aniso/2 and $b$ = 3 aniso/2. The isotropic
amplitudes become an **isotropic** term and the anisotropic ones a **rank-two** term,
with the [tensor weights](raman-vector.md#declaring-and-using-tensor-terms) $(1, 0)$ and
$(1/4, 3/4)$. The tests verify this against the tensor directly: the two-component
polarisation equals Hellwarth's with optmat's $a$ and $b$ summed by a direct causal sum,
and a probe polarised across a rotational response sees $-\tfrac12$ of what a probe
along it sees. The rotational S lines are anisotropic only; a Q line's anisotropic
share follows its Placzek–Teller factor. optmat leaves out the vibrational O and S
branches, which carry 3 % (H₂, D₂) to 19 % (O₂) of a band's anisotropic share.

**Pump dispersion.** optmat disperses the vibrational strengths to the pump wavelength
it is given. The response always takes the run's carrier: the selection's
`carrier_wavelength` when it names one, otherwise the grid reference `lambda0`.

**Temperature.** A response fixes its temperature, which sets both the populations and
the widths' temperature laws. A temperature path along z takes the existing
[reference-temperature rule](raman-models.md#the-rotational-ladder), weighted now by the
composition's summed density, or an explicit `MolecularOptions(temperature=...)`, and
warns that the populations *and* the width laws are held there.

## Mixtures and their collision partners

A mixture's response is **one** response, for the composition of all its labelled
populations, Raman-active or not: an atom is a collision partner. Two labels of one
species merge into one constituent. The labels select whose lines are kept
(`raman=("N2",)` keeps nitrogen's lines, still broadened by oxygen and argon), and a
mapping's `MolecularOptions` which bands. In the translation, the response's one leaf
is driven and damped by the **summed density of the labels that form it**: one
amplitude table and one decay-rate path per term, within the
[kernel contract](../../dev/kernel-contract.md).

The composition must not vary along z. Populations whose densities are not
proportional, independent partial pressures for instance, are refused by name. The
partial-pressure rule keeps the fractions of a gradient fill proportional only to the
gases' differing non-ideality, which moves argon's share of dry air by
$1.6\times10^{-3}$ of itself between 10 bar and vacuum, and the composition's measured
spread is recorded; the tolerance is
{data}`~solprop.nonlinear.optmat_raman.COMPOSITION_TOLERANCE`, 1 %.

optmat builds a mixture's response only from the collision partners whose broadening it
has curated: N₂ and O₂ with each other and with Ar, SF₆ with any atom, and H₂, D₂ and
CH₄ with no other gas. Any other mixture with Raman is **refused by name**, naming the
pair optmat lacks (`optmat has curated no Raman broadening of H2 by He`). solprop does
not substitute self-broadening for the missing partner: the per-component,
self-broadened responses of [several responding species](raman.md#several-responding-species)
are not available under a selection. A fill whose populations are all atoms has no
Raman response and is not refused. A population optmat does not resolve (a supplied
species, or the dispersion-only `Gas.Air`) cannot be a partner beside a Raman-active
molecule, and is refused there.

## H₂'s line shift

H₂'s Q lines shift with density, $\Omega_k(N) = \Omega_{k,0} + 2\pi s n$, so their row
depends on the density.

- At a **constant density** the row is built at that density, which is exact.
- On a **varying density** the response is refused by name, unless the selection names
  the approximation `raman_shift="reference-density"`. That holds the shift at the
  density-weighted reference density $N_\text{ref} = \int N^2\,\mathrm{d}z /
  \int N\,\mathrm{d}z$, the analogue of the reference temperature, and records the
  phase bound $2\pi|s|\max_z|n(z) - n_\text{ref}|\,T_w$ over the grid's window $T_w$.
  Each line's phase is then wrong by at most the bound, and the response by at most
  $\sum_k |A_k| e^{-\Gamma_k t}$ times the frequency error times $t$, which the tests
  check against optmat's direct evaluation along the path.

[SEP 10](../../dev/seps/sep-0010-low-rank-raman.md) removes this restriction.

## Validated ranges, and vacuum

Each response holds between two densities, found for the strongest line of each band:
below, the motion of the molecules sets the widths (the Dicke regime); above, the lines
overlap and their mixing is not modelled. Among the stated limits:

- N₂ and dry air from 0.054 to 2.09 amagat, that is up to about 2.3 bar at 293.15 K, and
  between 293 and 300 K;
- O₂ from 0.034 to 4.2 amagat;
- H₂ above 0.94 amagat and D₂ above 0.61 amagat, the Dicke cutoffs.

The run declares the non-zero densities its path takes (the polarisation is exactly
zero in vacuum) to the response's range check, every density down to zero included
when the path falls to vacuum, and its temperature, once each, and
warns once per run ({class}`~solprop.exceptions.MaterialRangeWarning`). Nothing is
suppressed. **A capillary of several bar of N₂ or air therefore runs outside the
independent-line model's validated range, and so does H₂ at 1 bar and 293.15 K, which is
0.92 amagat**; a gradient that falls into the Dicke regime warns as well. The
[material manifest](../results/provenance.md#the-material-manifest) records the states outside.

The decay paths are evaluated at the actual density, vacuum included. optmat's group
law has a finite limit at zero density, $\pi(c + \Delta\nu_D)$ with a Dicke term and
$\pi c$ without one: zero for N₂'s and O₂'s lines, whose widths are pure broadening.
No floor is applied; the cap at {data}`~solprop.translate.raman.MAX_DECAY_EXPONENT`
over the fine sample spacing stays as a numerical guard.

## Cost, and what a run records

The kernel rebuilds $\sum_g e^{-\gamma_g t} u_g$ and transforms it whenever a decay rate
changes: once per run at constant density, and at every stage on a density gradient.
The rebuild scales with the number of rows: on this page's reference measurement, at
$2^{14}$ fine samples, 6 ms for N₂'s 72 rows and 10 ms for dry air's 126, against
0.3 ms for solprop's own two-term N₂ model and 0.25 ms for one convolution
([benchmarks](../../dev/benchmarks.md#raman-kernel-rebuild)). SEP 10's low-rank
representation removes this cost.

The run's manifest records the response's identity, the reference temperature
(`raman_reference_temperature`) and, for each response, its labels, damping groups,
terms and shift policy (`raman_terms`) among the hashed policies; and beside them its
composition, carrier, the densities declared to its range check, the composition's
spread, and a held shift with its phase bound over the run's window (`raman`).

## Sources and qualification of the molecular data

A built-in gas's molecular data are optmat's: each response states its sources, its
validated temperature and density ranges and its status, and optmat's documentation
records its comparisons with measured gains, revivals and linewidths. solprop keeps no
molecular constants of its own.

A supplied species' constants are the caller's, and the models they are built into
leave out what follows; read it before trusting a delayed-probe or picosecond-scale
prediction from them.

- **One linewidth per ladder.** A rotational ladder takes one width for every $J$, and
  its temperature dependence is not modelled; the $J$ dependence is real.
- **One oscillator per vibration.** A vibrational band is one damped oscillator: a
  Q branch's $J$-dependent frequencies, line mixing, hot bands and anisotropic
  vibration are absent, so the ensemble's loss of phase coherence without collisions
  is not represented, and a collision time is not a coherence time of the band.
- **Spin and nuclear state.** Electronic-spin fine structure (O₂) is not represented,
  and the rotational populations assume nuclear-spin equilibrium: cooled normal
  hydrogen without a conversion catalyst need not reach it.
- **Collision partners.** A mixture's widths use the total density, assuming like
  broadening by every partner ([several responding species](raman.md#several-responding-species));
  a pure-gas width is not a binary or whole-air coefficient.
- **Wavelength.** A polarisability derivative or anisotropy measured at one
  wavelength is used at the pump's; the transfer is an approximation.
- **Coordinates.** A polarisability derivative and an effective mass belong to one
  displacement coordinate: $(\mathrm{d}\alpha/\mathrm{d}Q)^2/\mu$ is what enters, so
  changing one without the other changes the strength.
