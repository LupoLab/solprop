# Gas mixtures

A **mixture** ({class}`~solprop.physdata.mixtures.Mixture`) is an ordered set of labelled
populations. Each population ({class}`~solprop.physdata.mixtures.Component`) supplies a
species and a number-density path along z in m⁻³; labels distinguish populations,
including two populations of the same species. One mixture object supplies both the
linear operator (the summed susceptibility) and the density weights of every nonlinear
response, so the dispersion and the nonlinearity of a fill cannot disagree about its
composition.

This page is for anyone filling a guide or a free-space medium with more than one gas,
with air, or with a gas whose optical data they supply themselves. It covers the
species and their data, the two ways of building a mixture (a shared fill and
independent profiles), how the linear operator and each nonlinear response combine the
components, the per-species statistics and energy accounting, how to build and update a
run from a mixture, and what the output file records. The single-gas data behind
the species are on the [materials page](materials.md); pressure and temperature paths are
on the [profiles page](../waveguides/profiles.md).

## Species and optical tables

`solprop.physdata.species("Ar")` builds a built-in gas as a `Species`: its
susceptibility, real-gas equation of state, transport properties and Kerr model are
optmat's ([materials from optmat](material-selection.md#materials-from-optmat)), and its
`IonisationData` is optmat's first ionisation energy. Its rate and Raman response are
optmat's too, resolved for the run ([SEP 3](../../dev/seps/archive/sep-0003-optmat-materials.md));
a built-in species carries no Raman constants of its own. A custom species carries
its own data, Raman constants included
({class}`~solprop.physdata.raman_types.MolecularRaman`). Missing response data is
explicit; a response requiring unavailable data reports the species name.

### Air

There are two different things called air, and only one of them is nonlinear.

- **`Gas.Air`, the built-in entry**, is a *dispersion-only* species: optmat's
  refractive index of the composition `"dry air (N2, O2, Ar)"` as a whole
  ([materials](materials.md)). It has no Kerr coefficient, no ionisation data and no
  Raman data. Used as a species in a fill built by hand, such as
  `Mixture.fill({"Air": 1.0}, 1.0)`, it gives the right linear index but carries no
  Raman response under the default selection, refuses a Kerr response
  (`Air (Air) has no Kerr data`) and refuses plasma, and an explicit Raman selection of
  its label raises. On the keyword interfaces `kerr=False, raman=True` is refused for
  every gas (`raman=True needs kerr=True`), and with `kerr=False` and `raman` unset the
  fill runs with no nonlinear response at all and no warning, so a hand-built `Air` fill
  is a linear medium; use `Mixture.air` for a nonlinear one.
- **`Mixture.air(pressure, ...)`** builds separate nitrogen, oxygen and argon
  populations (optmat's CIPM-2007 composition `"dry air (N2, O2, Ar)"`, renormalised
  over the three), each with its own dispersion, Kerr coefficient,
  ionisation channel and — for N₂ and O₂ — Raman response. Carbon dioxide and the other
  trace constituents are omitted; `fractions=` replaces the composition.
  `Mixture.air(..., argon=False)` takes optmat's `"dry air (N2, O2)"`, nitrogen and
  oxygen renormalised to each other, argon folded away with the trace gases.

**`gas="air"` on the drivers means the two-component mixture.** `prop_capillary`,
`prop_waveguide` and their `Simulation` constructors take `gas="air"` (any case) or
`gas=Gas.Air` as `Mixture.air(pressure, temperature=..., argon=False)`, never as the
dispersion-only entry, so the run has N₂ and O₂ Kerr, plasma and Raman channels; with
`raman` unset both molecules' Raman responses are on
([Raman](raman.md#gas-fills-in-a-capillary-or-waveguide)). Omitting argon is a
deliberate approximation: argon is 0.93 % of dry air's molecules, so leaving it out
costs about 1 % of the dispersion and Kerr response, and its ionisation contribution must be assessed at the actual field strength.
Its 15.76 eV ionisation potential does not exclude ionisation in every air calculation. Pass
`gas=Mixture.air(...)` to keep it; argon then contributes dispersion and Kerr, and its
missing Raman response is logged at INFO.

These rounded major constituents follow the dry-air composition described by
[NOAA](https://www.noaa.gov/jetstream/atmosphere). The fractions define the model; they
are not a claim about every ambient atmosphere.

### Tabulated optical data

`TabulatedSusceptibility` accepts increasing angular frequencies in rad/s, real
refractivity per density `(n - 1)/rho` in m³, intensity absorption cross-sections in
m², a declared wavelength range in metres, and a citation. It forms

```{math}
\gamma(\omega)=2\frac{n(\omega)-1}{\rho}
    +i\frac{c\sigma(\omega)}{\omega}.
```

This is a conversion of dilute optical data. The medium subsequently forms the exact
square root of the summed susceptibility. A cubic spline represents the real part;
shape-preserving interpolation of the absorption cross-section prevents negative
absorption between non-negative samples. Frequency derivatives differentiate these
interpolants, including `1/omega` in the imaginary part. Evaluation outside the
declared support raises `MaterialError`. Arrays are copied and made read-only.

```{code-block} python
import numpy as np
from solprop.constants import C
from solprop.physdata import Species, TabulatedSusceptibility

omega = np.linspace(2e15, 4e15, 101)
optics = TabulatedSusceptibility(
    omega=omega,
    refractivity=np.full(omega.shape, 1e-30),  # m³; synthetic example
    absorption_cross_section=np.full(omega.shape, 2e-24),  # m²
    wavelength_range=(2 * np.pi * C / omega[-1], 2 * np.pi * C / omega[0]),
    citation="Synthetic constant optical data for this example",
)
absorber = Species("example absorber", optics)
```

Custom species default to `IdealGas()`. Their nonlinear data (third-order
hyperpolarisability, ionisation data, Raman model) must be supplied explicitly;
selecting a response does not invent missing constants.

## Shared fills and independent profiles

`Mixture.fill` takes species-to-fraction mappings summing to one and a total pressure
in bar. Pressure may be a scalar, a `ZProfile`, endpoint pressures, or a pair of
position and pressure sequences. Endpoint pairs require `length=` in metres. The
standard pressure-gradient floor applies unless `floor=` is supplied.
A zero fraction is an exact zero-density population and retains its label.

```{code-block} python
from solprop.physdata import Mixture

fill = Mixture.fill({"He": 0.8, "Ar": 0.2}, (3.0, 0.2), length=1.5)
air = Mixture.air(6.0, temperature=300.0)
changed = fill.with_fill({"He": 0.7, "Ar": 0.3}, pressure=2.0)
```

Each component uses its own equation of state at its partial pressure:
`rho_s(z) = eos_s(x_s * p(z), T)`. These real-gas partials under Dalton's rule
approximate a mixture EOS; cross-species virial terms are absent. No mixture EOS
is used in propagation. The size of this approximation is quantified
[below](#linear-propagation-and-absorption) at ten bar; that comparison is not an accuracy
guarantee near condensation or at higher pressure. `IdealGas` uses `rho = p/(k_B T)`, converting
bar at the interface.

`Mixture.fill`, `Mixture.partial_pressures` and `Mixture.air` take `selection=`, a
{class}`~solprop.physdata.selection.MaterialSelection`, and without one take
`MaterialSelection()`: each named species is resolved from optmat, its linear law,
equation of state and transport properties, with the same partial-pressure rule over
optmat's equations of state, and `Mixture.air` takes optmat's named composition.
Supplied species are kept as they are. See
[materials from optmat](material-selection.md#materials-from-optmat).

**A shared fill is one flow.** Given the core radius of the guide it flows through,
`Mixture.fill(..., radius=)` (and `Mixture.air`, `with_fill`) builds an endpoint pair or
`(z, p)` table as the steady flow of the whole mixture
({func}`solprop.gasflow.steady_fill`, [profiles](../waveguides/profiles.md#steady-flow-fills)): it
follows the bore, chokes a vacuum end (`vacuum_end="choked"`, the default with a
radius; `"floor"` keeps the declared ends and still follows the bore) and checks the flow regime, using the
mixture's viscosity from Wilke's rule and its sound speed from the mole-weighted molar
mass and the mixed ideal-gas heat capacities
({mod}`solprop.physdata.transport`). The species are assumed not to separate along the
fill; separation is not modelled. Without a radius the uniform law and the numerical floor
remain, since the choke needs the bore; the simulation constructors supply the guide's.
A `vacuum_end=` beside a constant pressure or a profile, which have no end to choke, is
refused by name, and so is a varying `radius=` beside a varying pressure profile built
without the bore (it would be used as given; build it with
`PressureGradient.from_stages(..., radius=)`). A choked end depends on the gas, so
`with_fill(fractions)` that keeps a fill with a choked end is refused and asks for the
pressure again, which builds the flow for the new composition.

**A temperature along z.** `temperature=` of `Mixture.fill`, `Mixture.air`,
`partial_pressures` and `of` takes a number or a `ZProfile` in K (a constant path is
stored as its number, bit for bit). Every density path then follows it,
$\rho_s(z) = \mathrm{eos}_s\big(x_s\,p(z), T(z)\big)$
({class}`~solprop.profiles.DensityPath`), and a gradient built from end pressures follows
the flow weight $\eta(T)\,T$ of the whole mixture, with or without a radius
([profiles](../waveguides/profiles.md#a-temperature-that-varies-along-the-fibre)). A
`PressureGradient` supplied beside a varying temperature — to `fill`, `air`,
`with_fill` or `partial_pressures` — must carry the weight of this gas and this
temperature path (`solprop.gasflow.steady_fill(..., temperature=)`); a missing or
foreign weight is refused by name. A declared pressure path that is not a steady-flow
fill is taken as given. `with_fill(fractions)` refuses to keep a weighted fill, whose
weight is the old composition's. A Mixture owns its temperature: a gas-filled mode or a
simulation constructor given a Mixture refuses a `temperature=` that is not the
mixture's own (omit it). A shared fill stays one flow, but one
total pressure does not fix the densities once the temperature varies:
`Mixture.at_parameters({"pressure": ...})` is refused by name, and a gas-filled mode
declares the populations as its parameters, `density:<label>`, instead of `pressure`.
Two species in a uniform bore are two parameters and keep the tensor expansion of the
linear operator on every backend; three species, or two in a taper (whose radius is a
third), exceed its two parameters (`MAX_PARAMETERS`), so the operator is tabulated
along z, which `numpy_ref` and `numba_cpu` evaluate and the native backend refuses by name, as for
any three independent profiles. A single gas keeps its one `density` parameter. The
rotational Raman populations take one reference temperature
({func}`~solprop.nonlinear.mixtures.raman_temperature`,
[Raman](raman-models.md#the-rotational-ladder)). `serialise()` records a temperature path as
it records a density: samples with breakpoints.

`Mixture.partial_pressures` accepts independent scalar pressures or pressure profiles.
They are not one flow: the steady-flow law and the choke apply only to a shared fill,
and a `vacuum_end=` beside independent populations is refused by the constructors.
`Mixture.of` accepts arbitrary density profiles with explicit labels. The density
matrix has shape `(n_species, n_z)` in mixture order. `total_density(z)` sums this
axis, and `gamma3_path()` composes the density-weighted third-order susceptibility.

```{code-block} python
from solprop.physdata import Component, Mixture, species
from solprop.profiles import Constant, Linear

argon = species("Ar")
populations = Mixture.of(
    Component(argon, Linear(1e25, 2e25, 1.0), "reservoir"),
    Component(argon, Constant(1e25), "background"),
    temperature=300.0,
)
updated = populations.with_densities({"reservoir": Linear(0.0, 1e25, 1.0)})
```

`with_densities` retains the species objects and labels. `serialise()` records species,
labels, temperature, fractions and profile samples with breakpoints; it does not claim
a sampled callable is reconstructible. `Scaled(path, factor)` and `Sum(paths)`
compose values and derivatives, preserving component order and breakpoint unions.
Density paths continue endpoint values outside their span. They reject negative
pressure at declared boundaries, floor negative interpolant overshoot between those
boundaries, and return exactly zero at vacuum. Arbitrary density profiles are
validated by component label.

## Linear propagation and absorption

The bulk susceptibility is summed before forming the refractive index:

```{math}
\chi(\omega,z)=\sum_s\rho_s(z)\gamma_s(\omega),\qquad
n(\omega,z)=\sqrt{1+\chi(\omega,z)}.
```

`MarcatiliMode(mixture=fill)` adds this susceptibility inside the capillary index.
`SpatialMedium(material=fill)` uses the same composition for bulk propagation.
Frequency derivatives sum species derivatives before differentiating the square root.
The frame uses the real group delay. Positive imaginary index contributes loss; in
the dilute limit gas intensity attenuation approaches `sum_s rho_s * sigma_s`.

Capillary `loss=` controls total loss, including gas absorption: `False` removes it,
a non-negative number scales it, and a callable replaces it. `loss=` does not touch the
real shift of the mode index that the cladding causes, which stays as the mode model
defines it.

The operator has three possible representations:

1. A shared fill declares `pressure` and the capillary `core_radius`, and the linear
   operator is a tensor Chebyshev expansion in those two parameters, as for a single
   gas. A single population keeps the one `density` parameter, and the same arithmetic
   as a single-gas run, including its density spline.
2. Independent populations declare `density:<label>`. The tensor expansion supports at
   most two varying parameters (`MAX_PARAMETERS`); beyond that the operator is
   tabulated along z. `numpy_ref` and `numba_cpu` evaluate that table on the host;
   The native backend refuses it, naming the component labels, and suggests `numba_cpu`. Cartesian
   media must be constant in every component, including mixtures whose total density
   is constant while their composition varies.
3. A total-degree Chebyshev basis, keeping `comb(N+d, d)` terms for `N` parameters and
   degree `d`, was measured as an alternative to the z table and not adopted: it meets
   the accuracy budgets, but on a short test fill it needs more memory and build time
   (184,800 bytes and 19.62 ms, against 20,640 bytes and 0.584 ms), though on a
   ten-bar, 1.5 m fill it is cheaper in both.

The index must be smooth across the sampled parameter box. A mode cutoff or a
loss clamp inside the box can invalidate a polynomial approximation, in which case the
translation falls back to the z table. The fallback is logged at DEBUG level. Every
CPU backend runs the table; a GPU build of the native backend refuses it, saying which
quantity took it.

At 300 K and ten bar the partial-density approximation differs from CoolProp 8.0.0's
HEOS mixture model by −0.1362% for the default air composition and +0.1599% for a
50/50 helium–argon fill. To second virial order, the relative difference is
`(B_mix - sum_s x_s² B_ss) p/(RT)`, where `B` is in m³/mol. Cross-partner data enter
`B_mix`; [CoolProp's mixture model](https://coolprop.org/fluid_properties/Mixtures.html)
requires binary interaction parameters beyond the pure-fluid data. These comparisons
quantify a model approximation and do not certify either model against experiment.
The second-virial estimate, made before the comparison, gives −0.1452 % and +0.1493 %;
the remainders, $0.9\times10^{-4}$ and $1.1\times10^{-4}$, are higher-order terms.


## Nonlinear components and plasma channels

`responses_for` binds declarations to labels before translation. The same mixture
object supplies the optical and nonlinear densities. For example, the following
selects optmat's rates for nitrogen and oxygen (the same default as a single-gas run)
and optmat's Raman response of the composition:

```{code-block} python
from solprop.nonlinear import responses_for
from solprop.physdata import Mixture

fill = Mixture.air(6.0, temperature=300.0)
responses = responses_for(
    fill, plasma={"N2": True, "O2": True}, raman=True, lambda0=800e-9
)
```

**Selecting Raman.** `raman` takes the same vocabulary here and on the drivers:

- `True` selects every available Raman term and logs, at INFO, the components without
  data (here argon);
- a sequence of labels, e.g. `("N2",)`, selects those components only;
- a mapping from label to `True`, `False` or a
  {class}`~solprop.physdata.raman_types.MolecularOptions` selects per component, with the
  rotational and vibrational terms and the population temperature chosen separately.

An explicit selection fails by label and term when the data are absent. optmat has
validated responses for H₂, D₂, N₂, O₂, CH₄ and SF₆ and a provisional one for nitrous
oxide; the noble gases and the dispersion-only `Gas.Air` have none. A custom species may supply a
rotational-only model and select it explicitly. The defaults differ by route:
`responses_for` has `raman=False`, while `prop_capillary` and `prop_waveguide` with
`raman` unset turn on every component with data (see
[Raman](raman.md#gas-fills-in-a-capillary-or-waveguide)).

Default plasma selection takes each built-in gas's optmat rate model
([rates from optmat](ionisation.md#rates-from-optmat)), and refuses a custom species,
whose rate is given by label: an optmat model from explicit parameters, or a rate of the
caller's own ([a rate of your own](ionisation.md#a-rate-of-your-own)). Borrowing an EOS
adapter supplies no rate. Citation text does not affect the numerical selection.

Pass `envelope=True` to `responses_for` for an envelope simulation. Explicit
`plasma=True` or a per-label rate mapping then builds `EnvelopePlasmaResponse`
channels and defaults third-harmonic generation off. Each rate must declare the
instantaneous isotropic convention; a pre-averaged rate is rejected. For example,
`responses_for(fill, envelope=True, plasma={"N2": True}, raman=False)`
opens only nitrogen's channel. `Simulation.with_mixture` retains this response
selection and reuses the immutable rate data while updating each population's
density and preionisation paths.

Kerr uses `epsilon_0 * sum(rho_s * gamma3_s)` with one cubic evaluation. Plasma
channels keep independent exposures and fractions. Their charge-weighted populations
and ionisation work are summed before the two shared current integrals. A single
channel uses the same field-grid arithmetic as a single-gas run, with its density
outside those integrals.
A zero-density component stays in the species metadata and contributes exactly zero.
On an envelope grid, channels instead add their Drude drives and directional
loss currents separately. Projection and temporal transforms precede the common
spectral factors. Their fractions and peak-rate statistics use the same averaged
rate as the source calculation.

Raman components share effective convolution kernels: one history for scalar
or isotropic-only vector responses, three for vector rank-two responses
([Raman](raman-vector.md#vector-convolution-and-resolution-guidance)).
Explicit tensor labels survive component selection, dephasing overrides and
composition updates. For a mixture of built-in gases the default is optmat's one response
for the whole composition, with its curated partner-resolved broadening
([mixtures and their collision partners](raman-data.md#mixtures-and-their-collision-partners)).
For a supplied or hand-built component, each component's partial density scales its
amplitude, while its own dephasing coefficients are evaluated at the **total** number
density. That assumes like broadening by all collision partners: nitrogen broadened
by oxygen does not establish the reverse partner's coefficients. `Component.dephasing` accepts a supplied law when
suitable data exist. Cache validity includes every changing amplitude and decay rate.
A single Raman component uses the same arithmetic as a single-gas run.

## Population statistics and energy accounting

Statistics preserve mixture order, including non-ionising species. The five entries
are `species_ionisation` (fraction at each transverse node),
`species_electron_density` (m⁻³ at the node with largest total electron density),
`species_peak_ionisation_rate` (s⁻¹), `species_density` (m⁻³), and
`species_ionisation_energy` (J/m). The last quantity is the final ionised inventory:

```{math}
U_s(z)=I_{p,s}\sum_j w_j\rho_s(z)f_s(t_{\rm end},z,j).
```

The area measure is the physical modal quadrature, the radial/Cartesian grid measure,
or the effective area for a mode-averaged run. A taper scales the modal area by its
radius ratio squared. Periodic statistics support the fractions and species electron
densities between save planes.

Preionisation is already present before the pulse. Electrons produced during the
pulse are `rho_s * (f_end - f_initial)`; their ionisation work subtracts the analogous
initial inventory from `U_s`. A final inventory is therefore not, by itself, energy
deposited by that pulse. Linear absorption is separate from this ionisation work.

## Public construction and updates

`prop_capillary`, `prop_waveguide` and their `Simulation.from_capillary` and
`Simulation.from_waveguide` constructors accept, as `gas`, a built-in gas, `"air"` (the
two-component mixture [above](#air)), a mapping of species names to mole fractions, or a
complete `Mixture`. A built-in gas or a mapping takes `pressure` as the total pressure
(one bar when omitted); a `Mixture` carries its own paths, and `pressure`,
`pressure_floor`, `vacuum_end`, `susceptibility` and a different `temperature` beside it
are refused.
Pressure endpoints require a length in mixture constructors; the capillary entry
points already supply their `length`. Explicit conflicting pressure, unknown labels,
and unavailable selected response models fail by name.

```{code-block} python
from solprop.driver import Simulation
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.physdata import Mixture

material = Mixture.air(6.0, temperature=300.0)
simulation = Simulation.from_capillary(
    15e-6,
    0.225,
    material,
    lambda0=800e-9,
    wavelength_limits=(300e-9, 2e-6),
    trange=200e-15,
    fwhm=30e-15,
    energy=2.2e-6,
    thg=False,
    plasma={"N2": OptmatRate("N2", 800e-9), "O2": True},
    raman=("N2",),
)
changed = simulation.with_mixture(material.with_fill(pressure=5.0))
```

`plasma` maps labels to switches or rates; omitted labels are off. `raman` takes the
vocabulary [above](#nonlinear-components-and-plasma-channels). Initial
ionised fractions can be global or per label, each a number or a path along `z` (the
fraction ionised before the pulse arrives at each position). Omitted labels
retain their component's declaration, including when the whole `preionisation`
argument is omitted. A path is validated in `[0, 1]`, printed by its own description
and recorded in the run metadata by its samples and breakpoints, like a density path;
the steps of a run land on its kinks
([nonlinear.md](field-plasma.md#a-pre-ionised-fraction-along-the-fibre)).

```{code-block} python
from dataclasses import replace

import numpy as np
from solprop.physdata import Mixture
from solprop.profiles import PiecewiseLinear

fill = Mixture.fill({"Ar": 0.5, "He": 0.5}, 1.0)
# Argon 5 % ionised beyond 0.2 m, nothing before 0.1 m, a ramp between.
ramp = PiecewiseLinear(np.array([0.0, 0.1, 0.2, 0.5]), np.array([0.0, 0.0, 0.05, 0.05]))
seeded = replace(
    fill,
    components=tuple(
        replace(c, preionisation=ramp) if c.label == "Ar" else c
        for c in fill.components
    ),
)
```

The interface does the same with `preionisation={"Ar": ramp}`.

`with_mixture` preserves ordered labels and species data, grid, geometry, quadrature,
response selections, initial field, noise and numerical settings. It shares frozen
rate tables, unchanged undamped Raman kernels, the transverse transform and the
initial record. Density, Kerr and Raman amplitude tables and linewidth tables are
rebuilt. Expansion coefficients are shared only for the same operator function and
paths contained in its original sampled ranges. A pressure expansion changes its
function when the fill fractions or EOS temperature change, so those changes rebuild
its coefficients. A change to a parameter the original simulation held fixed also forces a
rebuild. Rebuild
reasons appear in the translation log. This applies to radial expansions as well as
modal expansions; Cartesian media are always constant.

An update from a shared fill to independent densities preserves an existing pressure
statistic by summing each component's partial pressure obtained from its own EOS.
This uses the same declared partial-EOS approximation as fill construction, in bar.
It does not introduce a pressure parameter into an independent-profile operator.

## Saved material analysis

The output file records mixture labels, temperature, species model data, and path
constructors or samples with breakpoints under `configuration["medium"]`. Capillary
mode records carry the mixture rather than a gas and a pressure. Files of output schema
4 or older, which do not record mixtures, remain readable; the material accessors
refuse, by name, reconstruction data that those files did not store.

The processing functions preserve species and node axes, even when an axis has one
entry:

| Function | Result and definition |
|---|---|
| `processing.species_ionisation(run)` | Final fractions, `(n_z, n_species, n_nodes)`, including preionisation. |
| `processing.species_ionisation_energy(run)` | Final ionisation inventory, `(n_z, n_species)`, in J/m. |
| `processing.absorbed_spectral_energy(run, species=None)` | Ascending active-band omega and `(n_z, n_species, n_omega)` energy loss density in J/(m rad/s). A selected label retains a length-one species axis. |
| `processing.node_fluence(run)` | `(n_z, n_nodes)` fluence in J/m² from the saved physical analytic field. |

Material absorption is attributed through `MarcatiliMode.absorption_by_species` to
first order in susceptibility: `alpha_s = rho_s omega Im(gamma_s)/c`. The processing
function multiplies this by spectral energy density and sums the modal contributions;
spatial outputs use their spectral quadrature measure. Wall leakage and ionisation
work are separate. `loss=False` removes all attributed loss, numeric `loss` scales it,
and callable replacement of total loss has no justified species attribution. The
last case is recorded explicitly and refused by the absorption accessor.

Node fluence is `epsilon_0 c/2` times the fine-grid temporal sum of the physical
analytic flux, summed over polarisation. For a modal run that is
$\operatorname{Re}[E\cdot E_w^*]$, with $E$ synthesised from the saved true-power
amplitude divided by $\sqrt w$ and $E_w$ from it multiplied by $\sqrt w$ (the
Poynting flux of [power and intensity](../results/power-and-intensity.md)); for a radial or
Cartesian run, which has no modal weight, it is $|E|^2$, reconstructed by the inverse
Hankel or Cartesian transform. The stored synthesis matrix has its temporal padding
scale removed. No extra spectral or temporal window is applied. A mode-averaged run has
one effective node, with fluence equal to pulse energy divided by effective area.
Files of output schema 5 or older keep the vacuum-admittance functional $|E|^2$
throughout. A finite radial Hankel transform is only approximately unitary: its physical
and spectral quadrature energies converge with radial resolution, rather than being
bitwise equal on an arbitrary coarse grid.

The [air tutorial](../../auto_tutorials/capillaries/plot_04_air_mixture) shows separate nitrogen and
oxygen ionisation, the nitrogen Raman kernel and a three-point composition update.

A regression test uses a 15 µm capillary of length 0.225 m, air at 6 bar and 300 K,
and a 30 fs, 2.2 µJ, 800 nm pulse. Nitrogen and oxygen take an explicit rate by label;
an additional absorbing species has explicitly synthetic table data.
Saved planes are checked against independent exposure sums, physical transverse
weights, density-times-cross-section absorption, and temporal fluence sums. Three
fine-grid resolutions check source and spectrum convergence. These checks test the
numerics only: they say nothing about the accuracy of the nitrogen and oxygen index
fits, which are extrapolated into the deep ultraviolet. The air tutorial uses a finer
temporal grid than the test, to resolve weak sub-cycle ionisation at the entrance.
