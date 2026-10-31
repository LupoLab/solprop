# Selecting and inspecting material data

Start with the recommended material models, then change a model only when the measurement or spectral range calls for it. Record both the material data and the response approximation. This chapter shows how to select, inspect and supply data; [material laws](materials.md) explains their units, density scaling and validity.

## Materials from optmat

[optmat](https://github.com/LupoLab/optmat) is a curated library of optical material
data: every model states its sources, its qualified range, its reference state and a
status, and carries an identity whose SHA-256 digest changes whenever its numbers can.
solprop takes its gases' linear index, equation of state and transport properties, and
its solids' complex indices, from optmat; it keeps the propagation physics (linear
density scaling, the mixture rule, the flow problem, the mode equations). It also takes
each gas's electronic Kerr coefficient, at the run's carrier, fused silica's nonlinear
response — its total $n_2$, Raman
fraction and Raman shape — from optmat's silica bundle, each gas's ionisation rate
([rates from optmat](ionisation.md#rates-from-optmat)) and a gas fill's Raman response
([Raman responses from optmat](raman-data.md#raman-responses-from-optmat)). An omitted selection is `MaterialSelection()`, which takes optmat's recommended
models. The [material integration proposal](../../dev/seps/archive/sep-0003-optmat-materials.md)
records how the data are mapped into solprop.

## Selecting gases

A {class}`~solprop.physdata.selection.MaterialSelection` names the models. Its
`linear` field is `"recommended"` (optmat's recommended model of each gas), an optmat
model name (`"borzsonyi_2008/sellmeier_Ar"`, from `optmat.gas.models.registry`), or an
optmat model object. `solprop.physdata.species()` and the mixture constructors take
it, and without one take `MaterialSelection()`:

```{code-block} python
from solprop.physdata import Mixture, MaterialSelection, species

argon = species("Ar")  # fit/Ar, eos/Ar, transport/Ar, and its
# hyperpolarizability/Ar, evaluated at a run's carrier
fill = Mixture.fill({"He": 0.5, "Ar": 0.5}, 3.0)
air = Mixture.air(1.0)  # CIPM-2007 "dry air (N2, O2, Ar)"
other = species("Ar", selection=MaterialSelection(linear="borzsonyi_2008/sellmeier_Ar"))
```

The species then carries an
{class}`~solprop.physdata.sellmeier.OptmatSusceptibility` and an
{class}`~solprop.physdata.species.OptmatGas`. A supplied
{class}`~solprop.physdata.species.Species` in a mixture is taken as it is and never
matched to a built-in gas by its name, so a custom species keeps its own law beside
optmat-selected ones.

`electronic` and `carrier_wavelength` are resolved as the next three sections describe:
`"recommended"` and `"carrier-kerr"` (one policy) take the Kerr coefficient at the
carrier, and the dispersive policies `"gas-polynomial"` and `"symmetric-kerr"` take
optmat's whole hyperpolarisability model
([below](#the-dispersive-electronic-policies)). `raman` selects a gas response in a gas-filled guide or a silica bundle for the GNLSE;
the permitted objects differ between those contexts. `ionisation` names the run's rate models:
`plasma=True` takes optmat's rate for each species, its plasma loss pays
that rate's own energy per event (`energy_loss="peak-weighted"`), and a custom species
needs an explicit rate; see [rates from optmat](ionisation.md#rates-from-optmat).
`raman` names a gas fill's Raman response and `raman_shift` its density-shift
approximation ([Raman responses from optmat](raman-data.md#raman-responses-from-optmat)).

## The electronic Kerr coefficient at the carrier

optmat fits each gas's electronic second hyperpolarisability $\gamma_T(\omega)$, with
its dispersion, in the Taylor convention $p^{(3)} = \gamma_T E^3/6$. In solprop's
$P^{(3)} = \varepsilon_0\chi^{(3)}E^3$ the per-particle coefficient is therefore

$$
\gamma^{(3)}(\lambda_c) = \frac{\gamma_{T,\mathrm{Kerr}}(\lambda_c)}{6\varepsilon_0},
\qquad \chi^{(3)}(z) = \sum_s N_s(z)\,\gamma^{(3)}_s(\lambda_c),
$$

which optmat's public `gamma3("kerr", wavelength)` returns. A built-in species carries
the model, an {class}`~solprop.physdata.nonlinear_data.OptmatKerr`,
not a number: its `gamma3` is `None`, and
{meth}`~solprop.physdata.species.Species.kerr_coefficient` evaluates the model at a
carrier $\lambda_c$. A run's carrier is its **grid's reference wavelength**, which the
drivers build at `lambda0`; the selection's `carrier_wavelength` may be left unset, and
if it is given it must agree with that, or the run is refused — one carrier sets every
optmat species' coefficient. The Kerr fits hold from 400 nm to 20 µm and warn once per
run outside.

This is the `carrier-kerr` approximation (SEP 3, section 4.3): one frozen coefficient
per species and run, used for every process the cubic term drives — self- and
cross-phase modulation and, with `thg=True`, the third harmonic. It is not broadband
four-wave mixing or process-correct third-harmonic generation, and the carrier is not
tracked as the spectrum moves (the dispersive policies below are the alternative).
optmat's distinct third-harmonic coefficient measures the cost: the run's manifest records, for each optmat population, the Kerr coefficient
used, the third-harmonic one at the same fundamental and their ratio (1.18 for argon at
800 nm), or no ratio where optmat's third-harmonic evaluation leaves its fitted data
(N₂O at 800 nm). No Raman amplitude is multiplied by the electronic coefficient, and
no local-field factor is added.

The nonlinear index keeps the exact conversion with the linear index $n_0$,
$n_2 = 3\chi^{(3)}/(4\varepsilon_0 c n_0^2)$: optmat's own gas `n2` takes $n_0 = 1$
(1.6 % high for 30 bar of argon) and is never imported. `chi3(..., wavelength=)` and
`n2(..., wavelength=)` give the selected values (or take the carrier from
`selection=`'s `carrier_wavelength`); a
{class}`~solprop.physdata.mixtures.Mixture` evaluates its populations at a carrier with
`gamma3_path(wavelength)`, which {func}`solprop.tools.soliton_params` passes its
`wavelength`.

```{code-block} python
from solprop.physdata import MaterialSelection, chi3, n2, species

selection = MaterialSelection()
argon = species("Ar", selection=selection)
argon.kerr_coefficient(800e-9)  # 1.476e-51 m^5/V^2
chi3("Ar", pressure=3.0, selection=selection, wavelength=800e-9)
n2("Ar", pressure=3.0, wavelength=800e-9, selection=selection)
```

## The dispersive electronic policies

`MaterialSelection(electronic="gas-polynomial")` and `"symmetric-kerr"`, from
[SEP 6](../../dev/seps/archive/sep-0006-dispersive-nonlinearity.md), evaluate optmat's whole fit,
$\gamma_T=\gamma_0[1+AS_2+BS_2^2+B'S_4]$ over each quartet's four signed vacuum
wavenumbers, at the absolute frequencies of every quartet the cubic couples, instead of
one coefficient: the polynomial itself, or a symmetric factorisation of its Kerr
diagonal. Each built-in gas resolves its fit with optmat's support record of it, the
fitted frequency sums and the resonance levels, as a
{class}`~solprop.physdata.nonlinear_data.DispersiveKerrModel` in SI. From the record
solprop derives the gas's leg bound and band edge; every leg beyond the bound is held
there, the run's band is labelled supported, bounded or best effort, and a run is never
refused for the frequencies its field occupies. solprop writes none of those numbers
down: a changed record changes the run's identity. `carrier_wavelength` is not used by
the electronic response under these policies — they have no carrier — but still sets
the Raman pump dispersion and the ionisation rate.

The policy is the run's: under it every optmat gas of a capillary run, its mixture
route and its scans gets a
{class}`~solprop.nonlinear.responses.DispersiveKerrResponse`, one per population. A
species without optmat's model (a supplied one) is refused by name rather than given a
borrowed curve, a constant `KerrResponse` declared beside the policy is refused, and so
is the GNLSE route, whose coefficient is constant. On the general modal route a region
declaring its gas fill (`GasKerr`) takes the policy. The manifest's
`electronic` entries record each population's model and support digests, leg bound,
band edges and regions in place of the carrier coefficient. The responses run on
`numpy_ref`, `numba_cpu` and the native backend, on its CPU build and its qualified
device builds. The physics, the two evaluators, the regions and how to choose
between the policies are in
[a frequency-dependent electronic response](dispersive-kerr.md).

## Fused silica's nonlinear response

optmat's **silica bundle** ({class}`~solprop.physdata.glasses.SilicaBundle`) holds
fused silica's total long-pulse $n_2(\lambda)$ (the Kerr fit `kerr/SiO2`), a Raman shape
$h_R$ of unit integral and that shape's own Raman fraction $f_R$, so that the three
cannot be mismatched. The electronic part is $(1 - f_R)n_2$ and the delayed part
$f_R n_2$, convolved with $h_R$.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 30 12 36

   * - bundle
     - shape
     - :math:`f_R`
     - used for
   * - ``"default"``
     - Hollenbeck and Cantrell (2002), paired with the Kerr fit
     - 0.2188
     - the scalar GNLSE; copolarised only
   * - ``"tensor"``
     - Lin and Agrawal (2006), with Hellwarth's isotropic :math:`a` and anisotropic
       :math:`b`, :math:`a + b = h_R`
     - 0.245
     - a vector silica response
```

Both hold the same total $n_2$, so a scalar and a vector run agree for long pulses; the
vector split's electronic part is 3.3 % smaller and its Raman part 12 % larger.
`prop_gnlse` takes the bundle's $f_R$ for an omitted `raman_fraction` with
Raman on, its sampled $h_R$ as the response, and `n2="recommended"` as its $n_2$ at the
carrier ([GNLSE](../waveguides/gnlse.md#the-raman-fraction-is-not-silently-taken)). A material
region's tensor response takes the tensor bundle as
`SilicaTensorResponse(parameters=SilicaBundle.resolve(tensor=True))`: its isotropic
channel is $h_0 = a + b/3$ and its rank-two channel $h_2 = \tfrac23 b$, which the seam
weights $(1, 0)$ and $(\tfrac14, \tfrac34)$, so a copolarised probe sees $a + b$ and a
cross-polarised one $a$; {func}`~solprop.nonlinear.raman.bundle_response` builds the
same two terms as a {class}`~solprop.nonlinear.raman.RamanResponse`. A tensor response
of the copolarised-only default bundle is refused by name. A general modal guide's
`SilicaTensorResponse()` is this tensor bundle by default, and the bundle carries an
identity into the manifest and the guide's record. Lin and Agrawal's own time
constants and weights are given explicitly, as
`SilicaTensorResponse(parameters=LinAgrawalRaman(...))`
({class}`~solprop.physdata.raman_types.LinAgrawalRaman`), and are the caller's; the
two kernels agree to $4\times10^{-16}$.

optmat's silica `raman_coefficient` and `electronic_susceptibility` include its index
$n_0^2 \approx 2.1$: $K = 2n_0^2\varepsilon_0 c f_R n_2$ and
$\chi_e = \tfrac43 n_0^2\varepsilon_0 c(1 - f_R)n_2$. solprop's GNLSE calibration and
its vacuum-normalised amplitudes take $n_0 = 1$ in both
([conventions §4](../../dev/conventions.md)), so only $n_2$, $f_R$ and the normalised
$h_R$ are imported; a test records the $n_0^2$ ratio.

## Selecting a run's materials

The drivers take the same selection as `materials=`: `prop_capillary`,
`prop_waveguide`, `prop_gnlse`, `prop_free`, `prop_radial` and `prop_planar`, the
`Simulation.from_*` constructors behind them, and `solprop.sol`'s re-exports; an
omitted one is `MaterialSelection()`. What the driver resolves itself comes from
optmat; what the caller built is taken as it is.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - driver
     - resolved under the selection
   * - ``prop_capillary``
     - the gas by name, a fraction mapping's names, ``"air"`` and ``Gas.Air`` (the
       composition ``"dry air (N2, O2)"``), and the default cladding of a numbered mode
       set, which becomes optmat's recommended silica ``fit/SiO2``
   * - ``prop_waveguide``
     - the gas, the fraction mapping and ``"air"``; the supplied modes keep their
       claddings. A provider's material regions are its own; a region's
       ``SilicaTensorResponse()`` is optmat's tensor silica bundle by default
   * - ``prop_free``, ``prop_radial``
     - a :py:class:`~solprop.spatial.SpatialMedium` naming a built-in gas becomes the
       one-population mixture of that gas under the selection, at the same pressure
       and temperature; a supplied susceptibility is kept beside optmat's equation of
       state. The declared responses are the caller's: write
       ``KerrResponse(gamma3("Ar", wavelength=lambda0))`` for optmat's coefficient at
       the carrier. An optmat rate's energy per event is bound to the input's peak
       field, as on the capillary route
   * - ``prop_planar``
     - the slab modes' fill of a built-in gas with its default susceptibility, which
       takes the selection's single species at the slab's own pressure and
       temperature (no mixture is needed): its law, and its equation of state for
       the density. The manifest records the species and its dilute-scaling check,
       as for a mixture's population. A slab declared without a cladding takes
       optmat's recommended silica. A supplied susceptibility, cladding or response
       is the caller's. An optmat rate's energy per event is bound to the input's
       peak field, as on the capillary route.
       Gas mixtures in a planar guide remain unsupported, with or without a
       selection
   * - ``prop_gnlse``
     - fused silica's response from optmat's silica bundle: the Raman fraction an
       omitted ``raman_fraction`` takes with Raman on, the Raman shape, and
       ``n2="recommended"``; the fibre's dispersion and a given ``gamma`` or ``n2`` are
       the caller's
```

With Raman on, a capillary or waveguide fill resolved under the selection takes
optmat's gas Raman response: one response for the fill's whole composition, built only
from the collision partners optmat curates, and refused by name for any other mixture
([Raman responses from optmat](raman-data.md#raman-responses-from-optmat)).

A supplied mixture, mode set, cladding, glass or susceptibility is never rebuilt or
matched to a built-in by its name.

```{code-block} python
from solprop.driver import Simulation

simulation = Simulation.from_capillary(
    50e-6,
    4e-4,
    "Ar",
    1.0,
    lambda0=800e-9,
    wavelength_limits=(300e-9, 3e-6),
    trange=100e-15,
    fwhm=30e-15,
    energy=1e-6,
    plasma=False,
    n_r=8,
    n_save=2,
)
manifest = simulation.material_manifest()
sorted(record.name for record in manifest.records)
# ['eos/Ar', 'fit/Ar', 'fit/SiO2', 'hyperpolarizability/Ar', 'transport/Ar']
```

Each run is **one evaluation context**: its construction, its translation, its record
and its run share one log of what has been warned about, so a range, status or
dilute-scaling warning is raised once per run, however many modes or stages evaluate
the material. An adapter also evaluates each frequency array once: a translation
reads a material at every sample of its tables along z, and a repeated array takes the
stored index or polarisability, bitwise the same, with the same warnings and records.
The run writes its **material manifest** into its output: every optmat model it resolved, its species, compositions and solids with their
axes, the policies in force (the `carrier-kerr` law and its carrier, or a dispersive
law with its regions; a GNLSE run's held-back Raman fraction and where it came from),
the dilute-scaling error over its band, the bands outside a model's qualified
intervals, each population's carrier Kerr coefficient with its third-harmonic
discrepancy or its dispersive model's record, and the citations of its other
material objects
([output](../results/provenance.md#the-material-manifest)). A resume, or the reuse of a completed scan
point, is refused when any of its identities, policies or the optmat version differs,
naming the entry that changed. A scan resolves the selection at each point, from the
point's own objects, so no point depends on what was resolved before it.

## Units, reference state and operating state

optmat works in SI throughout: pressures in pascals, vacuum wavelengths in metres. At
solprop's interface pressures stay in bar and are converted by exactly $10^5$; an
angular frequency becomes $\lambda = 2\pi c/\omega$, and $\omega = 0$ is optmat's
static limit $\lambda = \infty$. Derivatives are taken at fixed state, with
$\mathrm{d}/\mathrm{d}\omega = -(\lambda/\omega)\,\mathrm{d}/\mathrm{d}\lambda$
inside optmat, which returns them in $\omega$ directly.

A source's **reference state** is where its data were measured or reduced to; it is
recorded with the model and is not the simulation's **operating state**. optmat's gas
models give the per-particle polarisability $\alpha(\omega)$, which is a dilute,
zero-density quantity: there is no reference density to divide out, and
`reference_state` records zero pressure and the temperature the source refers it to.
The operating state is the fill's own pressure and temperature, 293.15 K unless given.

## The gas index at linear density scaling

The susceptibility is solprop's existing linear law with optmat's refractive
polarisability (which already includes the magnetic term),

$$\chi(\omega; z) = \sum_s \frac{N_s(z)\,\alpha_s(\omega)}{\varepsilon_0},\qquad
n - 1 = \frac{\chi}{\sqrt{1+\chi}+1},$$

with optmat's exact $\partial_\omega\alpha$. This is the dilute limit of optmat's
own constitutive law, Lorentz–Lorenz with a refractivity virial, and differs from it
by a relative amount of order $N\alpha/(3\varepsilon_0)$. At 800 nm and 293.15 K:

| gas | 1 bar | 10 bar | 30 bar |
|---|---:|---:|---:|
| He | −0.002 % | −0.016 % | −0.047 % |
| Ar | −0.019 % | −0.19 % | −0.57 % |
| Xe | −0.053 % | −0.55 % | −1.9 % |
| SF₆ | −0.055 % | −0.62 % | condensed |

A fill with an optmat-selected gas therefore measures this error at each gas's
largest density along the fill, over its band, when its spectrum is sampled
({meth}`~solprop.physdata.mixtures.Mixture.dilute_scaling_checks`), and records it
in the run's manifest whatever its size. Above
{data}`~solprop.physdata.mixtures.DILUTE_SCALING_THRESHOLD`, 1 % relative in
$n-1$, it raises one {class}`~solprop.exceptions.DiluteScalingWarning` per gas, with
the value; the run keeps the dilute law. At 800 nm and 293.15 K argon and nitrogen pass
the bound near 52 and 53 bar, krypton near 31 bar, xenon near 17 bar and SF₆ near 15 bar. A finite-density law is
[SEP 11](../../dev/seps/sep-0011-finite-density-gas-index.md)'s.

`Gas.Air` is optmat's named composition `"dry air (N2, O2, Ar)"`: the mole-weighted
sum of its constituents' polarisabilities, with its own identity and derivative.
`OptmatSusceptibility.resolve(Gas.Air, "dry air (N2, O2)")` is the argon-free
composition. `Mixture.air` takes its fractions from the same named compositions,
renormalised over the constituents they keep.

## Density and transport

{class}`~solprop.physdata.species.OptmatGas` is optmat's real-gas law
$N = p/(Zk_\mathrm{B}T)$ and its exact inverse, on the gas's phase domain: below
saturation, or above the critical temperature. A **condensed state is refused**, with
the gas, the state and its saturation pressure named: SF₆ at 30 bar and 293.15 K is
refused because SF₆ condenses at 21.0 bar there. Zero pressure is zero density, and back, without calling optmat.
`Gas.Air` takes the density of CoolProp's pseudo-pure dry air, which includes the
unlike-pair interactions, and that fluid's molar mass, 0.0289655 kg/mol, for every
mass density, sound speed and mean free path; the constituents' mean, 2 × 10⁻⁴
lower, enters only the viscosity's Wilke rule. A mixture keeps the partial-pressure rule: each population's
density is its pure gas's at its partial pressure. The two differ for air: the
partial-pressure populations of `Mixture.air` are $1.9\times10^{-4}$ less dense than
the pseudo-pure air at 1 bar, $1.8\times10^{-3}$ at 10 bar, while their
polarisabilities are identical.

For a fill of optmat-selected species, {mod}`solprop.gasflow` takes optmat's
transport properties (`optmat.gas.transport`): the zero-density viscosity, the
ideal-gas heat capacity and the molar mass, each over optmat's own temperature range,
the real-gas sound speed of the equation of state, and optmat's Wilke rule for a
mixture (air resolved into its constituents). The mean free path, the flow law and the
regime checks are solprop's. A custom species without an
{class}`~solprop.physdata.species.OptmatGas` equation of state has no transport
properties, and a flow fill containing one is refused.

## Solids

The solid factories take `model=`: `None` (the default) or `"recommended"` takes
optmat's recommended model; an optmat model name or object takes that one. Each
returns an {class}`~solprop.physdata.glasses.OptmatGlass`, usable wherever a glass
is. Two adapters of one model compare equal, and hash alike, when they compute the same
index — the same model digest under the same transparent approximation and bound —
so two default walls or cores built separately are one material.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 30 26 18 26

   * - call
     - optmat model
     - range
     - notes
   * - ``fused_silica()``
     - ``fit/SiO2``
     - 24.8 nm–125 µm
     - absorbing, causal; 20 °C bulk grade
   * - ``calcium_fluoride()``
     - ``daimon_2002/sellmeier_20C``
     - 138 nm–2.33 µm
     - measured in nitrogen, made absolute
   * - ``magnesium_fluoride(axis=...)``
     - ``dodge_1984/sellmeier_o``, ``_e``
     - 0.20–7.04 µm
     - axis required
   * - ``barium_fluoride()``
     - ``malitson_1964/sellmeier``
     - 0.27–10.3 µm
     - room air, made absolute
   * - ``sapphire(axis=...)``
     - ``weber_2003/sellmeier_o``, ``_e``
     - 0.22–5.0 µm
     - axis required
   * - ``bk7()``
     - ``schott_2023/sellmeier``
     - 0.31–2.33 µm
     - lossless
   * - ``silver()``
     - ``table/Ag``
     - 4.1 nm–12.4 µm
     - absorbing table
   * - ``aluminium()``
     - ``table/Al``
     - 0.79 nm–200 µm
     - absorbing table
```

`glass(name, axis=...)` reaches the same models by name; so do the offline optics
and the compressors, which name their glasses.

**Named models.** `model=` selects any other optmat model of the material, among them
the published Sellmeier formulas, made absolute (see below):

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - factory
     - optmat model
   * - ``fused_silica``
     - ``malitson_1965/sellmeier``; ``franta_2016/udm_lithosil_q2``
   * - ``calcium_fluoride``
     - ``daimon_2002/sellmeier_20C``; ``malitson_1963/sellmeier``
   * - ``magnesium_fluoride``
     - ``dodge_1984/sellmeier_o``
   * - ``bk7``
     - ``schott_2023/sellmeier``
   * - ``sapphire``
     - ``weber_2003/sellmeier_o`` (Malitson and Dodge's ordinary-ray formula, as Weber prints it)
   * - ``barium_fluoride``
     - ``malitson_1964/sellmeier``
```

- **Absolute index.** Every optmat index is relative to vacuum. The published
  Sellmeier formulas of CaF₂, MgF₂, BaF₂, sapphire and N-BK7 were measured relative to
  air (or nitrogen), and optmat multiplies by the reference gas's index; taking such a
  formula as an absolute index would put it 3.7–5.6 × 10⁻⁴ low.
- **Sign.** optmat's index is $n + i\kappa$ with $\kappa \ge 0$ for loss, solprop's
  convention, and is never conjugated.
- **Birefringence.** MgF₂ and sapphire are uniaxial, and an optmat model is per ray: a
  window or wall names its ray, `axis="o"` or `axis="e"`, and the recommended model
  without an axis is refused by name — `magnesium_fluoride()`, `sapphire()` and
  `glass("MgF2")` included, a deliberate breaking change. A named model is per ray
  too, and refuses the other `axis`.
- **Metals** are tables with their interpolant's exact derivative. Outside their range
  they hold their end values, with a warning.
- **Lossless routes.** optmat's silica carries a small $\kappa$ even in its
  transparent window, and a route that solves only a lossless problem cannot hold it:
  the step-index modes ({class}`~solprop.modes.stepindex.StepIndexMode`, core and
  cladding), a general modal guide's material laws
  ({class}`~solprop.modes.material.IndexMaterial`, which the step-index and
  graded-index providers solve with) and the step-index vector fields. Such a route
  takes an optmat index through its **transparent approximation** automatically and
  never refuses it for absorption
  ({func}`~solprop.physdata.glasses.lossless_index`): the real part of the index and
  of its derivative, a *recorded* approximation within a declared domain. Every
  lossless route of one adapter takes one transparent counterpart, so the modes of one
  guide still share their material. Dropping an absorption coefficient
  $4\pi\kappa/\lambda$ above
  {data}`~solprop.physdata.glasses.TRANSPARENT_ABSORPTION_BOUND` (100 m⁻¹, which
  silica's window from 200 nm to 2.5 µm passes and its 170 nm edge and 2.73 µm band
  fail) over the band the run evaluates warns once per run
  ({class}`~solprop.exceptions.MaterialRangeWarning`, naming the material, the largest
  dropped absorption and its wavelength). The run's log records what the
  counterpart dropped in that run, which the run's manifest carries with the bound
  (`transparency`); the counterpart, shared by the runs of one source, also keeps
  what it has dropped over its life.
  `material.as_transparent(absorption_bound=...)` declares the approximation with
  another bound. A complex index of solprop's own keeps each route's existing
  handling: the step-index modes take its real part as documented, while the general
  providers and the vector fields refuse it. A hollow guide's wall, and bulk
  propagation, take the complex index as it is.

## Qualified intervals, statuses and warnings

A model holds on its **qualified intervals** of vacuum wavelength. The atoms, H₂, D₂,
N₂ and O₂ have one, which reaches the static limit; the infrared-active molecules have
windows between their absorption bands (CH₄ five, N₂O four, SF₆ one up to 5 µm). A band
outside them is still evaluated, and raises one
{class}`~solprop.exceptions.MaterialRangeWarning` per material and evaluation context
(a fill's spectral sampling, say). It names the parts of the band outside — below the
shortest qualified wavelength, in a gap between two intervals with the gap's reason
(absorption, width, representation), or beyond the longest — and quotes optmat's own
warning. The same holds for the solids, whose ranges are one interval each.

Every model has a **status**: `validated` (a default with a stated uncertainty),
`provisional` or `untested` (a stopgap, or nothing to test it against), `comparison`
(a curated alternative), or `user` (built from a caller's parameters). A `provisional`
or `untested` model warns once when used,
{class}`~solprop.exceptions.MaterialStatusWarning`, with optmat's reason; among the
defaults that is CH₄'s viscosity in a gas-flow fill and SF₆'s ionisation rate, and
N₂O's Raman response where `raman="provisional"` admits it. `comparison` and `user` models were
chosen by the caller and are recorded without a warning.

optmat's errors become {class}`~solprop.exceptions.MaterialError`, naming the
material, the quantity, the state or band asked for, and the model.

## Records

Every optmat-backed object exposes `material_records`: a tuple of
{class}`~solprop.physdata.selection.MaterialRecord`, each the model's name, quantity,
material, digest, status, optmat version and any overrides. A
{class}`~solprop.physdata.species.Species`, a
{class}`~solprop.physdata.mixtures.Mixture` (each model once), an
{class}`~solprop.physdata.sellmeier.OptmatSusceptibility`, an
{class}`~solprop.physdata.species.OptmatGas`, an
{class}`~solprop.physdata.nonlinear_data.OptmatKerr`, an
{class}`~solprop.physdata.glasses.OptmatGlass`, a
{class}`~solprop.physdata.glasses.SilicaBundle` (the bundle, its Kerr fit and its
shape) and a {class}`~solprop.nonlinear.raman.SampledShape` all carry them, and a run
records them in its material manifest. A mixture built by `Mixture.air(selection=...)` also records
the name of the optmat composition its fractions came from (`composition`).

## Quick experiments

A modified material needs no change to optmat. optmat's own overrides keep the
model's identity honest: the result has status `user`, its changes in `modified` and
the curated model's digest in `base`.

```{code-block} python
from optmat.gas.models.registry import default_model
from optmat.solid.models.formulas import FormulaModel
from optmat.solid.models.registry import default_model as solid_model

from solprop.physdata import MaterialSelection, OptmatGlass, calcium_fluoride, species

# a curated gas model with a narrower qualified range
narrow = default_model("Ar").with_parameters(intervals=((120e-9, float("inf")),))
argon = species("Ar", selection=MaterialSelection(linear=narrow))

# a curated solid with modified Sellmeier terms
curated = solid_model("CaF2")
terms = dict(curated.terms) | {"B": tuple(1.001 * b for b in curated.terms["B"])}
window = calcium_fluoride(model=curated.with_parameters(terms=terms))

# a caller's own Sellmeier formula, as an absolute index
mine = OptmatGlass.from_model(
    FormulaModel.sellmeier(
        "SiO2",
        {"B": (0.6961663, 0.4079426, 0.8974794), "C": (0.004679, 0.013512, 97.934)},
        (210e-9, 3.71e-6),
        citation="Malitson (1965)",
    )
)
```

A caller's table of measured $n$ and $k$ is made the same way, with optmat's
`TabulatedModel.from_arrays` passed to `OptmatGlass.from_model`; that is the one
constructor of a table from arrays, and solprop adds none to its
{class}`~solprop.physdata.glasses.TabulatedGlass`. solprop's own supplied types
— {class}`~solprop.physdata.glasses.SellmeierGlass`,
{class}`~solprop.physdata.glasses.TabulatedGlass`, a custom species with a
{class}`~solprop.physdata.species.TabulatedSusceptibility` — keep working beside them,
and carry their own citations.

## Limitations

- The gas law is linear density scaling, with the error above declared and measured,
  not removed.
- solprop requires optmat 0.3.0 or later: the noble gases' recommended ionisation rates
  take the frequency-dependent Coulomb factor from 0.3.0, which run records and their
  digests assume; a gas's Raman status is read without
  building its response, which 0.2.1 could not do; D₂'s viscosity is a smooth
  correlation from 0.2.2, so a D₂ temperature path over its whole range has a
  viscosity series; `Gas.Air` takes its molar mass from the equation of state, which
  0.2.0 did not expose; and 0.1.0's density inverse (a
  pressure from a density) failed for Ar, N₂ and CH₄ below about 255 K, 284 K and 256 K. A
  temperature at an equation's lowest temperature, its triple point, is refused.
- One grade and temperature per solid; N-BK7 is lossless; no OH, dopant or
  temperature absorption model.
- Gas mixtures in a planar guide remain unsupported: a slab's fill is one species.
- By default the Kerr coefficient is one number per species and run, at the carrier
  (`carrier-kerr`); the third harmonic is driven by the same number. The dispersive
  policies are opt-in and gases only; they run on `numpy_ref`, `numba_cpu` and the native
  backend ([backends](dispersive-kerr.md#backends));
  beyond a gas's band edge they are best effort, with no resonant response
  ([limitations](dispersive-kerr.md#limitations)).
- A free-space run's declared {class}`~solprop.nonlinear.responses.KerrResponse` is the
  caller's, coefficient included; `gamma3(gas, wavelength=lambda0)` gives optmat's.
- The design tools take the same `materials=` ([design tools](../workflows/design-tools.md)), so a
  design and a run under one selection agree.
- Objects that describe a gas by name outside a driver — a mode's own `gas=` fill, a
  {class}`~solprop.spatial.SpatialMedium` evaluated on its own, and the accessors
  `gamma`, `gamma3`, `chi3`, `n2`, `density` and `refractive_index` — take optmat's
  recommended model, as `MaterialSelection()` does.

This page documents the functionality proposed and implemented by
[SEP 3](../../dev/seps/archive/sep-0003-optmat-materials.md); its dispersive electronic
policies are [SEP 6](../../dev/seps/archive/sep-0006-dispersive-nonlinearity.md)'s.
