# Materials: gases, metals and glasses

Material data determine the refractive index, nonlinear response and losses of a
calculation. Check the wavelength range, pressure, temperature and physical model
against the observables you want to predict. A spectrum can extend into a region
where the propagation is numerically resolved but the material data are uncertain.

Start with the catalogue below, then use [material selection](material-selection.md)
to inspect or change a model. The [material-choices tutorial](../../auto_tutorials/basics/plot_06_material_choices.rst)
shows how to compare choices and read their records. [Gas mixtures](mixtures.md)
combines populations; [solid guides](../waveguides/solid-fibres.md) use bulk glass
data together with a mode model. General second-order ($\chi^{(2)}$) propagation
and a general anisotropic crystal response are not modelled.

solprop keeps no material database of its own. Every built-in gas, solid and response
comes from the curated package optmat
([SEP 3](../../dev/seps/archive/sep-0003-optmat-materials.md)): an omitted selection means
`MaterialSelection()`, optmat's recommended model of every family, and
[materials from optmat](material-selection.md#materials-from-optmat) says what that resolves to. What solprop
keeps is the propagation physics (linear density scaling, the mixture rule, the flow
problem, the mode equations) and the types a caller's own data take.

The built-in gases are thirteen: the noble gases He, Ne, Ar, Kr and Xe; the molecules
N₂, H₂, D₂, O₂, CH₄, SF₆ and N₂O; and air. For ordered compositions and custom absorbing
species, see [gas mixtures](mixtures.md). The `Gas.Air` entry is a *dispersion-only*
species, optmat's composition `"dry air (N2, O2, Ar)"`: it has a refractive index but no
Kerr, ionisation or Raman data. `Mixture.air()` resolves nitrogen, oxygen and argon
separately, and `gas="air"` on the drivers means the nitrogen–oxygen mixture, not this
entry ([mixtures](mixtures.md#air)).

## Who owns what, and how to choose

**optmat owns the data**: every number that describes a material, the model it comes
from, its sources, its qualified range, its reference state, its status and its stated
uncertainty. **solprop owns the propagation physics** built on it: the linear density
scaling of a gas's index and its measured error, the mixture rule, the flow problem,
the mode equations, the electronic policies — `carrier-kerr`, which freezes the Kerr
coefficient at the carrier, and the dispersive policies with their bound on the legs
beyond a gas's data — the rate table and the hold above a rate's maximum field, and the mapping of a
Raman response onto the kernel's terms. A run records both halves in its
[material manifest](material-selection.md#selecting-a-runs-materials).

Choosing is done in one of three ways, from the most to the least common:

1. **Nothing.** An omitted `materials=` is `MaterialSelection()`: optmat's recommended
   model of every family, for every built-in gas and solid a run resolves.
2. **A selection** ({class}`~solprop.physdata.selection.MaterialSelection`), passed as
   `materials=` to every driver and design tool: `linear=` names another of optmat's
   index models, `ionisation=` a literature rate form or a rate model object, `raman=`
   a Raman response object or `"provisional"`, and `carrier_wavelength=` the carrier
   ([selecting gases](material-selection.md#selecting-gases)). A solid takes `model=` on its factory
   ([solids](material-selection.md#solids)).
3. **An object of your own**: an optmat model modified with optmat's own
   `with_parameters` or built by `from_parameters`, which is recorded with status
   `user`, or one of solprop's supplied types — a
   {class}`~solprop.physdata.glasses.SellmeierGlass`, a custom
   {class}`~solprop.physdata.species.Species`, a rate of the caller's own — which carries
   its own citation ([quick experiments](material-selection.md#quick-experiments)).

A supplied object is never rebuilt or matched to a built-in by its name. The
[material-choices tutorial](../../auto_tutorials/basics/plot_06_material_choices) shows
all three on a short run, and reads from the runs' manifests which part of the physics
each change touched.

**What is supported.** The recommended models of the built-in gases, at optmat 0.8.0 (unchanged since 0.4.0)
(the shortest qualified wavelength of the index, rounded where shown; the
rate's stated uncertainty factor at 800 nm). Inspect the record for your own
carrier, and retain gaps in the qualified intervals:

| gas | index | qualified from | equation of state, transport | Kerr | Raman response | ionisation rate |
|---|---|---|---|---|---|---|
| He, Ne, Ar, Kr, Xe | `fit/He` … `fit/Xe` | 60, 76, 107.5, 125, 148 nm | `eos/`, `transport/` | `hyperpolarizability/` | none (atoms) | short-range PPT form; ×1.3 (He), ×1.5 (Ne, Ar), ×2 (Kr), ×3 (Xe) |
| H₂, D₂ | `fit/H2`, `fit/D2` | 112 nm | `eos/`, `transport/` | `hyperpolarizability/` | validated | molecular; ×1.5, ×2 |
| N₂, O₂ | `fit/N2`, `fit/O2` | 145, 195 nm | `eos/`, `transport/` | `hyperpolarizability/` | validated | molecular; ×1.5, ×2 |
| CH₄ | `fit/CH4` | 240 nm, five windows to 12.1 µm | `eos/`; transport provisional | `hyperpolarizability/` | validated | molecular; ×2 |
| SF₆ | `fit/SF6` | 230 nm to 5 µm | `eos/`, `transport/` | `hyperpolarizability/` | validated | molecular, untested; ×10 |
| N₂O | `fit/N2O` | 320 nm, four windows to 7.05 µm | `eos/`, `transport/` | `hyperpolarizability/` | provisional, opt-in | molecular; ×3 |
| air (`Gas.Air`) | `air/dry air (N2, O2, Ar)` | — | `eos/dry air (Lemmon 2000)` | none | none | none |

Every index model reaches the static limit except where a window ends. The solids are
in [solids](material-selection.md#solids): fused silica (`fit/SiO2`, absorbing, with its nonlinear bundle),
CaF₂, BaF₂ and N-BK7, the uniaxial MgF₂ and sapphire per ray (`axis="o"` or `"e"`,
required), and the metal tables of silver and aluminium. A gas's
[Raman response](raman-data.md#raman-responses-from-optmat), its
[ionisation rate](ionisation.md#rates-from-optmat) and fused silica's
[nonlinear bundle](material-selection.md#fused-silicas-nonlinear-response) have their own sections.

## Per-particle data and density scaling

A gas is described by its **per-particle** susceptibility $\gamma(\omega)$, in m³, rather
than by a refractive index at some fixed pressure. The linear susceptibility at a number
density $\rho$ is then

$$\chi^{(1)}(\omega) = \gamma(\omega)\,\rho, \qquad n = \sqrt{1 + \chi^{(1)}},$$

and the third order works the same way, $\chi^{(3)} = \gamma^{(3)}\rho$. Three things
follow.

First, a pressure gradient is a scalar multiplying one fixed spectral profile, which is
what lets the linear operator be sampled cheaply along a gradient fibre.

Second, $n - 1$ can be formed **directly**, as

$$n - 1 = \frac{\chi^{(1)}}{1 + \sqrt{1 + \chi^{(1)}}},$$

instead of as $\sqrt{1 + \chi} - 1$. For a gas at 1 bar $\chi \sim 10^{-4}$, so the
subtraction would throw away four significant figures, and at 1 mbar it would throw away
seven. {func}`~solprop.physdata.sellmeier.refractive_index_minus_one` uses the form
above.

Third, converting a published measurement into $\gamma$ means dividing it by the number
density **at the state the paper reports it for**. Getting that state wrong is a 1.3 %
error in every susceptibility, because 1 bar and 1 atm differ by that much, and it shows
up as a ratio to independent data that is flat in wavelength. optmat's polarisabilities
are dilute, zero-density quantities with nothing to divide out
([below](material-selection.md#units-reference-state-and-operating-state)); a caller's own published
expression records the state its source states as a
{class}`~solprop.physdata.sellmeier.ReferenceState`, and is normalised by the density
there.

The supplied analytic laws provide a first derivative with respect to angular
frequency. Accurate first derivatives matter when subtracting the moving-frame
group delay. Higher dispersion orders may use numerical differentiation of that
derivative; they are not all exact analytic derivatives. See [dispersion and
moving frames](../numerics/linear-operator.md) and the [design decisions](../../dev/decisions.md)
for the conventions and accuracy considerations.

## Refractive-index laws

{func}`~solprop.physdata.sellmeier.gamma` returns a gas's law: optmat's polarisability,
an {class}`~solprop.physdata.sellmeier.OptmatSusceptibility`, with its exact derivative
and its qualified wavelength intervals. `model=` names another of optmat's models of
the gas (`gamma("Ar", model="borzsonyi_2008/sellmeier_Ar")`), and a selection does the
same for a run ([selecting gases](material-selection.md#selecting-gases)). The accessors
{func}`~solprop.physdata.sellmeier.chi1`,
{func}`~solprop.physdata.sellmeier.refractive_index` and their derivatives evaluate it
at a state with optmat's equation of state.

A caller's own published expression takes one of three formula types, each a rational
function of $1/\lambda^2$ with its derivative in closed form, its validity range, its
reference state and a citation:

- $\chi = \sum_i B_i \lambda^2/(\lambda^2 - C_i)$, divided by the reference density
  ({class}`~solprop.physdata.sellmeier.PoleSumSusceptibility`): Börzsönyi's two-pole fits
  and the three-pole fits extended into the vacuum ultraviolet;
- $n - 1 = A_0 + \sum_i B_i/(C_i - \lambda^{-2})$, squared into $\chi$
  ({class}`~solprop.physdata.sellmeier.InversePoleIndex`): Peck & Huang, Zhang and
  Bideau-Mehu, which the sources write in three different but equivalent ways;
- $(n-1)\times 10^8 = a + b/(c - \nu^2)$ with the wavenumber $\nu$ in cm⁻¹
  ({class}`~solprop.physdata.sellmeier.WavenumberPoleIndex`): He *et al.*

A tabulated law is a {class}`~solprop.physdata.species.TabulatedSusceptibility`. Each is
given to a custom {class}`~solprop.physdata.species.Species`, or to a mode or medium as
its `susceptibility=`.

### Validity ranges, and what callers must handle

Evaluating a law outside its range raises
{class}`~solprop.exceptions.MaterialRangeWarning`, naming the range and the
citation; an optmat law names the parts of the band outside its qualified intervals
([below](material-selection.md#qualified-intervals-statuses-and-warnings)). The two directions are not
equally behaved: some dilute-gas fits approach a smooth static limit at longer
wavelengths, while extrapolating towards shorter wavelengths can approach a pole.
Smooth extrapolation is not evidence of physical validity, especially across
absorption bands or gaps between qualified intervals.

Two consequences are deliberate, and are the caller's to deal with rather than this
module's:

- **A whole propagation grid will warn.** The spectral axis of a run extends past the
  wavelength limits on both sides, so evaluating a model over all of it straddles the
  validity range every time. Restrict to the physical band — the
  [active band](../setup/grids.md#the-active-band-and-absorbing-collars), where the
  dispersion relation lives anyway — or filter {class}`~solprop.exceptions.MaterialRangeWarning` deliberately. The warning
  is not suppressed here, because for a scalar query it is exactly the thing a user
  needs to be told.
- **A negative angular frequency raises** {class}`~solprop.exceptions.MaterialError`
  rather than returning a mirrored or masked value. An envelope grid in FFT order can
  reach below zero in its far wings; a refractive index is not defined there, so the
  caller slices to the physical band instead of being handed a number that means
  nothing.

```{eval-rst}
.. plot::
   :include-source: False
   :alt: Gas refractivity versus wavelength on logarithmic axes, with each gas labelled at the right-hand end.
   :caption: Refractivity of every gas at 1 bar and 293.15 K from optmat's recommended
             models, each drawn from the shortest wavelength its model is qualified for
             to 3.6 µm (the infrared-active molecules' qualified windows have gaps,
             drawn through here). A factor of twenty separates helium from xenon.

   import warnings

   import matplotlib.pyplot as plt
   import numpy as np

   from solprop.constants import C
   from solprop.physdata import Gas, gamma, refractive_index_minus_one

   # tab10 repeats after ten entries and there are thirteen gases.
   colours = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, len(Gas)))

   fig, ax = plt.subplots(figsize=(7.0, 4.2))
   ends = []
   for colour, gas in zip(colours, Gas, strict=True):
       model = gamma(gas)
       shortest, longest = model.wavelength_range
       wavelength = np.geomspace(shortest, min(longest, 3.6e-6), 300)
       with warnings.catch_warnings():
           warnings.simplefilter("ignore")
           refractivity = refractive_index_minus_one(
               gas, 2 * np.pi * C / wavelength, pressure=1.0
           )
       ax.plot(wavelength * 1e9, refractivity, color=colour, lw=1.4)
       ends.append((float(refractivity[-1]), wavelength[-1] * 1e9, gas.value, colour))

   ax.set_xscale("log")
   ax.set_yscale("log")
   ax.set_xlim(50, 5000)
   ax.set_xlabel("wavelength (nm)")
   ax.set_ylabel(r"$n - 1$ at 1 bar, 293.15 K")
   ax.grid(which="both", lw=0.3, alpha=0.4)
   fig.tight_layout()

   # Label each curve at its long-wavelength end rather than in a legend, pushing
   # labels apart in display space where curves end together.
   fig.canvas.draw()
   to_display, to_data = ax.transData, ax.transData.inverted()
   occupied = -np.inf
   for value, position, name, colour in sorted(ends):
       pixels = max(to_display.transform((position, value))[1], occupied + 11.0)
       occupied = pixels
       ax.annotate(
           name,
           (position, to_data.transform((0.0, pixels))[1]),
           textcoords="offset points",
           xytext=(5, 0),
           fontsize=7.5,
           color=colour,
           va="center",
       )
```

## Density

{func}`~solprop.physdata.equation_of_state.density` returns the number density in m⁻³
from optmat's real-gas equations of state (CoolProp's Helmholtz-energy equations, through
`optmat.gas.eos`), adapted by {class}`~solprop.physdata.species.OptmatGas`.
{func}`~solprop.physdata.equation_of_state.pressure` inverts it exactly and
{func}`~solprop.physdata.equation_of_state.ideal_gas_density` gives $p/k_\mathrm{B}T$ for
comparison.

```{note}
Pressures are in **bar**. This is the one documented departure from SI at the user
interface ([conventions](../../dev/conventions.md), §1), because the field works in bar.
Everything else is SI, and temperatures default to 293.15 K.
```

The equation of state is real-gas, which matters for the high-pressure fills used in
self-compression: argon at 293.15 K departs from ideal by 0.14 % at 2 bar and 2.5 % at 40 bar.
Below 0.1 bar every gas here is within 0.2 % of ideal, and below 1 mbar within
2 × 10⁻⁵. A pressure of exactly zero returns zero without consulting the equation of
state, which is undefined there — a gradient fill terminating in vacuum needs that. A
condensed state is refused ([below](material-selection.md#density-and-transport)).

## Third-order nonlinearity

{func}`~solprop.physdata.nonlinear_data.gamma3` is the per-particle hyperpolarisability in
m⁵/V², so $\chi^{(3)}$ is strictly linear in number density. It is optmat's electronic
Kerr coefficient evaluated at a carrier, so it needs one: `gamma3("Ar", wavelength=800e-9)`
([the electronic Kerr coefficient at the carrier](material-selection.md#the-electronic-kerr-coefficient-at-the-carrier)).
{func}`~solprop.physdata.nonlinear_data.chi3` and
{func}`~solprop.physdata.nonlinear_data.n2` need it too, as `wavelength=` or as the
selection's `carrier_wavelength`, and are refused without one. The nonlinear index
follows from

$$n_2 = \frac{3\chi^{(3)}}{4\varepsilon_0 c\, n_0^2},$$

with $n_0$ read at the same carrier;
{func}`~solprop.physdata.nonlinear_data.n2_from_chi3` and
{func}`~solprop.physdata.nonlinear_data.chi3_from_n2` are that relation and its inverse.
One run takes one coefficient per species, at its carrier.

Air has no Kerr coefficient and raises; its nonlinearity comes from the nitrogen and
oxygen of a mixture ([mixtures](mixtures.md#air)).

## Ionisation and Raman data

{func}`~solprop.physdata.species.ionisation_potential` returns optmat's first ionisation energy in
**joules** for every gas but air. A built-in species carries it as
{class}`~solprop.physdata.species.IonisationData`, a potential and its citation; the
rate itself is optmat's rate model ([ionisation](ionisation.md#rates-from-optmat)), and
the energy the plasma loss pays is that model's own.

A built-in gas's Raman response is optmat's
([Raman responses from optmat](raman-data.md#raman-responses-from-optmat)), and a built-in
species carries no Raman constants (`raman=None`).
{func}`~solprop.nonlinear.optmat_raman.raman_status` reports what optmat holds for a gas:
`"validated"` for N₂, O₂, H₂, D₂, CH₄ and SF₆, `"provisional"` for N₂O, and `"atom"` for
the noble gases. `prop_capillary` and `prop_waveguide` turn the Raman response on by
default for every validated molecule; for N₂O they leave it off and warn, except under
`MaterialSelection(raman="provisional")`, which turns optmat's provisional N₂O response
on, with one status warning per run. A caller's own molecular constants go on a custom
species as a {class}`~solprop.physdata.raman_types.MolecularRaman`, from
{mod}`solprop.physdata.raman_types` (also exported by {mod}`solprop.physdata`), with each
dephasing rate a {class}`~solprop.physdata.raman_types.Linewidth`: a constant term, a
Dicke-narrowing term and a collisional term, so a gas with only a measured $\tau_2$ is
the special case with the constant alone ([Raman](raman-models.md#the-models)).

## Silver and aluminium

{func}`~solprop.physdata.metals.silver` and
{func}`~solprop.physdata.metals.aluminium` return optmat's tables, `table/Ag` and
`table/Al` ([solids](material-selection.md#solids)), for waveguide walls. A table carries its interpolant's
exact derivative. Outside the tabulated support the endpoint is held with a warning;
that is not a physical extrapolation. For a good metal it is wrong in a known direction:
$n$ and $k$ grow as $\sqrt{\lambda}$ in the Hagen–Rubens regime, so a held endpoint
understates both beyond the long-wave end. An optical grid does not reach there, but a
terahertz grid on a metal wall would.

## Window glasses

Five glasses join fused silica for the offline optics of
[the compression page](../workflows/compression.md): a window, a pair of wedges, a lens.
{func}`~solprop.physdata.glasses.glass` finds any of them by name, and each factory
returns optmat's recommended model ([solids](material-selection.md#solids)):
{func}`~solprop.physdata.glasses.calcium_fluoride` (`CaF2`),
{func}`~solprop.physdata.glasses.magnesium_fluoride` (`MgF2`),
{func}`~solprop.physdata.glasses.bk7` (`BK7`, `N-BK7`),
{func}`~solprop.physdata.glasses.sapphire` (`sapphire`, `Al2O3`) and
{func}`~solprop.physdata.glasses.barium_fluoride` (`BaF2`). Magnesium fluoride and
sapphire are uniaxial and need a ray, `axis="o"` or `axis="e"`. A caller's own formula is
a {class}`~solprop.physdata.glasses.SellmeierGlass` — a sum of poles with an analytic
frequency derivative — or a {class}`~solprop.physdata.glasses.TabulatedGlass`, which
warn outside their stated range as optmat's models do.

Where solprop's choices differ from Luna.jl's, the differences are listed in
[Noteworthy differences from Luna](../workflows/luna.md).
