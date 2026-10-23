(primer-linear-media)=
# 7. Linear response and material data

Every quantity in the propagation equation that is not geometry comes from the medium:
the refractive index of the gas and the wall, the nonlinear susceptibility, the Raman
response, the ionisation rate, and the number density that scales all of them. The
quality of a simulation is bounded by the quality of these data, and the uncertainty of
the data is usually far larger than the numerical error. This short chapter explains
where solprop's material data come from, how a gas's per-particle data become the
susceptibility of a fill at a given pressure, and what to do when a spectrum leaves the
range over which the data were qualified.

The three chapters that follow take the nonlinear responses in turn. This one is about
the linear response, and about the division of labour that runs through all of them.

## 7.1 optmat and solprop: who owns what

solprop keeps no material database of its own. All of its built-in gases, solids and
responses come from [optmat](https://github.com/LupoLab/optmat), a separate package of
curated optical-materials data with its own [documentation](https://optmat.readthedocs.io)
([materials](../guide/media/materials.md#who-owns-what-and-how-to-choose)). The split is
deliberate:

- **optmat owns the data.** It holds every number that describes a material, the model the
  number comes from, its sources, the range over which it is qualified, its reference state,
  its status and its stated uncertainty.
- **solprop owns the propagation physics built on the data.** That covers the scaling of a
  gas's index with density and the error of that scaling, the mixture rule, the gas-flow
  problem, the mode equations, how the Kerr coefficient is used (frozen at the carrier, or
  dispersive), the tables of ionisation rates, and the mapping of a Raman response onto the
  kernel.

Every run records both halves in a **material manifest**. It lists each model by name and
content digest, and each choice solprop made with it
([selecting a run's materials](../guide/media/material-selection.md#selecting-a-runs-materials)).
An omitted selection means optmat's recommended model for every material. You can name
another model, modify a model, or supply an object of your own, and the manifest records which.

## 7.2 Gases: from polarisability to refractive index

A gas is described by its **per-particle** polarisability, not by a refractive index at some
fixed pressure. For a dilute gas of number density $N$, the linear susceptibility is
proportional to density,

$$
\chi^{(1)}(\omega)=\frac{N\alpha(\omega)}{\varepsilon_0},\qquad n=\sqrt{1+\chi^{(1)}} ,
$$

and similarly for the third-order susceptibility,
$\chi^{(3)}=N\gamma^{(3)}$ ([per-particle data and density
scaling](../guide/media/materials.md#per-particle-data-and-density-scaling)). This form has two
practical advantages. A pressure gradient becomes one scalar multiplying a fixed spectral
profile, which makes the linear operator cheap to evaluate along a gradient fibre (Chapter 12).
And the refractivity can be formed directly as

$$
n-1=\frac{\chi^{(1)}}{1+\sqrt{1+\chi^{(1)}}},
$$

instead of as $\sqrt{1+\chi}-1$. At 1 bar $\chi\sim10^{-4}$, so the subtraction would discard
four significant figures, and at 1 mbar seven.

Linear density scaling is the dilute limit of the Lorentz–Lorenz relation that optmat itself
uses, $(n^2-1)/(n^2+2)=N\alpha/3\varepsilon_0$, with a refractivity virial for the remaining
density dependence. The two differ by a relative amount of about $\chi/3$: $2\times10^{-4}$ in
$n-1$ for argon at 1 bar, 0.57% at 30 bar, and 1.9% for xenon at 30 bar. solprop keeps the
linear law, which is what makes the gradient cheap, and warns when the difference exceeds 1%.
That happens near 50 bar for argon and nitrogen and near 17 bar for xenon
([the gas index at linear density
scaling](../guide/media/material-selection.md#the-gas-index-at-linear-density-scaling)).

The density itself comes from a real-gas equation of state through optmat, $N=p/(Zk_BT)$. For
argon at room temperature the compressibility $Z$ departs from one by 0.14% at 2 bar and 2.5% at
40 bar. A condensed state is refused: sulphur hexafluoride liquefies at 21 bar at 20 °C. Pressures
are given in bar, the one documented exception to SI. It is worth knowing that 1 bar and 1 atm
differ by 1.3%. That is a flat error in every susceptibility, so the ratio to independent data is
the same at every wavelength.

The polarisabilities themselves are curated from the literature, and for the lighter gases from
*ab initio* calculations. optmat's recommended helium model rests on the calculations of Puchalski
and colleagues, and its argon model on those of Lesiuk and Jeziorski. Each model is qualified over
a stated range of wavelengths: from 60 nm for helium and 107.5 nm for argon to the static limit, and
in windows between infrared absorption bands for the molecules
([materials](../guide/media/materials.md)). Near the short-wavelength end of the range, resonances
make the index rise steeply. A spectrum that extends there is often exactly what an experiment
wants to predict, and also where the data are least certain.

## 7.3 Mixtures

A mixture of gases sums its susceptibilities before taking the square root,

$$
\chi(\omega,z)=\sum_s\rho_s(z)\gamma_s(\omega),\qquad n(\omega,z)=\sqrt{1+\chi(\omega,z)} ,
$$

with each population at the density of its pure gas at its partial pressure
([gas mixtures](../guide/media/mixtures.md)). This differs from a full mixture equation of state by
about 0.15% at 10 bar. Each population also carries its own nonlinear data: its Kerr coefficient,
its ionisation rate, which produces its own electrons, and its Raman response. The same mixture
object feeds the linear susceptibility and the nonlinear densities, so the two can never disagree
about what is in the fibre. Air is a mixture of nitrogen, oxygen and argon in this sense. A separate
dispersion-only entry gives just the refractive index of dry air.

## 7.4 Solids

A solid enters a simulation in two ways: as the wall of a hollow guide, where only its index
matters, and as the core of a solid fibre or a bulk medium, where its nonlinearity matters too.
solprop's default fused silica is optmat's `fit/SiO2`. It starts from the universal dispersion model
of Franta and colleagues (Franta et al. 2016), which is causal and absorbing from 24.8 nm to 125 µm
([solids](../guide/media/material-selection.md#solids)). Causality matters here for the reason
given in §1.3: a model fitted only to the refractive index in the transparent region, a Sellmeier
formula, has poles where its resonances are placed and is not trustworthy outside its fit. The other
solids are Sellmeier fits over their stated ranges: calcium and barium fluoride, N-BK7, and the
uniaxial magnesium fluoride and sapphire, taken per ray. Silver and aluminium are tabulated for metal
walls.

Where a mode solver needs a real index, as the exact step-index solver of Chapter 4 does, solprop
takes the glass's transparent approximation and warns when the absorption it drops exceeds 100 m⁻¹.
Fused silica's nonlinearity comes as a **bundle**: a total long-pulse $n_2(\lambda)$, a Raman shape
of unit integral and that shape's own Raman fraction, held together so that they cannot be
mismatched (Chapter 9).

## 7.5 Qualified ranges, and what to do at their edges

Every optmat model states the range over which it was qualified, and solprop checks the run's
spectrum against it. Inside the range, the model's stated uncertainty is the accuracy of the data.
Outside it, the model may still return a number, and the run will use it, with a warning
([qualified intervals, statuses and warnings](../guide/media/material-selection.md#qualified-intervals-statuses-and-warnings)).

Such a warning is a prompt to interpret, not to tighten a numerical tolerance. A smaller step does
not make an extrapolated refractive index more accurate. The useful questions are physical:

- Does the observable you care about depend on the spectrum in that region?
- How much would it change if the index or rate there were different by its plausible uncertainty?

The same applies to the nonlinear data. A dispersive wave in the vacuum ultraviolet is sensitive to
the Kerr coefficients of legs far outside any measured range. Chapter 8 shows how big that
sensitivity can be. For ionisation rates, optmat states an uncertainty factor at 800 nm: ×1.3 for
helium up to ×3 for xenon among the atoms ([materials](../guide/media/materials.md)). The
consequence of a factor-of-two uncertainty in a rate that rises as the eighth power of the intensity
is not obvious until you test it.

:::{admonition} What to remember
:class: tip

- optmat owns the material data, their sources, their qualified ranges and their uncertainties.
  solprop owns the propagation physics built on them. A run's manifest records both.
- A gas is described per particle: $\chi^{(1)}=N\alpha/\varepsilon_0$ and $\chi^{(3)}=N\gamma^{(3)}$.
  solprop keeps linear density scaling, whose error relative to Lorentz–Lorenz is about $\chi/3$,
  and warns above 1%. $n-1$ is formed directly.
- The density comes from a real-gas equation of state. Pressures are in bar; 1 bar and 1 atm differ by
  1.3%.
- A mixture sums its susceptibilities before the square root, and each population brings its own
  nonlinear data.
- Default fused silica is causal and absorbing over 24.8 nm–125 µm. Sellmeier fits are trustworthy
  only within their stated range.
- A warning about a qualified range calls for a physical sensitivity check, not a smaller step.
:::

The precise statements are in [materials](../guide/media/materials.md), [selecting and inspecting
material data](../guide/media/material-selection.md) and [gas mixtures](../guide/media/mixtures.md),
and in optmat's own documentation. [Chapter 8](kerr.md) turns to the third-order response.
