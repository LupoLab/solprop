# Shot noise

Weak fluctuations can grow into a substantial part of a propagated spectrum.
Solprop provides a classical one-photon-per-mode seed with random spectral phase,
useful for studying modulation instability, Raman-seeded broadening and
shot-to-shot coherence. Select `noise="input"` or `noise="modified"` and provide
`seed=` to reproduce a realisation.

Begin with the [modulation-instability ensemble](../../auto_examples/supercontinuum/plot_03_modulational_instability_supercontinuum.rst)
and [comb-to-continuum shot](../../auto_examples/supercontinuum/plot_04_raman_comb_supercontinuum.rst).
A seed ensemble estimates the variability of the chosen noise model. It does not
automatically include experimental pulse-energy, duration, timing or phase noise;
those need explicit input variations.

## One photon per bin

The rule is that each mode of the field — each frequency bin, each waveguide mode, each
polarisation component — starts with the energy of one photon, $\hbar\omega$. Written as
a spectral energy density that is

$$ S(\omega_k)\,\delta\omega = \hbar\omega_k , $$

and since [conventions](../../dev/conventions.md), §4, fixes $S(\omega)$ in terms of the
stored DFT bins, the amplitude follows immediately:

$$ |\tilde E_k| = \sqrt{\frac{\hbar\omega_k}{\delta\omega}}\;\frac{\sqrt{2\pi}}{2\,\delta t}
   \quad\text{(field grid)}, \qquad
   |\tilde A_k| = \sqrt{\frac{\hbar\omega_k}{\delta\omega}}\;\frac{\sqrt{2\pi}}{\delta t}
   \quad\text{(envelope grid)} . $$

The factor of two between them is the analytic signal's doubling of every interior bin of
a one-sided spectrum, and it pairs exactly with the two prefactors of
{py:func}`~solprop.fourier.spectral_density`. The two are a matched set: change one and
the photon number changes with it. {py:func}`~solprop.noise.photons_per_bin` reads the
number back out of any stored spectrum and is how the pairing is tested.

**The modulus is deterministic; only the phase is random.** This is the standard
one-photon-per-mode seeding of the supercontinuum literature. It assigns exactly one photon of energy to each active bin; it is not a full
quantum-state model, whose modulus would be Rayleigh-distributed. Ensembles over seeds
sample the random phase of that fixed spectral amplitude, not amplitude noise. It also
differs from the truncated-Wigner vacuum of
[Drummond and Corney, J. Opt. Soc. Am. B **18**, 139 (2001)](https://doi.org/10.1364/JOSAB.18.000139),
Eq. (6.13), which assigns half a photon per mode with a Gaussian random amplitude, and from
their Raman Langevin noise (Eq. 6.16), which solprop does not add. This semiclassical prescription and its distinction from
technical pump noise are illustrated by [Adamu et al., Scientific Reports (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7080841/).

```{note}
Because the modulus is fixed, the photon number per bin is exactly one — to round-off,
not to a statistical error. A test of the noise amplitude is a test of arithmetic, not a
test of a distribution; what needs a statistical test is the *phase*, which is checked
against a uniform distribution with a Kolmogorov–Smirnov statistic.
```

## The active band, on both grids

Noise goes only where the physics is: the active mask
$\omega_{\min}/2 < \omega < 1.1\,\omega_{\max}$ of
[conventions](../../dev/conventions.md), §3 (see also
[the active band](../setup/grids.md#the-active-band-and-absorbing-collars)), on both
grids. Outside it the stored spectrum
is exactly zero. On a field grid this matters: without the mask, noise would sit in the
zero-frequency bin and above the window edge. A photon at $\omega = 0$ is not a photon,
and energy placed above the apodised window edge is energy the spectral collar would
spend the run absorbing.

The random phase, by contrast, is drawn over the *whole* spectral axis and the amplitude
is masked afterwards. That way the random stream does not shift when the band moves, and
a seed keeps meaning one realisation on a given grid size.

## Two models

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * -
     - input
     - modified
   * - what it is
     - a realisation added to the initial field at ``z = 0``
     - a realisation injected into the nonlinear operator at every step
   * - propagation
     - dispersed, amplified and absorbed like any other field
     - advanced by the unitary part of the linear operator: dispersion yes, gain and
       loss no
   * - strength
     - the physical picture, correct wherever seeding happens near the input
     - no artificial four-wave mixing from a propagated background
   * - cost
     - nothing during the run
     - about eight times as many integrator steps
   * - use it when
     - the process is seeded early and the fibre is short
     - the seeding is distributed, or the spontaneous gain is what you are measuring
```

Both come from the same draw. {py:func}`~solprop.noise.input_noise` and
{py:func}`~solprop.noise.modified_noise` produce identical spectra for the same seed and
differ only in the label they record, because the difference between the models is
entirely in how the driver consumes the realisation.

### Why the modified field must be advanced

The modified model is credited to Chen and Wise, who derived it for a gain medium with no
dispersion; solprop has not traced it to a specific publication. Taken literally, it holds the noise field constant in $z$. In a dispersive
medium that is wrong: the noise source and the field it seeds drift out of phase. Measured
against the exact Bogoliubov result for spontaneous modulation instability, a constant
noise field is wrong by a factor of 1.4 to 6.6 per frequency, rising to 50–80 in
conservative propagation, as reported in Luna.jl pull request 440 (solprop has not
reproduced these figures).

solprop draws one realisation $N_0(\omega)$ and uses

$$ N(z) = N_0(\omega)\,e^{\,i\,\mathrm{Im}\,\Phi(\omega, z)} , $$

with $\Phi$ the same cumulative linear operator the stepper already has, so the noise
disperses but neither grows nor decays: only the phase of the linear operator is applied
(dispersion yes, gain and loss no), because the source stands for vacuum fluctuations
that are present afresh at every $z$ (see the [design decisions](../../dev/decisions.md)).
It costs one complex
multiply-add per element per stage and no extra transform.

The price of the modified model is step count: an undecaying broadband source is injected
at every $z$, and a pure-Kerr run needs roughly eight times as many steps as the same run
with input noise (the figure Luna.jl pull request 440 reports).

## Reproducibility

In a run, choose the model with the driver's `noise` keyword (`None`, the default,
`"input"` or `"modified"`) and fix the realisation with `seed`;
`prop_gnlse`, `prop_capillary`, `prop_waveguide`, `prop_radial` and `prop_free` all take
both (the hybrid planar route, `prop_planar`, does not carry noise). The realisation and
its seed are stored in the output. The noise functions can also be called directly, and
the generator is always passed in, never created inside:

```{code-block} python
import numpy as np
from solprop.noise import input_noise

realisation = input_noise(grid, np.random.default_rng(20260905), n_modes=3, n_pol=2)
realisation.seed          # 20260905, recorded for the output file
realisation.spectrum.shape  # (3, 2, n_omega)
```

Each mode and each polarisation component gets an independent realisation, because they
are separate field modes and each is entitled to its own photon. The entropy the
generator was created with is recovered and stored on the
{py:class}`~solprop.noise.NoiseRealisation`, so a run can be replayed from its output
file — including a run that was left unseeded, which records the operating-system entropy
it drew. No draw ever uses NumPy's global generator.

A mode axis and a polarisation axis are always present: every draw has the shape
`(n_modes, n_pol, n_omega)`, so a one-mode run and the first mode of a two-mode run with
the same seed see the same realisation. For the radial, Cartesian and line grids,
{py:func}`~solprop.noise.spatial_noise` draws one photon per admitted transverse column,
polarisation and frequency bin in the same way. It takes the entrance plane's
{py:class}`~solprop.spatial.SpatialFlux`: each column and bin is scaled so that its
energy, weighted by the real axial admittance of the entrance medium, is $\hbar\omega$,
the measure a launch is normalised in. Nonpropagating, inactive and zero-frequency bins
are not power ports and carry no photon; the random stream is drawn as before, so a seed
keeps its meaning ([energy and flux in a medium](../free-space/flux.md#the-discrete-measure-endpoints-and-noise)).
A planar grid carries no noise draw.

An ensemble is a sequence of seeds. Run it on a fixed step schedule (`schedule=`, see
[scans](../workflows/scans.md)): the numerical error of a fixed schedule is then a common systematic
across the members, so the spread between members is the physics and not the stepper.
The [supercontinuum examples](../../auto_examples/supercontinuum/plot_03_modulational_instability_supercontinuum)
run such ensembles and compute the relative intensity noise and the coherence from
them ([processing](../results/processing.md)).

## A declared change of basis

One photon per channel per bin is a statement about an **orthonormal true-power** basis:
only there does "one photon in this channel" mean an energy of $\hbar\omega$. A
[general modal guide](../waveguides/general-modal.md) that declares a unitary change of its channel
basis therefore draws **once, in the reference basis, with the same seed and the same
stream**, and carries that one realisation into the run's coordinates through the
inverse transform, {py:func}`~solprop.noise.transformed_noise`.

```{code-block} python
from solprop.noise import transformed_noise

carried = transformed_noise(realisation, basis.transform)  # d = U.H c
```

The **total** photon number in each bin is exactly what it was, because $U$ is unitary;
the per-channel numbers are redistributed and the moduli are no longer all equal. That
is the physics, not a defect: a fixed-modulus random-phase draw is *not* invariant in
distribution under unitary mixing, so drawing again in the declared basis would be a
different stochastic model. A basis with unequal channel norms is neither orthonormal
nor a unitary transform of one, and a draw in it is refused by name.

Where solprop's choices differ from Luna.jl's, the differences are listed in
[Noteworthy differences from Luna](../workflows/luna.md).
