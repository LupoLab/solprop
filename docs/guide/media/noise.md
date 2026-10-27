# Shot noise

Weak fluctuations can grow into a substantial part of a propagated spectrum.
Solprop provides a classical one-photon-per-mode seed with random spectral phase,
useful for studying modulation instability, Raman-seeded broadening and
shot-to-shot coherence. Every run carries it by default, as `noise="input"`; pass
`seed=` to reproduce a realisation, `noise=None` for a noiseless run, or
`noise="modified"` for the alternative model described below.

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
     - input (the default)
     - modified
   * - what it is
     - a realisation added to the initial field at the launch
     - a realisation kept apart and passed, with the field, to the nonlinear response at
       every stage
   * - propagation
     - dispersed, amplified and absorbed like any other field
     - advanced by the unitary part of the linear operator: dispersion yes, gain and
       loss no
   * - saved field
     - the whole field, seed included
     - the field without the seed; processing does not restore it
   * - refused beside
     - single precision and the planar grid
     - those, plus interpolated planes, field planes for coupling and coupled flows
```

Both come from the same draw. {py:func}`~solprop.noise.input_noise` and
{py:func}`~solprop.noise.modified_noise` produce identical spectra for the same seed and
differ only in the label they record, because the difference between the models is
entirely in how the driver consumes the realisation.

### Where one photon per mode comes from

The seed is not the vacuum itself: in the symmetrically ordered (Wigner) picture that a
classical field simulation implicitly uses, the vacuum carries half a photon per mode.
One photon per mode is the equivalent input noise of a phase-insensitive amplifier.
[Smith, Appl. Opt. **11**, 2489 (1972)](https://doi.org/10.1364/AO.11.002489), appendix A,
showed that the spontaneous Raman or Brillouin emission along a fibre, each slice
amplified by the gain that remains after it, is the same at high gain as one photon per
mode injected at the input and propagated by the stimulated equations alone. So the
input model does carry spontaneous Raman scattering: as amplified vacuum, through the
ordinary Raman response.

What it gets wrong is known and bounded:

- For a parametric process (modulational instability, four-wave mixing), which has no
  reservoir, the exact seed is half a photon, so one photon overestimates spontaneous
  parametric power by a factor of two at high gain. It enters a supercontinuum
  logarithmically and shifts its development length by about $\ln 2/(2\gamma P_0)$.
- Under linear loss the seed decays with the field instead of staying at the vacuum
  level, so a spontaneous process far into a lossy guide is under-seeded by about the
  guide's power transmission up to that point.
- Spontaneous Raman scattering at a shift $\Omega$ is seeded at $n_\mathrm{th} + 1$
  photons on the Stokes side and $n_\mathrm{th}$ on the anti-Stokes side, with
  $n_\mathrm{th} = 1/(e^{\hbar\Omega/k_BT} - 1)$. At vibrational shifts that is one and
  zero, but for the rotational lines of nitrogen at room temperature (0.36–5 THz)
  $n_\mathrm{th}$ is about 1 to 17. Neither model includes it; a Raman Langevin term
  with the thermal factor would.
- The modulus of each bin is fixed. A gain mode spanning many bins sums many independent
  phases and approaches Gaussian statistics anyway; a single-bin statistic does not.

### Why the modified model is not the default

The modified model is due to
[Chen and Wise, arXiv:2410.20567](https://doi.org/10.48550/arXiv.2410.20567). solprop's
version differs from theirs in three ways: it passes the full field $\psi + N$ to the
nonlinear response rather than an expansion to first order in the noise, it has no
separate emission gain (solprop has no gain media), and it advances the draw by the
dispersive phase,

$$ N(z) = N_0(\omega)\,e^{\,i\,\mathrm{Im}\,\Phi(\omega, z)} , $$

rather than holding it fixed. Held fixed, a noise field drifts out of phase with the
field it seeds in a dispersive medium; Luna.jl pull request 440 reports that this is
wrong by 1.4 to 6.6 times per frequency in spontaneous modulational instability, and by
50 to 80 times in conservative propagation (solprop has not reproduced these figures).

Write the linear operator as $\hat D = \hat D_u + \hat D_r$, with $\hat D_u$ its unitary
part. The reconstructed field $\chi = \psi + N$ of the modified model obeys

$$ \partial_z \chi = \hat D\chi + \hat N(\chi) - \hat D_r N . $$

Where the linear operator is unitary, $\hat D_r = 0$ and $\chi$ is exactly the input
model's field, to all orders and whatever the nonlinear response, Raman scattering and
plasma absorption included: the two models are one calculation in two variables. Where it
is not unitary (propagation loss, which every default capillary run has, the absorbing
collars, and the normalisation along a varying guide) they differ by $-\hat D_r N$, a
forcing correlated with the original draw rather than fresh reservoir noise. It can be
the worse of the two: for a gain $G$ followed by a transmission $\eta$ it gives
$[1 + \sqrt\eta(\sqrt G - 1)]^2 - 1$ spontaneous photons against the true
$\eta(G - 1)$. The modified model is therefore kept as an option, not the default. Its
one practical advantage, a spectrum without the seed's floor, is available for an input
run from the [floor-subtracted spectrum](#the-floor-subtracted-spectrum).

### What noise costs

The cost of noise depends on how large the seed is against the field, not on the model:
the two take the same number of steps where they are the same equation. A seed of one
photon per bin over $n$ active bins is a share $r \approx \sqrt{n\hbar\bar\omega/U}$ of a
pulse of energy $U$ — about $10^{-2}$ for a 20 pJ fibre soliton and $10^{-4}$ or less for
a microjoule capillary pulse. For capillary pulses noise adds no measurable steps; for
the 20 pJ soliton of the introduction it multiplies them by 8 to 65, and those steps are
needed to integrate the seed's own dynamics. A schedule found without noise integrates
the noise part badly, so a replayed schedule for a noisy ensemble should be calibrated
with noise on, across more than one seed.

### The floor-subtracted spectrum

{py:func}`~solprop.processing.fields.get_spectral_density` takes `noise_floor="subtract"`,
which removes exactly one photon per active bin and channel. Because every bin's seed
has a fixed modulus, for one realisation this equals the modified model's corrected
readout $|\psi + N|^2 - |N|^2$ wherever the linear operator is unitary. Under loss the
input seed has decayed, so the subtraction removes more than remains; values near the
floor can be negative and are returned signed. It refuses a run without noise and a
modified run, whose saved field holds no seed. See
[processing](../results/processing.md#removing-the-noise-floor).

## Reproducibility

In a run, choose the model with the driver's `noise` keyword and fix the realisation with
`seed`; `prop_gnlse`, `prop_capillary`, `prop_waveguide`, `prop_radial` and `prop_free`
all take both. The keyword's default, `"default"`, resolves to `"input"` wherever the
run can carry noise. Where it cannot, it resolves to none: under single precision, whose
binary32 rounding swamps the seed, on the hybrid planar grid, which declares no photon
measure, and beside the TDSE response, whose runs are expensive, rarely repeated as
ensembles and checked coherently (an explicit noise model there is warned on). An interior launch keeps a default draw off the spectrum it is given,
which may already carry one, while an explicit `noise="input"` there adds the draw at the
launch. `seed=` beside a default that resolves to none is refused. The realisation and
its seed are stored in the output, with how the model was resolved
(`RunMetadata.noise_resolution`), and the start-of-run
summary prints both.

Input noise is carried beside every coupled linear flow (a coherent wall, the taper
coupling of a multimode capillary, a split block and a moving solid guide) and beside
the TDSE response. Its draw is part of the launch, so the flow, or the atom, sees it as
part of the field. It is one photon per channel and bin because each route's entrance
basis is orthonormal in true power: the capillary's HE₁ₘ set by its Bessel orthogonality
and per-mode power normalisation, and a general guide's basis by its Gram certificate (a
declared rotation keeps the total per bin and redistributes it among the channels). The
modified model is still refused on these routes: its draw is kept apart from the field,
and its transport through a coupled flow, or into the atom's drive, is not measured. The noise functions can also be called directly, and
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
Calibrate that schedule with noise on: one discovered on a noiseless run integrates the
seed's dynamics poorly.
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
