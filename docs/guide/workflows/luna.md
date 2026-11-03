# Noteworthy differences from Luna

solprop is the successor of [Luna.jl](https://github.com/LupoLab/Luna.jl), the Julia
pulse-propagation code by the same authors (see [About solprop](../../introduction/about.md)). It covers
many of the same physical problems, with additional geometries and response
models. The two codes have different conventions, defaults and supported
combinations; matching an input script does not by itself match a calculation. Where solprop's choices differ
from Luna's — a convention, a default, a corrected formula or data set, a more accurate
numerical method — the difference is recorded here, so that a Luna user knows what to
expect and a comparison between the two codes can be set up to agree.

The rest of this manual describes solprop on its own terms; this page collects the
practical differences. The qualification evidence behind the comparisons is in the
[accuracy record](../../dev/accuracy.md).

## Coming from Luna

The keyword interfaces keep Luna's shape: `prop_capillary(radius, length, gas,
pressure; ...)` becomes `prop_capillary(radius, length, gas, pressure, ...)` with the
keywords in Python spelling and every quantity in SI units (pressures stay in bar).

| Luna | solprop |
|---|---|
| `λ0`, `τfwhm`, `τw`, `ϕ`, `λlims`, `saveN`, `shotnoise` | `wavelength`, `fwhm`, `width`, `phases`, `wavelength_limits`, `n_save`, the default `noise="input"`, with a `seed` |
| `pulses=[Pulses.GaussPulse(...), ...]` | `pulses=[GaussianPulse(...), ...]` ({class}`~solprop.pulses.PulseSum`) |
| `modes=:HE11` (mode-averaged), `modes=1`, `modes=4` | `mode_average=True`, `modes=1`, `modes=4`, or a list of mode objects; solprop's default is the resolved `modes=1` |
| `Plotting.prop_2D`, `time_1D`, `spec_1D` | `prop_2d`, `time_1d`, `spec_1d` |
| `trange`, `λrange`, `dBmin` of a plot | `time_range`, `spec_range`, `db_min`; `spec_range` is in the units of the plot's axis, and `wavelength_to_axis` converts a wavelength range. `trange` stays the time window's length on `prop_*`, as it is in Luna |
| `Plotting.stats`, `Plotting.energy` | `plot_stats`, `plot_energy` |
| `Processing.getEt`, `getEω`, `getIω` | `get_field`, `get_spectrum`, `get_spectral_density` |
| `getEt(...; propagate=f, FTL=true)` | `get_field(..., transfer=f, transform_limit=True)` |
| `Processing.getφ`, `nearest_z` | `get_phase`, `nearest_plane` |
| statistics `fwhm_t_min`, `fwhm_t_max` | `fwhm_t`, `fwhm_t_outer` |
| `Tools.params`, `Tools.capillary_params` | `soliton_params`, `capillary_params` |
| `Tools.λRDW`, `pressureRDW`, `pressureZDW` | `lambda_rdw`, `pressure_rdw`, `pressure_zdw` |
| `Tools.getN`, `Ld`, `Lnl`, `Lfiss`, `getγ` | `soliton_order`, `dispersion_length`, `nonlinear_length`, `fission_length`, `nonlinear_coefficient` |
| `Fields.prop_taylor`, `prop_material`, `prop_mirror` | `TaylorPhase`, `Material`, `ChirpedMirror` (elements, applied with `transfer=`) |
| `Fields.optcomp_taylor`, `optcomp_material`, `optfield_cep` | `compress_taylor`, `compress_material`, `optimise_cep` |
| `PhysData.lookup_mirror(:PC70)` | `ChirpedMirror.lookup("PC70")` |
| `PhysData.ref_index(gas, λ, P)`, `PhysData.γ3_gas(gas)`, `PhysData.density(gas, P)` | `refractive_index(gas, omega, pressure=P)`, `gamma3(gas, wavelength=λ0)` (optmat's Kerr fit, at a carrier it requires), `density(gas, P)`; all optmat's models |
| `Ionisation.ionrate_fun!_PPTcached(gas, λ0)` | `OptmatRate(gas, λ0)`, optmat's recommended rate; `MaterialSelection(ionisation="ppt_ilkov")` for a literature PPT form |
| pseudo-gases such as `:HeB`, `:HeJ`, `:ArB` | the gas with another of optmat's index models, `MaterialSelection(linear="borzsonyi_2008/sellmeier_He")` or `susceptibility=gamma("He", model=...)` |

Four conventions change what the numbers mean:

- **Spectra in files use the optics sign convention**, $\tilde E(\omega) = \int E(t)\,
  e^{+i\omega t}\,dt$; Luna's files store the FFT convention. `fields(convention="fft")`
  returns the stored field in Luna's convention, bin for bin, and
  {func}`~solprop.processing.fields.get_spectrum` carries the same normalisation (`fftnorm`) so
  that a spectrum from either code is the same number in the same bin. See the
  [conventions](../../dev/conventions.md).
- **Modal amplitudes are true power.** solprop's launch and output amplitudes are
  normalised to the Poynting power in the guide; Luna's to a vacuum amplitude. A Luna
  run of nominal energy $E$ therefore carries $\bar w E$ of true energy, with
  $\bar w = \mathrm{Re}\,n_\text{eff}$ averaged over the pulse — 1.0000357 on the
  standard helium dispersive-wave case, a difference of 0.0036 %. See
  [power, energy and intensity](../results/power-and-intensity.md).
- **Free-space energy and intensity are the axial Poynting flux in the medium.**
  solprop's radial, Cartesian and line launches, noise and statistics measure the field
  with the axial admittance of the medium at the plane, $n_0\varepsilon_0c|E|^2/2$ for a
  paraxial beam; Luna's free-space normalisation and statistics use the vacuum factor
  $\varepsilon_0c/2$. A Luna free-space launch of energy $E$ in a medium of index $n_0$
  therefore has the field of a solprop launch of $n_0E$, and solprop reports $n_0$ times
  Luna's energy and intensity for the same field; to start from the same field, scale the
  energy by $n_0$. Local intensities are signed fluxes, and the field that drives
  ionisation is reported separately. See [energy and flux in a medium](../free-space/flux.md).
- **Free-space boundaries are watched, and absorbed only on request.** Luna.jl's
  free-space routes apply no transverse absorber, so a radial run reflects from its outer
  radius and a Cartesian one wraps. So does solprop by default, but it detects light
  reaching the edge and stops the run, naming a size to try, carries a safety ramp at the
  grid's highest transverse wavenumbers, and offers a designed real-space collar on
  request ([transverse boundaries](../free-space/model.md#transverse-boundaries)).
- **The unrotated `HE_1m` mode is polarised along x**; Luna's is polarised along y. The
  two are the two members of the degenerate pair, a quarter turn apart
  ($\phi_\text{solprop} = \pi/(2n) - \phi_\text{Luna}$ for a family of order $n$).

## Defaults that differ

| what | Luna | solprop |
|---|---|---|
| Raman response of a molecular fill | on for N₂, H₂, D₂, N₂O, CH₄ and SF₆ | optmat's response, on for every species whose response is validated: N₂, O₂, H₂, D₂, CH₄, SF₆ (and air through N₂ and O₂); N₂O's is provisional, so it warns and runs without unless `MaterialSelection(raman="provisional")` admits it |
| ionisation rate of a molecule | ADK for H₂, D₂, N₂O, CH₄, SF₆; PPT for N₂ and O₂ | optmat's recommended rate model of every gas ([SEP 3](../../dev/seps/archive/sep-0003-optmat-materials.md)) |
| barrier-suppression correction | none in released Luna (the pull request that proposes one uses Zhang's) | optmat's atomic rates carry Tong and Lin's correction, its molecular ones their own ([ionisation](../media/ionisation.md#rates-from-optmat)) |
| a pressure gradient to vacuum | reaches zero pressure, where $dp/dz$ diverges | the steady flow is **choked** at the exit, at the pressure where the laminar throughput is sonic; `vacuum_end="floor"` with `pressure_floor=0.0` reproduces a true vacuum |
| the fill of a taper | the uniform-bore law | the steady flow follows the bore ($p^2$ linear in $\int a^{-4}\,dz$), with a flow-regime check |
| `prop_gnlse(raman=False)` | still assigns $(1-f_R)$ of $\gamma$ to Kerr, so 82 % of the requested nonlinearity (an "$N = 6$" soliton is really $N = 5.43$) | all of $\gamma$ is Kerr; Luna's $(1-f_R)$ with `raman_fraction` given explicitly |
| `prop_gnlse(raman=True)` fraction | $f_R = 0.18$ with the Hollenbeck–Cantrell shape | optmat's silica bundle: the same shape with $f_R = 0.2188$, paired with optmat's total $n_2$ fit |
| step control | `rtol`, `atol`, the initial step and `dz_max` cannot be set through the interface | all are keywords; `prop_capillary` defaults to Luna's $10^{-6}$ and $10^{-10}$ with `Multistep()`, `prop_waveguide` to $10^{-5}$ and $10^{-10}$ and `prop_gnlse` to $10^{-7}$ and $10^{-14}$, both with `RungeKutta()` ([the defaults](../numerics/choosing-a-stepper.md#the-defaults)) |
| initial step | $10^{-4}$ m whatever the fibre | $10^{-4}$ of the length; `dz_max` half the length in both |
| radial transverse quadrature | 64 nodes | $\max(32, 8 \times \text{modes})$ |
| taper of the nonlinear polarisation in the spectral collar | always | off (`polarisation_taper=False`): the absorber acts on the field, and the taper costs a systematic error the size of the field in the collar |
| `temperature` | a constant, reaching the density, the mode set and the Raman response | also reaches the steady flow, and may vary along z |
| default capillary cladding | an uncited 400-row complex table, extrapolated beyond its ends | optmat's absolute fused silica `fit/SiO2`, causal and absorbing from 24.8 nm to 125 µm |
| gas indices, equations of state, Kerr coefficients and Raman responses | Luna's own tables | optmat's curated models ([materials](../media/material-selection.md#materials-from-optmat)) |

The rate-based path uses optmat's own effective electron numbers rather than
Luna's ionisation occupancy factor. The optional [helium TDSE
response](../media/tdse.md), which Luna.jl has no counterpart for, replaces the Kerr and
plasma terms of helium by a microscopic atomic current beyond linear response, with an
explicit current scale (an empirical 1.537 for the single-active-electron model, exactly
one for haCC) that is neither an ionisation occupancy nor a fitted electron count. Its
source is gated in time with the gate's switch-off edge removed
([the edge-free gated source](../media/tdse.md#the-edge-free-gated-source)), and the
rate-based defaults are unchanged.

## Corrected physics and data

These change the numbers a simulation produces.

**Ionisation.** solprop keeps no rate formula of its own: its rates are optmat's
curated models ([ionisation](../media/ionisation.md#rates-from-optmat)). Luna's ADK sets
$l^* = 0$ and $Z = 1$ without angular factor, so it is not the $\gamma \to 0$ limit of
its own PPT; optmat names that potential-only form `adk_hydrogenic` for a molecule. SF₆'s
first ionisation energy is optmat's, 15.35 eV; Luna's is 0.5 hartree, the hydrogen
value.

**Plasma.** A pre-ionised fraction $f_0$ multiplies the survival exponential,
$f = 1 - (1 - f_0)\,e^{-\int W dt}$, so that $df/dt = W(1-f)$ exactly; Luna adds it
outside, $f_0 + 1 - e^{-\int W dt}$, which can exceed one and unbalances the loss term.
The reported electron density is the same ionised fraction the polarisation uses. Luna
has no plasma response on an envelope grid; solprop has a separate cycle-averaged
[envelope plasma](../media/envelope-plasma.md) model, and refuses the field plasma
response there. Luna forms the column's three running integrals with the cumulative
trapezoid, which is 9 % short per integral at $\omega_{\max}$ on a default grid, and 11 %
short at the top of the active band, $1.1\,\omega_{\max}$. solprop
offers a band-exact rule ([SEP 18](../../dev/seps/archive/sep-0018-band-exact-plasma-integrals.md),
`integration="band"`) and keeps the trapezoid, `integration="trapezoid"`, which every
comparison with Luna states among its frozen inputs.
solprop's field-grid plasma holds the post-pulse current through
`DFT[gJ]/(i omega) - DFT[g'J]/(i omega)**2`
([SEP 27](../../dev/seps/archive/sep-0027-held-plasma-current.md)); Luna's gated polarisation
switches off both the residual current and polarisation at the temporal collar. The
propagation comparisons with Luna therefore run Luna's gated source through a test-only
hook, under a separate identity that cannot be resumed.


**The modal nonlinear prefactor.** solprop's modal prefactor is
$-i\omega/(2\varepsilon_0 c\,\operatorname{Re} n_\text{eff})$: it keeps the
$1/n_\text{eff}$ that Luna's drops, a relative effect of about 1.3×10⁻⁵ on the
reference comparisons ([accuracy](../../dev/accuracy.md#3-agreement-with-luna-at-the-level-of-a-kernel)).
The GNLSE sets that index to one (`prefactor_index=1.0`).

**Raman.** The vibrational response's denominator is $2\mu\Omega$, not the $4\mu\Omega$
Luna inherits, so a vibrational amplitude built from the same constants is twice
Luna's. The response function $h$ is linear in $\varepsilon_0$ (with $\varepsilon_0$
outside the integral); Luna's is quadratic, so a Luna value converts by dividing by
$\varepsilon_0$. The rotational ladder's truncation counts levels (Luna compares a $J$
value with an array index, off by one at `min_j = 0`). The vibrational dephasing is
finite at zero density (Luna evaluates $e^{-0/0}$). The gas responses themselves are
optmat's, not Luna's constants; N₂O's is provisional and is used only when asked for,
where Luna silently drops its vibrational term. Vector Raman is supported; Luna raises.

**Gas refractive indices.** Luna's implementation of the He *et al.* expression returns
$n/\rho$ instead of $(n^2-1)/\rho$: methane at 1 bar comes out at $n = 1.4037$ instead
of 1.00042, and the $\chi^{(3)}$ of N₂O, which is converted through that index, is 1.98
times too large. The Ermolov helium expansion is referenced to 1 atm and 0 °C, not to
1 bar (1.2–1.3 % in helium's refractivity). solprop's built-in gas laws are optmat's,
whose reference states are recorded per model, and dispersion orders are computed analytically, not by finite
differences.

**Metals.** The silver and aluminium indices are optmat's tables (`table/Ag`,
`table/Al`), not Luna's stitched ones.

**Capillary modes.** The imaginary part of $n_\text{eff}$ is clamped at zero wherever
the square root would give gain (Luna clamps only inside its real-part floor, and with a
cladding index below one returns gain, e.g. $n_\text{eff} = 0.115 - 57.3i$). The reduced
model's loss has one arithmetic kernel (Luna's cached path had its sign reversed).
$n_\text{eff} - 1$ is formed directly, gaining five digits at a gas fill. The azimuthal
order is never negative, and frequency zero is refused.

**Antiresonant fibres (Vincetti model).** Four terms follow the papers or the physics
rather than Luna's (and pyJCMwave's) form; the legacy form is available as
`solprop.modes.antiresonant.LEGACY_CONVENTIONS`. For the tutorial fibre (15 µm core,
six 9 µm tubes with 180 nm walls) in argon:

| term | solprop | Luna | effect |
|---|---|---|---|
| effective-radius coefficient | $3 + 10\lambda/a$ (as printed) | 20 | $n_\text{eff}$ −1.7×10⁻⁶ and $\beta_2$ −7.7 fs²/m (1.7 %) at 800 nm; area −1.0 % |
| waveguide decrement | $D_0/n_g$ | $D_0/n_g^2$ | at 30 bar, $\beta_2$ +8.9 fs²/m (1.7 %) at 1030 nm |
| effective-area prefactor | $1/n_g$ | none | at 30 bar, area −0.8 % |
| normalised frequency | $\sqrt{n_t^2 - n_g^2}$ | $\sqrt{n_t^2 - 1}$ | at 30 bar, the first loss peak 2.5 nm shorter; loss ×0.74 at 400 nm |
| dispersion width $\gamma_d$ | 0.03 | 0.03 | the same (the paper prints 0.003; 0.03 fits finite-element solutions better) |

The loss is $c\alpha/(2\omega)$ (Luna's `imag(neff)` carries twice the loss in its
gradient operator), and the silica index is sampled statelessly (Luna's spline bracket
goes stale below 706 nm, changing its 450 nm loss by about 0.7 %).

**Step-index fibres.** The TE and TM characteristic equations are the right way round
(Luna's are swapped); Bessel poles are never reported as roots (Luna returns
$n_\text{eff} = 1.45749$ for both TE₀₁ and TM₀₁ of a standard test fibre); the profile
wavelength is declared (Luna freezes it at 1030 nm); the integration radius follows the mode's tail
(Luna uses a fixed $10a$); and the TE and TM fields are the correct limiting fields (at
$n = 0$ Luna's hybrid construction returns the TM field for the TE mode).

**Rectangular and planar guides.** The wall contrast is the filled-core one,
$\sqrt{\varepsilon_\text{cl} - \varepsilon_\text{co}}$, and the TM-like coefficient
carries the permittivity ratio $\varepsilon_\text{cl}/(\varepsilon_\text{co} s)$; Luna
and the printed paper subtract one and write $\varepsilon_\text{cl}/s$, which is the
same for an evacuated core (up to 0.55 % of the loss for a filled one). A constant
geometry is reported as constant along z (Luna re-evaluates the fields at every step).

**Statistics and processing.** The peak intensity is the true on-axis value of the mode
profile (Luna's mode-averaged run reports $P/A_\text{eff}$, 1.77 times lower for HE₁₁).
The windowed mode-summed peak power is the peak of the sum (Luna's reduction stores a
row of per-mode powers). The first-order coherence $g_{12}$ excludes the $i = j$ terms,
so it tends to zero for independent fields (Luna's tends to $1/N$). The ellipse
orientation is $\tfrac12\operatorname{atan2}(U, Q)$ (Luna's is always zero). The
zero-dispersion search is stateless and raises when there is no zero (Luna returns a
missing value and seeds the next search from it). `lambda_rdw` returns the longest root
and raises when there is none.

**Stepper.** `atol` is additive to the scaled error (Luna's weak norm uses it as a floor
that never engages); a step below `dz_min` raises rather than being accepted silently;
the last step lands exactly on the fibre end.

**Noise.** Both codes carry shot noise by default, but with different models: Luna
defaults to its modified model and solprop to the input model, which is the same
calculation wherever the linear operator is unitary and the less uncertain one where it
is not ([shot noise](../media/noise.md#why-the-modified-model-is-not-the-default)). The
shot noise is confined to the active band on both field and envelope
grids (Luna's field-grid noise sits in the zero-frequency bin and above the window);
the `"modified"` noise model is advanced by the unitary part of the linear propagator
along z, not held constant; the generator is seeded explicitly and the seed recorded;
one draw has the same shape for single-mode and multimode runs.

**Pulses.** A measured pulse (`DataPulse`, `TimeDataPulse`) keeps its own time origin; a continuous wave
(`CWPulse`) is built in one pass on either grid and is seeded; envelopes are validated.

**Launching a beam into a guide.** Luna's `GaussBeamPulse` takes one scalar overlap of
the beam with each mode at the central wavelength and gives that mode the fraction
$|c_m|^2$ of the pulse's energy, so the captured share is the same at every frequency.
solprop projects the beam at **every** frequency bin, coherently and in the paraxial
measure of the guide's core index, after giving the beam its requested energy over its
whole incident plane: the modal shares follow the frequency dependence of the overlap,
and the launch is the projected amplitudes as they are. The energy the modes do not
capture is not restored by renormalising to the requested energy (nor is it in Luna),
and the coupling report states the incident, captured and uncaptured energies. A field taken from another run is coupled at its own
physical scale ([coupling simulations](coupling.md),
[SEP 31](../../dev/seps/archive/sep-0031-simulation-coupling.md)).

## More accurate numerics, the same physics

These do not change the model, but they change results at the level of the numerical
error.

- **The linear operator is integrated exactly** along a z-varying fibre, as a propagator
  tabulated or expanded in the medium's parameters; Luna applies it with a
  right-endpoint rule, first order in the step and invisible to the error estimate.
- **Group delay and dispersion are analytic**: $\beta_1$ from the analytic
  $dn_\text{eff}/d\omega$, $\beta_2$ by Richardson extrapolation of it (5×10⁻⁹ relative
  against symbolic results); Luna differences $\beta$ with a seven-point stencil at every
  stage (about 10⁻⁶ in $\beta_2$).
- **One reference frequency** for the co-moving frame and the grid (Luna mixes the
  caller's $\lambda_0$ and the grid's $\omega_0$).
- **Absorbing boundaries are rates**, so the absorption over a distance does not depend
  on how many steps cross it: the temporal rate is integrated inside every stage on the
  modal routes, and applied as its exact factor once per accepted step on the radial,
  Cartesian and planar routes unless planes are interpolated
  ([absorbing boundaries](../numerics/boundaries.md)). Luna multiplies by its window once
  per accepted step, so its absorption depends on `rtol` (refining the step sixteen-fold
  erodes the spectral wings fifty-fold). The temporal collar always
  has a minimum width (Luna's degenerates to a single sample when the realised time
  window equals the requested one, as it does for a power-of-two `trange`).
- **Out-of-band bins**: the nonlinear prefactor is exactly zero outside the active band
  (Luna leaves raw polarisation there, which every stage and error estimate sees), and
  the absorber acts on the whole axis (Luna's operator is zero outside it).
- **Loss limits** are derived from the fibre length, without empirical clamps.
- **Grids**: the coarse grid size divides the fine one on 2-3-5-smooth grids (Luna
  rounds to a power of two); an envelope grid's span is measured on both sides of the
  carrier (Luna's can drop the low-frequency half); the analytic signal keeps its
  Nyquist bin and does not double the zero-frequency bin.
- **Ionisation rate tables** are uniform in $\log E$ and refined to a tolerance (Luna's
  $2^{16}$ uniform samples reach about 1 %); above the table the last value is held
  (Luna raises mid-propagation).
- **Raman**: the kernel's tail is tapered over the grid's own collar width (Luna uses a
  fixed 70 % of the window); an intermediate-broadening line set, such as the
  Hollenbeck–Cantrell silica lines, is normalised analytically (Faddeeva function).
- **Transverse quadrature**: a fixed half-step-offset azimuthal rule (adaptive rules with
  on-grid nodes can converge on a wrong value for products such as $E_x E_y$).
- **Energies** use the exact discrete Parseval sum (Luna's extended Simpson weights give
  time- and frequency-domain energies that disagree); the half-maximum crossing is
  interpolated in a numerically stable form.

## Output and workflow

- Statistics are recorded at saved planes, with an optional periodic subset between
  them, not at every accepted step (which costs Luna about a fifth of a step); each has
  one definition.
- A file records the configuration, package versions and git commit, not a copy of the
  source; a resume compares the whole recorded configuration path by path, and a
  refused resume names what changed (Luna's key covers only the statistic names and the state shape).
  The step schedule and history are stored. The file is opened once and, as Luna's
  is, preallocated for the planned planes with an `n_saved` count; it also grows by
  doubling past them and is trimmed to the count when closed. Compression is gzip or
  lzf, not Blosc.
- Processing: the oversampled time axis is centred; the spectral phase has no linear
  ramp removed; spectral windows are explicit keywords; a bandpass edge defaults to the
  narrower of Luna's 64 bins and a quarter of the pass band, so a narrow band on a short
  window is not swamped by its edges; `specres` does not rewrite non-positive values; a
  modal sum of complex amplitudes is refused.
- Plotting: one decibel reference, from the modal sum, for every map of a run; line
  styles repeat past the sixth mode; the spectrogram names its mode on multimode runs.
- Scan subsets keep their global point numbers.

## Capabilities beyond Luna

- **Radial rules for expensive responses**: Gauss and Gauss–Radau rules in the HE₁₁
  field scale, which integrate the linear and Kerr source exactly from two nodes and
  place the atoms of an expensive response where the ionisation is
  ([radial nodes](../waveguides/radial-quadrature.md)), and the plasma statistics on the axis
  beside the node maxima.
- The general modal route for guides that supply their own vector **E** and **H** fields:
  step-index and graded-index fibres, split degenerate pairs, and guides that vary
  along z.
- The coherent capillary wall that couples the `HE_1m` modes, and the taper connection
  that converts between them.
- Tabulated modes from any mode solver.
- Planar guides, including a planar guide resolved across its width, and a line
  geometry of free space.
- Gas mixtures and steady-flow fills.
- Vector Raman responses.
- A frequency-dependent electronic response (opt-in, on every backend): optmat's
  whole hyperpolarisability fit evaluated at every quartet's frequencies, where Luna
  uses one constant $\gamma_3$ per gas
  ([a frequency-dependent electronic response](../media/dispersive-kerr.md)).
- A cycle-averaged envelope plasma model.
- True-power modal amplitudes, with analytic Poynting-flux statistics, and physical
  medium flux in free space, with forward amplitude transport through a certified radial
  pressure gradient ([energy and flux in a medium](../free-space/flux.md)).
- Grating and prism compressors.
- Coupling one simulation's output field into another (a capillary, a fibre, a gas gap) through coherent field planes, with transport through free space, lenses, apertures, mirrors and spectral optics ([coupling simulations](coupling.md)).
- Design tools with multi-soliton content, and the inverse design of a capillary's
  pressure and energy for a soliton order, count or largest eigenvalue.
- Scans on process pools and Slurm clusters.
- The native backend for multi-core CPUs and GPUs.

## Validation against Luna

Where the physics is the same, solprop reproduces Luna:

- Luna's gas refractive-index expressions, supplied to solprop's formula types as a
  caller's own laws, agree with Luna's evaluation to 10⁻¹¹;
- the Marcatili mode derivatives reproduce the symbolic reference numbers of Luna's test
  suite to 10⁻¹³, and the effective area Luna stores exactly;
- the intermodal Kerr overlaps of the `HE_1m` modes agree to better than 10⁻¹¹, and the
  two-mode Kerr coupling of both codes is 0.3995 against a closed form of 0.3987;
- the normalisation of the Hollenbeck–Cantrell silica line set, supplied as a
  caller's own, reproduces Luna's to 10⁻¹¹, and the chirped-mirror
  transfer functions Luna's own;
- the helium resonant dispersive-wave case (125 µm, 0.8 bar to vacuum, 7.5 fs at
  800 nm), run with Luna's media, Luna's PPT rate times a barrier-suppression factor, and
  a floored vacuum end: on its reduced form (two modes, 1 m) the fundamental's in-band
  content is 1.028 of Luna's and the second mode's energy 1.107, most of the remaining
  shape difference being that factor, which Luna's own rate lacks. On the full
  four-mode, 1.5 m case Luna's second-mode energy falls by a factor of 1.76 as its
  tolerance tightens from $10^{-6}$, its fixed default, to $10^{-9}$, while solprop's
  moves by parts in a million, so that difference is Luna's to converge
  ([accuracy](../../dev/accuracy.md#5-milestone-m2-the-rdw-example-four-modes-15-m-gradient));
- polarised Kerr and radial free-space runs started from the same fields and media
  agree with Luna's to about 1 % in pulse shape and on-axis field; that residual has not
  been attributed, so these comparisons are diagnostics rather than tests.

To make a solprop run comparable with a Luna one, give it Luna's media (the cladding
and the gas reference states), `vacuum_end="floor"` with `pressure_floor=0.0` for a
gradient to vacuum, Luna's rate as a rate of your own
([ionisation](../media/ionisation.md#a-rate-of-your-own)), the Raman choice explicitly, and the same tolerances; and scale energies by the modal index weight
above.

## Data taken from Luna

The seven chirped-mirror data sets (`PC70`, `HD59`, `PC147`, `PC1611`, `PC1821`,
`HD120` and `ThorlabsUMC`) are Luna's files, shipped with every number unchanged (a
provenance header added and line endings normalised) under Luna's MIT licence (the
notice is in `solprop/physdata/data/mirrors/NOTICE`); Luna records no vendor, part
number or citation for them.
