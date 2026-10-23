(primer-synthesis)=
# 14. Putting it together

This last chapter steps back. It first retells the whole chain of the primer on one page,
from Maxwell's equations to a converged result, so that you can see where each choice sits.
It then walks through how to choose a model for a new problem, using three typical problems
as examples. A glossary collects the terms whose meanings are easiest to confuse, and the
chapter ends with pointers to where to read next.

## 14.1 The chain on one page

1. **Maxwell's equations in a medium** (Chapter 1). Split the response into a linear part with
   its complete, causal $\epsilon(\omega)$, and a nonlinear polarisation and current. Work in SI.
   Write equations in the optics Fourier convention, and remember that solprop's internals use
   its conjugate.
2. **The unidirectional approximation** (Chapter 2). The second-order wave equation splits exactly
   into forward and backward waves. Evaluating the polarisation from the forward wave alone is the
   one approximation, with an error near $10^{-4}$ in gases. The result is the UPPE,
   $\partial_z\tilde E=ik_z\tilde E+\frac{i\omega^2}{2\varepsilon_0c^2k_z}\tilde P-\frac{\omega}{2\varepsilon_0c^2k_z}\tilde J$,
   written in a moving frame. Self-steepening and space-time focusing are already inside it.
3. **The transverse representation** (Chapters 3–6). Choose one of three: a single frozen mode with an
   effective area (the GNLSE), a set of guided modes with the response evaluated on a transverse
   quadrature and projected back, or plane waves in free space with the exact $k_z$. Each excludes the
   physics its symmetry or truncation cannot represent.
4. **The temporal representation** (Chapters 1, 2 and 11). Choose between the real field and an envelope
   about a carrier. An envelope is exact if nothing is dropped, and approximate when it drops harmonics
   or averages the cycle.
5. **The medium** (Chapters 7–10). Linear data, Kerr coefficients, Raman responses and ionisation rates
   come from optmat with their qualified ranges and uncertainties. solprop turns them into a linear
   operator, a cubic or dispersive polarisation, a causal convolution, and the Geissler currents with
   their population, and adds a one-photon noise seed.
6. **The numerics** (Chapters 11–12). Sample the field on a coarse grid and evaluate the response on a
   fine one. Absorb at the edges with collars expressed as rates. Integrate the linear operator exactly
   through $\Phi$. March the nonlinear part with an exponential Adams method or an embedded Runge–Kutta
   pair, under a global error norm and a PI controller.
7. **The checks** (Chapter 13). Refine each resolution separately, compare complex fields against
   independent references, converge the weak observables explicitly, and test the sensitivity of the
   conclusion to the uncertain physics.

## 14.2 Choosing a route

[Choosing a propagation model](../guide/setup/model-selection.md) has the full table. The questions
below are a way into it.

**What guides the light?**

- **A gas-filled hollow capillary.** This is the capillary route, `prop_capillary`, with Marcatili modes
  and the gas from optmat. Add the coherent wall when the higher modes carry energy and their coupling
  through the wall matters. A taper converts between modes automatically.
- **Another hollow guide.** Antiresonant, rectangular, planar or tabulated guides go through
  `prop_waveguide` with the guide's modes and a gas. A planar guide with a beam of finite height goes
  through `prop_planar`, which resolves $y$.
- **A solid-core fibre.** If only $\gamma$ and $\beta(\omega)$ matter, use `prop_gnlse`, on an envelope grid. If
  the modes' vector structure or frequency dependence matters, or several modes are involved, use the
  general modal route with a provider of the fibre's own fields.
- **Nothing.** A free-space beam uses a radial grid if it stays round, a Cartesian grid if it does not,
  and a line grid for a sheet beam per unit width.

**How many modes?** One, if the pulse stays in the fundamental and the response is Kerr and Raman. More
if energy can move: self-focusing at high power, plasma, a taper, a graded-index fibre. Then converge the
number of modes as a separate check.

**Field or envelope?** Use the field grid if the carrier matters: harmonics, very broad spectra, tunnelling
ionisation. Use an envelope for a narrowband problem, or for a GNLSE.

**Which responses?** Kerr is always present. Add Raman for molecular gases and silica, and ionisation
and plasma once the intensity approaches $10^{13}$–$10^{14}$ W/cm². Add a dispersive Kerr response when the
spectrum spans more than an octave and the result depends on widely separated colours. Add noise when the
physics grows from noise or you need ensemble statistics.

**Then refine.** Converge the tolerance, the window, the band, the modes, the quadrature and the spatial
grid, one at a time.

## 14.3 Three problems, thought through

**Soliton self-compression and a deep-ultraviolet dispersive wave in a helium-filled capillary.** The pulse
compresses to a few femtoseconds, its spectrum reaches the vacuum ultraviolet, and near the compression point
it ionises the gas. That calls for the capillary route on a field grid, with plasma and enough $\mathrm{HE}_{1m}$
modes to hold the energy that self-focusing and plasma defocusing push outwards. The wavelength range must
reach well below the dispersive wave, and the quadrature must be fine enough for the ionisation, which the
default rule is. If the fill is a gradient, its profile is part of the model, and its vacuum end limits the
convergence order. The dispersive wave is a weak observable, so converge its energy explicitly, and test its
sensitivity to the Kerr coefficients of its legs, which lie beyond the gas's data
(Chapter 8). The [resonant dispersive-wave lesson](../introduction/lessons/plot_01_rdw.rst) is this problem.

**Supercontinuum generation in a photonic-crystal fibre.** One mode, Kerr and silica Raman, and a spectrum
spanning an octave or more. `prop_gnlse` with the fibre's measured or computed dispersion, the default silica
bundle and the shock term is the standard model. If the mode area changes markedly across the continuum, the
general modal route carries the four-frequency overlap properly (Chapter 3). The incoherent part of the
continuum grows from noise, so add the noise seed and run an ensemble of seeds for its coherence.

**A filament in argon.** A collimated beam above the critical power self-focuses, ionises and filaments. A
radial grid with an absorber is efficient. The Marburger estimate tells you roughly where to look, and the
radial resolution and radius must be converged. Remember what the radial grid cannot show. Above a few critical
powers, a real beam may break up azimuthally, and only a Cartesian grid can tell you whether it does. The [radial
filament lesson](../introduction/lessons/plot_03_filament.rst) is this problem.

## 14.4 Glossary

Active band
: The frequencies $\omega_{\min}/2<\omega<1.1\,\omega_{\max}$ on which the operator, the prefactor, the noise and
  the error norm are evaluated. Outside it the field is zero.

Analytic signal
: The complex signal $E_a$ built from the positive frequencies of the real field, doubled; $E=\operatorname{Re}E_a$.

Collar
: A Planck-tapered absorbing region at a spectral, temporal or spatial edge of a grid, applied as a rate.

Cumulative operator ($\Phi$)
: The integral of the linear operator along $z$, whose exponential is the exact linear flow between two positions.

Effective area
: $(\int|F|^2)^2/\int|F|^4$. It preserves the single-mode cubic overlap, and nothing else.

Envelope
: $A(t)$ with $E=\operatorname{Re}\{Ae^{-i\omega_0t}\}$ in the optics convention. Exact if nothing is dropped.

Field grid, envelope grid
: The two temporal representations: a real field with a one-sided spectrum, or an envelope with a two-sided spectrum
  about the carrier.

`ffac`
: The oversampling factor of the fine grid on which responses are evaluated. Its Nyquist frequency is
  $\mathtt{ffac}\,\omega_{\max}/2$ on a field grid, and $\mathtt{ffac}\,\Omega/2$ about the carrier on an envelope grid.

Field scale
: The factor $\eta=J_0(u_{01}r/a)$ by which the $\mathrm{HE}_{11}$ field at radius $r$ is the on-axis field. Radial rules
  for expensive responses are built on it.

Frame
: The retarded time $\tau=t-\int_0^z\beta_1\,\mathrm dz'$, $t-\beta_1z$ for a constant $\beta_1$. It is a coordinate, shared by every mode of a set.

Interaction picture
: The variable from which the exact linear flow has been removed, so that only the nonlinear source drives the steps.

Mode-averaged, resolved
: A mode-averaged run evaluates the response at the effective-area intensity. A resolved run evaluates it at quadrature
  nodes across the mode.

Optics convention, internal convention
: $\tilde E=\int Ee^{+i\omega t}\mathrm dt$ at the interface and in this primer, and the conjugate FFT sign inside solprop.

Raman fraction ($f_R$)
: The envelope share of the total nonlinear index carried by the delayed response. The density times the per-particle kernel's integral is
  $\tfrac32f_R\chi^{(3)}_{\rm total}$.

True-power amplitude
: $b=\sqrt{\operatorname{Re}n_{\rm eff}}\,a$, whose spectral energy is the mode's (and whose square is the power for a narrow band or constant index), as opposed to the vacuum-normalised internal
  amplitude $a$.

Unidirectional approximation
: Evaluating the nonlinear polarisation from the forward field alone. Once the scalar reduction is
  made and the reference medium is uniform, it is the only approximation between Maxwell's
  equations and the UPPE.

## 14.5 Where to go next

- To **run something**: the [introduction](../introduction/index.md) and its four lessons, each a complete calculation
  with its interpretation.
- To **set up your own problem**: [setting up a simulation](../guide/setup/index.md), then the chapter of the user
  guide for your geometry.
- For the **precise statements** behind any chapter of this primer: the links at the end of each chapter, and the user
  guide's [numerics](../guide/numerics/index.md) and [media](../guide/media/index.md) chapters.
- For **processes and experiments**: the [examples](../examples/index.md).
- For the **material data**: optmat's [documentation](https://optmat.readthedocs.io).
- For **design and verification**: the [developer's guide](../dev/index.md), its [algorithms](../dev/algorithms.md) and
  [accuracy records](../dev/accuracy.md).
- For the **papers**: the [annotated references](references.md).
