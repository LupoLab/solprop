# Polarisation-dependent Raman response

Molecular orientation makes the delayed response depend on polarisation. A scalar response can describe a fixed linear polarisation, but a changing ellipse or a delayed orthogonal probe needs the tensor response. This chapter gives the selected orientation average, its limits and the corresponding field and envelope forms.

## The vector response

A scalar response is enough while the field keeps one fixed linear polarisation. When
the polarisation is elliptical, rotates during the pulse, or a probe is polarised
differently from its pump, the delayed response is a tensor. This part derives it,
relates it to other formulations, and says how to declare and run it.

## The orientation average and the selected approximation

The rotational response acts through the orientation $\mathbf n$ in the polarisability
tensor of the [physical origin](raman.md#physical-origin). For isotropic equilibrium
orientations,

$$
\langle n_i n_j\rangle=\frac{\delta_{ij}}3,\qquad
\langle n_i n_j n_k n_l\rangle=
\frac{\delta_{ij}\delta_{kl}+\delta_{ik}\delta_{jl}
+\delta_{il}\delta_{jk}}{15}.
$$

Subtracting the product of the means gives the connected orientation response

$$
C_{ijkl}=\frac{\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk}}{15}
-\frac{2\delta_{ij}\delta_{kl}}{45}.
$$

Its $xxxx$ component is $4/45$. Normalising by this component makes the
scalar line strength the response to a linearly polarised field:
$T_{ijkl}=(45/4)C_{ijkl}$. On a symmetric quadratic drive $Q$,
$T_{ijkl}Q_{kl}=(3Q_{ij}-\delta_{ij}\operatorname{tr}Q)/2$.
These averages are over a **three-dimensional** molecular orientation sphere.
Propagating only two transverse field components does not change the factor
three to two; embed the transverse $Q$ in three dimensions with zero z entries.

The vibrational model retains only the response of the mean polarisability, which
gives an isotropic (trace) channel. Real diatomic vibrational Raman scattering can also
have anisotropic contributions; those need further strengths and resolved tensor data,
and are absent from the defaults. CH4 and SF6 retain only their totally symmetric bands.
This isotropic vibration and the weak (first-order) rotational response are the model
solprop implements.

## Cartesian tensor and envelope conventions

Let $h_v$ be the sum of isotropic scalar kernels and $h_r$ the sum of rank-two
scalar kernels, each damped and weighted by its declared model. Before mixture
density weighting the local response is

$$
\boxed{\mathbf P_R(t)=\varepsilon_0\rho\left\{
[h_v*\operatorname{tr}Q](t)\mathbf F(t)
+\frac12[h_r*(3Q-\operatorname{tr}Q\,I)](t)\mathbf F(t)\right\}.}
$$

The star is a causal history integral. Its argument is the quadratic field at
the **earlier** time; the final $\mathbf F$ is the field at the observation time.

| Representation | Final field $\mathbf F$ | History $Q$ |
|---|---|---|
| Carrier-resolved, `kind="field", thg=True` | real $\mathbf E$ | $\mathbf E\mathbf E^T$ |
| Beat removed, `kind="field", thg=False` | real $\mathbf E$ | $\operatorname{Re}(\mathbf E_a\mathbf E_a^\dagger)/2$ |
| Analytic envelope, `kind="envelope"` | complex $\mathbf A$ | $\operatorname{Re}(\mathbf A\mathbf A^\dagger)/2$ |

For $\mathbf E=\operatorname{Re}[\mathbf A\exp(i\omega_0t)]$, averaging over the
carrier gives $\langle E_iE_j\rangle=\operatorname{Re}(A_iA_j^*)/2$.
This is the entire cycle-average factor. Do not halve the kernel or final
polarisation again. On the beat-removed field route the analytic signal is
used only for the history; the returned polarisation remains real. On an
envelope grid `thg` does not change the Raman drive.

The accessor `response.impulse(delays, density)` returns the scalar,
linear-polarisation kernel $h_v+h_r$;
`response.tensor_impulse(delays, density)` returns two rows in the order
**isotropic, rank-two**. Neither accessor includes the final $\varepsilon_0\rho$.
For both, density enters each model's dephasing law. The response evaluator
takes local V/m fields, not modal amplitudes in square root watts.

## Scalar reduction, rotations and circular coordinates

With $\mathbf F=\mathbf u f$ for a fixed real unit vector $\mathbf u$, the
contraction is parallel to $\mathbf u$ and reduces to the scalar
convolution with $h_v+h_r$. This holds at any real linear-polarisation angle.
Under a fixed real rotation $O$, $Q\mapsto OQO^T$ and
$\mathbf P\mapsto O\mathbf P$. This covariance is checked independently of
the backend FFT implementation.

Choose circular columns $\mathbf e_\pm=(\mathbf e_x\pm i\mathbf e_y)/\sqrt2$
and write $\mathbf A=\mathbf e_+a_++\mathbf e_-a_-$. The equivalent envelope
response is

$$
\frac{P_\pm}{\varepsilon_0\rho}=
\left[\left(\frac{h_v}{2}+\frac{h_r}{8}\right)*
(|a_+|^2+|a_-|^2)\right]a_\pm
+\frac34[h_r*(a_\pm a_\mp^*)]a_\mp.
$$

The last term stores delayed coherence between helicities. It cannot be
replaced by a pointwise intensity coefficient when the relative phase or
polarisation varies during a pulse. When exactly one helicity is occupied it
vanishes. At equal total power, a fixed circular envelope has one quarter of
the rotational source of a fixed linear envelope and the same isotropic
vibrational source. This comparison is a source reduction, **not a universal
circular-to-linear Raman gain ratio**. Pump and Stokes can occupy different
helicities and drive the missing coherence channel.

An x-polarised pump followed by a nonoverlapping y-polarised weak probe gives
$-1/2$ of the rotational index response seen by an x probe at that delay; its
isotropic vibrational index is unchanged. An oblique pump also writes an
off-diagonal history: an x probe can then acquire a y-polarised source even
though the pump is already gone. Negative diagonal and off-diagonal tensor
channels are physical and must not be clipped.

## Relation to Chen and Wise

The response follows the deterministic vector theory of
[Chen and Wise, APL Photonics 9, 030902 (2024)](https://doi.org/10.1063/5.0189749),
with solprop's sourced scalar data and local-field units. It is useful to
separate the paper's two kernel normalisations. Write its Eq. (1) monospace
kernels as $\mathcal R_a,\mathcal R_b$, and its Eq. (31) kernels as $R_a,R_b$.
At the same molecular number density,

$$
R_{\rm vib}=\varepsilon_0\rho h_v/2,\qquad
R_{\rm rot}=\varepsilon_0\rho h_r/8,
$$

$$
\mathcal R_a/2=R_a=R_{\rm vib}-2R_{\rm rot},\qquad
\mathcal R_b/2=R_b=6R_{\rm rot}.
$$

Substitution into Eqs. (32a,b) gives the circular and Cartesian expressions
above, including delayed off-diagonal coherence. In Eq. (31b), the equilibrium
population is per magnetic sublevel; compare it with $p_J/(2J+1)$ in the
rotational ladder, not the total level population $p_J$.

Their propagation amplitudes are in square root watts and their single-mode
coefficient is $\kappa=1/(\varepsilon_0^2 n^2c^2A_{\rm eff})$. For a real
profile normalised by $\int |f|^2d^2x=1$, the local electric field is
$\mathbf F=\sqrt{2/(n\varepsilon_0 c)}f\mathbf A_{\rm modal}$.
Projection contributes $\int|f|^4d^2x=1/A_{\rm eff}$.
Combining synthesis, projection and the UPPE polarisation prefactor recovers
the paper's coefficient. Solprop already performs these steps. Multiplying
its response by $\kappa$ again would count the mode area and field conversion
twice. Fourier signs must also be converted: saved spectra use the optics
convention, while convolution spectra use the internal FFT convention.

The paper's supplement makes the approximations explicit: S112–116 discuss
the equilibrium vibrational orientation average; S142–149 develop the
perturbative rotational density matrix; S173–180 show the circular and
Cartesian contractions. The displayed S139–141 appear to omit the square
on the polarisability derivative that follows from S136 and S138. The
induced-dipole derivation and dimensional check require that square, retained
here.

The paper also treats electronic Kerr and stochastic Raman sources. This
implementation adds its deterministic delayed tensor only. The Kerr response
and [noise](noise.md) keep their own conventions; no thermal
Langevin force, new spontaneous-Raman spectrum or altered noise temperature
is inferred from this deterministic response.

## Relationship to other Raman models

The response models themselves — the damped oscillator, the vibrational mode, the
rotational ladder and fused silica's scalar and tensor fits — are defined once, in
[Raman response models](raman-models.md); this section relates the tensor response to
other formulations.

| Model | Connection and required assumptions |
|---|---|
| Scalar convolution / GNLSE | Fix a real polarisation direction and use $h_v+h_r$. A normalised empirical $h_R$ additionally separates strength from shape. The GNLSE also assumes an envelope and its chosen dispersion/nonlinear expansion. |
| Driven damped oscillator | For $h=K e^{-\gamma t}\sin\Omega t$, its convolution $x=h*D$ solves $\ddot x+2\gamma\dot x+(\Omega^2+\gamma^2)x=K\Omega D$ with zero initial coherence. The $\gamma^2$ term is needed for this exact sampled kernel; dropping it is a weak-damping approximation. |
| Weak-excitation density matrix | First-order off-diagonal coherences driven from fixed equilibrium populations produce the same causal line sums. This does not include population depletion, saturation, strong alignment, redistribution between rotational states or coherent preparation before the window. |
| Pump–Stokes gain equations | Linearise about a prescribed pump, resolve the relevant narrow spectral components, and impose the appropriate steady-state, phase-matching and depletion assumptions. Keep the conjugate anti-Stokes component unless a justified mismatch or scale separation removes it. |
| Impulsive alignment models | The weak-response limit shares the rank-two orientation tensor. Nonperturbative rotational wave packets and repeated strong kicks need evolving populations/coherences beyond this fixed-kernel model. |
| Solid-state vector Raman | A scalar glass fit does not determine its depolarisation tensor. Hollenbeck–Cantrell silica remains scalar-only unless a caller explicitly supplies a justified tensor declaration; the separate Lin–Agrawal tensor fit is [fused silica, as a vector response](raman-models.md#fused-silica-as-a-vector-response). |

For a monochromatic component in the internal convention, the continuous
oscillator transfer function is

$$
H(\omega)=\frac{K\Omega}{(\gamma+i\omega)^2+\Omega^2}.
$$

This fixes the gain sign and provides an independent check of a sufficiently
long, well-resolved sampled response. A small-signal **amplitude** growth rate
is half its intensity growth rate. Chen–Wise Eq. (36) additionally assumes the
specified pump/Stokes polarisation channels and reduced propagation system;
it cannot be applied to an arbitrary broadband elliptical pulse by inserting
the fixed-circular factor $1/4$. Their supplement S194–200 keeps the Stokes and
conjugate anti-Stokes eigenproblem before taking limiting gain expressions.

## Declaring and using tensor terms

The public constructor has one authoritative input route:

```{code-block} python
RamanResponse(models=None, fraction=None, thg=True, species=None, *, tensor_terms=None)
```

Supply exactly one nonempty tuple. `models=(...)` gives scalar behaviour and records
each term as `scalar_only`. `tensor_terms=(RamanTensorTerm(model, contraction), ...)`
accepts the labels `scalar_only`, `isotropic` and `rank2`. The frozen response
normalises either route into one immutable term tuple, and `.models` is a view derived
from it. When using `dataclasses.replace`, replace `tensor_terms` to change a model or
its assignment.

The material routes attach labels without inferring them from the oscillator class. A
caller's {class}`~solprop.physdata.raman_types.MolecularRaman` becomes a `rank2`
rotational ladder and an `isotropic` vibration (the retained band is the mean
vibrational response; that does not assert that a diatomic molecule's whole vibrational
polarisability is isotropic). A built-in gas's optmat response labels each damping
group by its tensor family ([Raman data](raman-data.md#the-same-convention-mapped-onto-terms)). The scalar
Hollenbeck–Cantrell silica shape has no assignment: a scalar empirical fit does not
determine vector depolarisation (silica's vector response is the separate tensor model of
[fused silica, as a vector response](raman-models.md#fused-silica-as-a-vector-response)).

For example, a caller with a justified isotropic oscillator can declare it
explicitly. The numerical values below are a synthetic example, not a material
fit:

```{code-block} python
import numpy as np
from solprop.nonlinear.raman import (
    DampedOscillator,
    RamanResponse,
    RamanTensorTerm,
    undamped_dephasing,
)

line = DampedOscillator(1e-38, 2e14, undamped_dephasing())
custom = RamanResponse(tensor_terms=(RamanTensorTerm(line, "isotropic"),))
time = np.arange(512) * 1e-15
pulse = 1e8 * np.exp(-(((time - 100e-15) / 20e-15) ** 2))
linear = np.array([pulse, np.zeros_like(pulse)], dtype=complex)
elliptical = np.array([pulse, 0.4j * pulse]) / np.sqrt(1 + 0.4**2)
circular = np.array([pulse, 1j * pulse]) / np.sqrt(2)
polarisation = custom.evaluate(elliptical, 1e25, kind="envelope", dt=1e-15)
```

Replace the declaration with
`response_for("H2", selection=MaterialSelection(carrier_wavelength=800e-9))` to include
optmat's isotropic and rank-two terms. Add a delayed weak probe to either Cartesian
component before evaluation to retain its full history. The deterministic
[vector Raman gallery](../../auto_tutorials/capillaries/plot_06_vector_raman) compares linear
and circular pumps and an oblique-pump, cross-polarised delayed probe.

A launch Jones vector sets the field amplitudes in the **real modal basis**. It does not make the modal profiles complex. The two supported modal
representations are a state polarisation axis or separate real polarisation
rows; both synthesise the same local Cartesian field before evaluating Raman.
Higher HE and TE/TM profiles can vary direction across the transverse plane.
Applying a scalar response to each modal amplitude separately would lose their
local cross terms. See [polarisation](../setup/polarisation.md) for launch conventions.

Metadata preserves one declaration: every response records its labelled
`tensor_terms` (an all-scalar one labels each `scalar_only`), its fraction, `thg` and
species, however it was declared. Labels take part in configuration hashes and resume
compatibility. A metadata record describes the model and is not an
instruction to execute arbitrary Python objects when loading an output file.

## Vector convolution and resolution guidance

The readable physics evaluator contracts $Q_{xx},Q_{yy},Q_{xy}$ directly.
Backends use the equivalent layout

$$
S=Q_{xx}+Q_{yy},\quad D=Q_{xx}-Q_{yy},\quad C=2Q_{xy},
\qquad K_S=h_v+h_r/4,\quad K_T=3h_r/4.
$$

Three causal convolutions give $U=K_S*S$, $V=K_T*D$, $W=K_T*C$, and

$$
P_x=\varepsilon_0\rho[(U+V)F_x+WF_y],\qquad
P_y=\varepsilon_0\rho[WF_x+(U-V)F_y].
$$

A scalar response keeps one history and the scalar summation order. An
isotropic-only vector needs just the trace history. Rank-two vectors need
three histories and two effective kernels. Every physical term is damped
before these kernel combinations are formed. In mixtures, each species'
amplitude density and response fraction are applied before adding its
contribution; the total-density approximation is used for linewidths.

The compiled leaf carries numeric weights `(1, 0)` for isotropic terms and
`(1/4, 3/4)` for rank-two terms, plus an explicit capability flag for scalar-only
declarations. Immutable problem ownership fixes roles, weights, fractions and
the grid. Runtime cache validity includes all relevant decay rates and mixture
amplitude densities. A sole component's density remains outside the convolution.

Real FFTs use doubled zero padding and one explicit sample spacing. NumPy's
normalised inverse and the compiled providers' explicit inverse scaling produce
the same causal sum. Full and partial node chunks have their own transform
plans at their actual widths and share persistent buffer storage. Per node,
each history requires two real arrays of length $2N$ and one complex spectrum
of length $N+1$, or $48N+16$ bytes in float64/complex128. Multiply this by one
or three for the Raman node workspace; add the field/projection and
plasma buffers. Kernel spectra and provider plan storage are separate entries
in `memory_ledger()` and are not multiplied by the node count.

Resolution needs several independent checks:

1. Resolve the shortest pulse feature and the highest included Raman frequency.
   A carrier-resolved run must also resolve the optical carrier and every
   retained harmonic. Halve the time spacing while holding the physical window
   fixed; three levels are needed before claiming an observed convergence order.
2. Extend the window without changing the time spacing. Include all launch
   delays and walk-off, and keep relevant history lags before the kernel taper.
   A long-pulse gain spectrum may require a window comparable to several decay
   times even when a short impulsive calculation does not.
3. Refine transverse quadrature separately. Real-rotation covariance of the
   local tensor does not cure an under-resolved modal angular integral.
4. Refine propagation steps or tolerances independently of kernel sampling.
   A backend agreement test fixes numerical implementation; it does not prove
   that a coarse grid represents the intended experiment.
5. Keep material uncertainty separate from numerical error. An accurate FFT
   cannot resolve missing rovibrational lines, spin structure or collision data.

The independent checks use sphere quadrature for the rank-four tensor, explicit
causal sums with absolute-summand bounds, and driven-oscillator ODEs. Positive
FFT-error allowances are added before comparing transforms. Carrier averaging
has its own 1% target in controlled narrowband studies; it is not folded into
rounding tolerances. Simplified small-signal tests retain the Stokes/conjugate
anti-Stokes system before projection. Conservation or Manley–Rowe identities
apply only after stating the simplified lossless, mode and sideband assumptions;
no conservation claim here applies to arbitrary lossy, ionising broadband runs.

One useful restricted balance can be derived directly. Take a single spatial
mode with real, frequency-independent normalisation, positive absolute optical
frequencies, no loss, ionisation, radiated harmonics or absorber, and the envelope
equation $\partial_z\widetilde{A}=-i\kappa\omega\mathcal{F}(M A)$, with real
constant $\kappa$. The delayed response matrix $M(t)$ is real and symmetric.
For $N=\int |\widetilde{A}(\omega)|^2/\omega\,d\omega$, differentiation and
Parseval give

$$
\partial_z N=2\kappa\operatorname{Im}\int
 \widetilde{A}^{\dagger}\mathcal{F}(M A)\,d\omega
 \;\propto\;\operatorname{Im}\int A^{\dagger} M A\,dt=0.
$$

Up to fixed Fourier-normalisation and Planck-constant factors, N is the photon
number. This is a balance of that restricted envelope system. Optical
energy has a different spectral weight and need not be constant as energy is
transferred to molecular excitation. It is not a separate undepleted-pump
conservation law: removing pump depletion removes part of the balance. The
vector source tests verify this infinitesimal balance with an undamped synthetic
response, positive sampled frequencies and explicit FFT/reduction error bounds.
Zero and sampled delta kernels independently check the causal convolution and
its instantaneous tensor limit; a delta kernel is a numerical limiting case,
not a molecular oscillator model.
