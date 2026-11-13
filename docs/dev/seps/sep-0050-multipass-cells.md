# SEP 50 — Multi-pass cells assembled from free-space passes

| | |
|---|---|
| **SEP** | 50 |
| **Title** | Multi-pass cells assembled from free-space passes |
| **Author** | Matthieu Guer |
| **Drafting agent** | Claude Opus 5.5 (2026-11-12) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
| **Type** | Feature |
| **Created** | 2026-11-12 |
| **Resolution** | — |
| **Requires** | SEP 22, SEP 31, SEP 34, SEP 40, SEP 47 |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

A multi-pass cell (MPC) folds a long nonlinear path between two concave mirrors: a pulse
crosses a gas fill, or one or more bulk plates, tens of times, and is refocused at every
bounce. This SEP proposes to simulate gas-filled and solid-state MPCs in solprop without
a new propagation route and without a change to the seam. A cell is described once, in
the physics layer, together with its linear and Kerr-corrected design: the eigenmode of
the cell, the beam sizes at the mirrors and at the centre, the critical-power ratio and
the B-integral per pass, after Hanna et al. [4]. A driver-level executor then runs the
cell as an ordered sequence of ordinary free-space runs, one per nonlinear segment and
pass, joined at each mirror by the linear elements of SEP 31, in an unfolded
optical-path coordinate. Each nonlinear segment keeps one translated problem and one
backend for every pass (`SimulationRunner`, SEP 22). The cell owns the chain's energy
ledger, its noise identity and its resume, and a scan accepts a builder of cells as it
accepts a builder of simulations. With plates the Kerr correction of [4], derived for a
fill, no longer holds as published; the design keeps its moment form segment by segment
and states the closure it then rests on (section 4.4). Success is judged by the
short-segment checks listed in [#191](https://github.com/LupoLab/solprop/issues/191) as
tests, by the design reproducing the published numbers of [4], [10], [12] and [13], and by
worked examples that reproduce the beam-size evolution of [4], the spectra and compressed
durations of the four argon cells of [10], and the spectra of a solid-state cell
[12, 13, 14] within stated tolerances.

### 2. Motivation and scope

Gas-filled MPCs are the standard post-compression stage for millijoule ytterbium lasers:
pulses of 300–700 fs are broadened by self-phase modulation over 15–30 passes in argon
and compressed to below 20 fs with transmissions above 90 % [9, 10]. Bulk MPCs do the same
with fused silica inside the cell, first the mirror substrates themselves [2], then plates
in air or in vacuum [12, 13, 15], up to 30 mJ [14]. Their design is a compromise
between the B-integral per pass, the peak power against the critical power, the fluence
on the mirrors, ionisation at the focus and the spatio-spectral homogeneity of the output,
and the beam size from pass to pass depends on the Kerr lens: launched on the linear
eigenmode, a cell near the critical power drifts by tens of per cent in mirror fluence
[4, 10]. Analytic and one-dimensional models size a cell; only a spatio-temporal
propagation over all passes shows the drift, the aberrations of the Kerr lens, the
homogeneity of the broadened spectrum and the compressibility of the output.

solprop already propagates the field of a beam in a bulk medium with the exact
longitudinal operator, Kerr, plasma and Raman responses, on radial and Cartesian grids,
and SEP 31 supplies field planes, thin lenses, mirrors, interfaces, spectral optics and
the handoff of a plane into another run. Nothing assembles them into a cell
([#191](https://github.com/LupoLab/solprop/issues/191)).

**In scope.**

- A two-mirror cell with spherical mirrors of the same radius of curvature $R$ and
  separation $L$, $0 < L < 2R$, simulated in its unfolded, on-axis equivalent on a
  `RadialGrid`; the Cartesian grid admitted for the same symmetric cell, as a check. The
  two mirrors may carry different coatings, as a complementary pair of chirped mirrors
  does [15].
- A gas fill at constant density, or an ambient medium (vacuum, a gas, air) with one or
  more plates of a bulk medium at normal incidence, each at a declared position along the
  pass. A mirror crossed in its own substrate, as in [2], is a plate on each side of the
  mirror, with the net power of the meniscus, $2/R$ in the thin limit, given to the mirror.
- Mirror coatings as frequency-only responses: reflectance, group-delay dispersion or a
  tabulated phase, through the existing optics models.
- The linear and Kerr-corrected design of the cell (sections 4.3 and 4.4) and a launch
  matched to its eigenmode at the first mirror.
- The executor, its ledger, its resume at pass granularity, and per-pass observables:
  energy, peak intensity at the centre, beam radius at the mirrors and at the centre,
  spectral width, Fourier-limited duration and an on-axis B-integral diagnostic.
- Scans of cells: `run_scan` accepts a builder that returns a cell, so the pressure,
  the energy, the length, the chirp of the launch, the plates or the coating can be
  scanned with the existing executors, resume and collected file (section 5.5).

**Out of scope, explicitly.**

- The input and output coupling optics (entrance window, matching lenses, injection and
  extraction mirrors) and their nonlinearity. The first version starts at the first cell
  mirror; propagating the design back to the matching optics, as [4] section 4 describes,
  is follow-up work.
- The Herriott spot pattern on the mirrors, re-entrance conditions and injection angles.
  In the unfolded model they do not change the propagated field; the pattern only adds a
  fluence per spot, clipping at a mirror edge and, on a Cartesian grid, the astigmatism of
  oblique incidence. Follow-up work.
- Interaction between passes: overlap of the beams at the mirrors or the centre, residual
  plasma, heating and gas dynamics between pulses. Counter-propagating waves, as in
  [#191](https://github.com/LupoLab/solprop/issues/191).
- Brewster-angle plates, mirrors of different radii, astigmatic or off-axis mirrors,
  dispersion-engineered coating design, damage models of the mirrors.
- Mirror events inside a native march (section 10), resampling between grids of
  different size from pass to pass, and recipes and front-end hooks for cells (SEP 41).

### 3. Context

**Free space.** A free-space run evolves the physical field on the order-zero Hankel
columns of a `RadialGrid` or the Fourier columns of a `FreeGrid`, with the exact
$k_z=\sqrt{n^2\omega^2/c^2-k_\perp^2}$, the axial flux of the medium as its energy measure
([the free-space model](../../guide/free-space/model.md), SEP 34) and transverse boundaries
watched by the monitor of SEP 40. A run's medium is constant along z, except a gas whose
pressure varies on the radial grid inside the adiabatic certificate
([#193](https://github.com/LupoLab/solprop/issues/193),
[#206](https://github.com/LupoLab/solprop/issues/206)): a gap followed by a plate is two
runs, not one.

**Coupling.** SEP 31 ([coupling simulations](../../guide/workflows/coupling.md)) gives the
`FieldPlane` with its affine reference, `transport` through `ThinLens`, `FreeSpace`,
`PlanarInterface`, `PolarisationTransform`, `Mirror` and `SpectralTransfer`, each with a
report and, for a screen, a thin-screen certificate; `Simulation.input_port`, `couple`
with the `same_basis` model and `Simulation.with_input` relaunch a plane into a run on
the same lattice and basis. A spatial port admits only `same_basis`, in the same medium,
at the same position and with the same orientation, so a transported plane is declared at
the receiving entrance (`transport(..., geometry=port.geometry)`). `Mirror` turns the
plane's forward axis, which `couple` then refuses as oriented differently: a folded cell
cannot be relaunched as it stands, and section 4.1 unfolds it. Two capillaries relaunched
this way reproduce one to the tolerance's level; SEP 31 section 9 names multi-pass-cell
assembly as future work.

**Reuse and run ownership.** `SimulationRunner` (SEP 22 section 5.2) keeps one translated
problem and one backend for many inputs; its `run` takes a pulse or an array, not a
`CouplingLaunch` (an array drops the coupling record and the inherited reference), and it
refuses an input whose noise model differs from its own. SEP 47 keeps orchestration
between simulations in Python and forbids a
change of physics inside a run or a Python hook inside a native march (SEP 47 section
5.1): a mirror can only act between runs, which is what this SEP does.

**Scans.** `run_scan` ([scans](../../guide/workflows/scans.md)) takes a `Scan` and a
builder that turns each `ScanPoint` into a `Simulation`, which it checks by type; it
dispatches the points through the executors (`Local`, `Pool`, `Batch`, `Queue`, `Slurm`),
draws unseeded noise from the point's seed, keeps one backend for consecutive points that
differ only in their input, skips completed points on a rerun and collects statistics,
fields and declared reductions into one file read by `ScanOutput`. A point is one run.

**Design helpers.** `solprop.spatial.critical_power` gives the Townes baseline, its mass
1.86226 fixed in the code; `solprop.tools.beam_params`, `pulse_peak_power` and
`self_focusing_power` give the numbers of one Gaussian beam. Nothing computes a cavity
eigenmode, an ABCD propagation of the complex beam parameter or a Kerr-corrected beam.
`solprop.processing.compression.transform_limit` and `apply_transfer` accept a spectrum,
while the other compression helpers read modal outputs.

**Issues.** This SEP answers [#191](https://github.com/LupoLab/solprop/issues/191) and
follows its list of qualification checks. It bears on:

- [#175](https://github.com/LupoLab/solprop/issues/175): a `FieldPlane` does not carry the
  frame it was written in; the executor passes the frame explicitly until it does.
- [#238](https://github.com/LupoLab/solprop/issues/238): a relaunched run's boundary
  monitor starts at zero and no object owns a chain's ledger; the cell owns its own
  (section 5.3).
- [#705](https://github.com/LupoLab/solprop/issues/705): a reused runner does not compare
  execution options; the executor builds each runner from one settings object, and the
  fix should precede phase 2.
- [#218](https://github.com/LupoLab/solprop/issues/218): `SpatialPulse` has a scalar waist;
  the matched launch, whose waist varies with frequency, is built with
  `FieldPlane.from_analytic` instead.
- [#618](https://github.com/LupoLab/solprop/issues/618) and
  [#78](https://github.com/LupoLab/solprop/issues/78): the transverse judgement of a
  relaunched run and the collar strength of a sliced run, which every pass after the first
  meets; `with_input` also re-checks a spatial run's transverse design for a launch that is
  not a multiple of the original, and warns when its band exceeds the designed one.
- [#206](https://github.com/LupoLab/solprop/issues/206) and
  [#193](https://github.com/LupoLab/solprop/issues/193): why a solid-state pass is a
  sequence of runs.

Two changes merged while this SEP was drafted: the split of `simulation.py`, which moved
`SimulationRunner` into `driver/runner.py` and replaced `from_radial` by `from_free` with a
`RadialGrid`, and SEP 49, which brought the TDSE response to native and left `numpy_ref` a
complete backend, with a new feature landing on both (ADR-077, ADR-068). The proposal adds
no backend code and is written against the concepts, not the names; the implementation
follows whichever names hold when it starts.

**Literature.** The cell is Herriott's [1]. MPC compression was first demonstrated in [2]
with the mirror substrates as the nonlinear medium: at each bounce the beam crosses 13 mm
of fused silica behind a concave anti-reflection face and is reflected by the convex back
face, and a transit of 18 bounces per mirror reaches an estimated B-integral of about
$5\pi$. [3] analysed the scheme; [4] derived the stationary beam in the presence of the
Kerr lens and the effective power of a pulse, for gas-filled cells only, noting that the
formalism "could be used" for cells with plates. [9] reviews the field; its Table 1 lists
three fused-silica and three argon cells with the mirror fluence, the peak intensity and
the B-integral computed for linear mode matching. [10] reports four argon cells designed
with the corrected ABCD model and compares a spatio-temporal simulation with experiment.
Four solid-state works serve here as references: two 2 mm plates in air with pre-chirped
200 µJ pulses and the spectrum measured pass by pass [12]; a hybrid cell of two 0.5 mm
plates in air whose nonlinear mode matching is published beside the linear one [13];
24 plates of 0.8 mm in vacuum at 30 mJ, near the edge of stability [14]; and two cells of
1 mm plates in air in series [15], in which nonlinear matching reduced the mirror radius
by about 30 % with five plates near the centre and raised it by about 15 % with two plates
near the mirrors.

### 4. Theory

SI units throughout; the optics sign convention at the interfaces of
[conventions.md](../conventions.md). $\lambda_0$ is the vacuum wavelength,
$k_0=2\pi/\lambda_0$, $n_0$ and $n_2$ the linear and nonlinear indices of the fill at the
carrier, $w$ the $1/e^2$ intensity radius.

#### 4.1 The unfolded cell

Unfold the cell about each mirror: pass $j=0,1,\dots,N-1$ propagates over
$z\in[0,L]$ in its own coordinates, and the unfolded optical-path position is
$z_{\rm abs}=jL+z$. A spherical mirror of radius $R$ adds the path $2s(r)$ of its sag
$s(r)=R-\sqrt{R^2-r^2}=r^2/2R+O(r^4/R^3)$ across the beam. In the paraxial limit the
reflected field, written in the next pass's coordinates, is

$$
\tilde E_{j+1}(r,\omega,0)=r_c(\omega)\,e^{-i k(\omega) r^2/R}\,\tilde E_j(r,\omega,L),
\qquad k=\operatorname{Re}n\,\omega/c ,
$$

a thin lens of focal length $R/2$ in the ambient medium followed by the coating's
amplitude reflection coefficient $r_c(\omega)$. Pass $j$ ends on the second mirror when $j$
is even and on the first when it is odd, and takes that mirror's $r_c$; the lens is the
same for both, so a pair of coatings leaves the design unchanged. The quartic term of the
sag is $k r^4/(4R^3)$; it is below $10^{-2}$ rad for a 2.5 mm beam on a 1 m mirror at
1 µm and is dropped, with the paraxial error of the screen bounded by SEP 31's thin-screen
certificate for each field (section 4.6 of SEP 31). The real cell reflects the beam: with
`Mirror`'s convention the outgoing local axes keep the p/s components of the incoming
ones and the pattern is imaged along p. A rotation by $\pi$ about the s axis carries the
outgoing frame $(\hat p_{\rm out},\hat s,\hat k_{\rm out})$, with
$\hat k_{\rm out}=-\hat k_{\rm in}$ at normal incidence, back onto the incoming one; in the
unfolded coordinates the reflection is therefore the Jones map
$\operatorname{diag}(r_p,r_s)$ in the local axes, the image $x\to-x$ along p (an identity
for each component on a radial grid) and the screen above. At normal incidence a
dielectric coating has $r_p=-r_s$ in this convention, so a circular polarisation changes
handedness at each bounce, as it physically does; an isotropic Kerr response is
unaffected.

An off-axis Herriott path is, in the same paraxial and small-angle limit, the same
sequence of passes along the ray: the mirrors act on the beam about its local axis, with
tangential and sagittal focal lengths $R\cos\theta/2$ and $R/(2\cos\theta)$ at incidence
$\theta$. The difference is $O(\theta^2)$, a few $10^{-4}$ for the injection angles of
[10]; it is neglected here and belongs to the Cartesian follow-up. The passes are
independent: they are separated in time by $L/c$, nanoseconds, and nothing carries from
one to the next except the field.

#### 4.2 Linear design

The half round trip from just after one mirror to just after the next is
$M=\begin{pmatrix}1&L\\-2/R&1-2L/R\end{pmatrix}$ [11]. Its trace gives the Gouy phase per
pass, $\cos\psi=1-L/R$, real for $0<L<2R$. The eigenmode has its waist at the centre and
the Rayleigh range fixed by the geometry, $z_R^2=L(2R-L)/4$, whatever the frequency, so

$$
w_0^2(\omega)=\frac{\lambda_0 z_R}{\pi n_0},\qquad
w_m^2(\omega)=w_0^2\left[1+\left(\frac{L}{2z_R}\right)^2\right],
$$

and the launch matched at every frequency is the isodiffracting Gaussian whose waist grows
as $\sqrt{\lambda_0}$. Near the concentric limit $w_0\propto(2R-L)^{1/4}$: a 0.5 mm change
of $L$ on a 1 m cell moves $w_0$ by 2.4 %, which is why a reproduction must use the exact
geometry. With plates, the half round trip is the product of the free-space, interface
and mirror matrices of its segments, with reduced lengths $d/n$ [11]; the eigenmode is the
fixed point of the round-trip map of the complex beam parameter, found numerically and
checked against the closed form above for the gas cell.

#### 4.3 Kerr-corrected design

**The aberrationless beam.** Bélanger and Paré [5] showed, and [4] uses, that a continuous
Gaussian beam of power $P$ in a homogeneous Kerr medium follows
$w^2(z)=w_0^2[1+\sigma(z/z_R)^2]$ and $R(z)=z[1+(z_R/z)^2/\sigma]$, with
$z_R=\pi n_0w_0^2/\lambda_0$ and $\sigma=1-P/P_c$. It propagates through any ABCD system
with the generalised reduced parameter [4, eq. (8)]

$$
\frac{1}{\hat q}=\frac{n_0}{R}+i\sqrt\sigma\,\frac{\lambda_0}{\pi w^2},
$$

that is, as a linear beam of wavelength $\lambda_0\sqrt\sigma$.

*Re-derivation.* For the paraxial equation
$2ik\partial_zA+\nabla_\perp^2A+2k^2(n_2/n_0)|A|^2A=0$, $k=n_0k_0$, the second moment
$V=\int r^2|A|^2\,dS$ obeys $V''=(2/k^2)H$ with the conserved
$H=\int|\nabla_\perp A|^2\,dS-k^2(n_2/n_0)\int|A|^4\,dS$ [8], so $V(z)$ is exactly parabolic
in a homogeneous segment, however the profile deforms. For a Gaussian at its waist,
$\int|\nabla A|^2=2P/w_0^2$ and $\int|A|^4=P^2/(\pi w_0^2)$, so
$H=(2P/w_0^2)(1-P/P_G)$ with $P_G=\lambda_0^2/(2\pi n_0n_2)$: the law above, exact for the
second moment of a Gaussian start, with $P_c=P_G$. A wavefront curvature adds its own term
to $H$ and leaves the nonlinear one unchanged, which is the split of $1/\hat q$ into a real
geometric part and an imaginary part scaled by $\sqrt\sigma$. After a lens the profile is
no longer Gaussian and the law becomes the aberrationless approximation.

**The critical power.** In the form $P=\alpha\lambda_0^2/(4\pi n_0n_2)$ of
`critical_power`, the derivation gives $\alpha=2$; the Townes baseline is
$\alpha=1.86226$; the computed collapse threshold of a collimated Gaussian is
$\alpha\approx1.89$, written $3.77\lambda_0^2/(8\pi n_0n_2)$ [6, 7], and is the value [4]
and [10] use. They place $\sigma=0$ at the power where a Gaussian actually collapses; the
derivation places it where $H$ vanishes, slightly above. The difference is 6 % in $P/P_c$:
for the cell of [4] section 3, $\sigma$ moves from 0.595 to 0.618, $w_0$ by 0.9 % and the
B-integral by 1.9 %, against the 30 % by which aberrations raise the simulated B-integral
above the aberrationless estimate in the same paper. The default follows the published
model, $3.77/8\pi$, recorded in the design and selectable (Q1); [9] takes
$\lambda_0^2/(2\pi n_0n_2)$, the value of the derivation. `critical_power` fixes the
Townes mass today; it gains the mass as a keyword, defaulting to today's value, rather
than the design keeping a second formula.

**The nonlinear index.** The design takes $n_2$ where the run takes its Kerr coefficient,
from optmat: `solprop.physdata.n2` for a gas, at the carrier and the medium's temperature
(293.15 K by default), and the glass's Kerr model for a plate. For argon at 1030 nm optmat
0.9.1 gives $n_2=1.000\times10^{-23}$ m²/W per bar (model `hyperpolarizability/Ar`, fitted
to Hättig et al.'s coupled-cluster values and to Shelton's measurements and compilation;
fit uncertainty 0.2 %), linear in pressure to 0.06 % up to 3.5 bar. The literature used here
takes $9.7\times10^{-24}$ m²/W per bar ([4], [10]) and $7.0\times10^{-24}$ ([9], Table 1):
optmat is 3.1 % and 43 % higher. The temperature alone moves $n_2$ by 1 % per 3 K
through the density, and none of [4], [9] and [10] states it. For fused silica at
1030 nm optmat's long-pulse $n_2$ is $2.53\times10^{-20}$ m²/W, 22 % of it from the Raman
response, against $2.2\times10^{-20}$ in [14] and $2.8\times10^{-20}$ in [9]; [12] states
none. With $x=P_{\rm eff}/P_c$, $d\ln B_{\rm axis}/d\ln n_2=1+x/(2\sigma)$, 1.2 to 1.5
for the cells here. For the cell of [4] and the four cells of [10], optmat's argon changes $\sigma$
by −1.5 to −2.8 %, the matched mirror radius by −0.4 to −0.7 %, the mirror fluence by +0.8
to +1.4 % and the B-integral by +3.9 to +4.6 %; the index of [9] would lower the
B-integral by a third. The B-integral sets the broadening: with the rule of [9] that the
compression factor is about $B/\pi$, 4 % more B-integral is about 4 % off the
Fourier-limited duration, the size of the agreement between simulation and measurement in
[10] (18.5 against 19.2 fs for MPC2). The effect on the propagated results is measured,
not assumed (section 7).

**Pulses.** Each time slice of a pulse whose shape is fixed over a pass (dispersion
negligible there) is its own continuous beam with power $P(t)$. The energy-weighted mean
square radius is
$\langle r^2\rangle=\int P(t)\langle r^2\rangle_t\,dt/E=(w_0^2/2)[1+\bar\sigma z^2/z_R^2]$
with $\bar\sigma=1-P_{\rm eff}/P_c$ and

$$
P_{\rm eff}=\frac{\int P^2\,dt}{\int P\,dt},
$$

which is eq. (6) of [4], obtained there from the energy-averaged nonlinear phase. A
Gaussian gives $P_{\rm eff}=P_{\rm peak}/\sqrt2$, a sech² $2P_{\rm peak}/3$. The design
evaluates the integrals on the launched pulse rather than assuming its shape.

**Stationary beam.** Setting the wavefront radius at the mirror, $z=L/2$, equal to $R$ gives
$z_R^2/\sigma=L(2R-L)/4$, hence [4, eq. (3)]

$$
w_0^2=\frac{\lambda_0L}{2\pi n_0}\sqrt{\sigma\left(\frac{2R}{L}-1\right)}
=\sqrt\sigma\,w_{0,\rm lin}^2 ,
$$

and the same factor $\sigma^{1/4}$ on the mirror radius; $\psi$ is unchanged. The cell keeps
a stationary beam only for one $\sigma$: as the energy falls by the reflectance $R_c$ at
each bounce, $\sigma_j=1-R_c^jP_{\rm eff}/P_c$, and the design reports the resulting drift
of a launch matched for the first pass, as [10] does.

**B-integral.** With $I_0(z)=2P_{\rm peak}/(\pi w^2(z))$,

$$
B_{\rm axis}=k_0n_2\int_{-L/2}^{L/2}I_0\,dz
=\frac{8\pi n_0n_2}{\lambda_0^2}\frac{P_{\rm peak}}{\sqrt\sigma}
\arctan\sqrt{\frac{L}{2R-L}}
$$

per pass on the axis. This is the expression [4, eq. (4)] calls the beam-averaged
B-integral per round trip, the beam average of a Gaussian being half the on-axis value.
$\sigma$ uses $P_{\rm eff}$ and the prefactor $P_{\rm peak}$, as [4] specifies. The design
reports both quantities under these two names.

**The convention of [10].** [10] writes $\sigma=1-P_0/P_c$ with $P_0$ the peak power, but
its numbers were computed with $P=E/(\sqrt2\,\tau_{\rm FWHM})$ (the author of this SEP, a
co-author of [10], 2026-11-13): the effective power of a Gaussian whose peak is taken as
$E/\tau_{\rm FWHM}$, which exceeds the exact $P_{\rm eff}$ by 6.5 % for a Gaussian and by
20 % for a sech². It reproduces the fluences of the Fig. 2 of [10] (111 and 170 mJ/cm² for
its MPC3 with $M^2=1.05$; the exact $P_{\rm eff}$ gives 164 mJ/cm²), and its Table 2 to the
table's two significant figures once MPC3 is taken at 2 bar, the pressure of Fig. 2; the
1.2 bar printed in Table 2 is an error (the author, 2026-11-13). The peak power $P_0$ is
excluded by three of the four cells (191 mJ/cm² for MPC3 against 160). The design uses the
exact $P_{\rm eff}$ on the launched pulse; an explicit effective power reproduces a
publication's convention (section 5.1, Q1).

#### 4.4 Cells with plates: what survives of the Kerr correction

[4] derives its correction for a Kerr medium that fills the cell and leaves plates to
future work. With plates, one ingredient survives and four fail.

**What survives: the moment law, segment by segment.** The variance identity holds in
every homogeneous segment, gas or glass, thin or thick, and every linear element (free
space, thin lens, plane interface) maps the second moments exactly, whatever the profile
[8]. With $W=w^2$ the second-moment radius and $U=ww'$, a segment of index $n_s$ and
critical-power ratio $x_s=P_{\rm eff}/P_{c,s}$ gives

$$
W''=2\left[(1-x_s)\frac{\lambda_0^2}{\pi^2n_s^2W}+\frac{U^2}{W}\right],
$$

constant along the segment, so $W$ is a parabola in $z$ for either sign of
$\sigma_s=1-x_s$. A thin lens of focal length $f$ maps $U\to U-W/f$; a plane interface
from $n_a$ to $n_b$ keeps $W$ and scales $U$ by $n_a/n_b$. For a fill this is the $\hat q$
form of section 4.3. The closure is that the beam keeps a Gaussian profile of fixed beam
quality at the entrance of each segment, which fixes both terms of the bracket from $W$
and $U$.

**What fails.**

- *The closed form.* The stationary beam of section 4.3 assumes one $\sigma$ over the whole
  pass. With plates the eigenmode is the fixed point of the round-trip map of $(W,U)$,
  found numerically as the linear one of section 4.2, and checked against the closed form
  when the plates are removed.
- *The complex parameter $\hat q$.* A plate runs far above its own critical power. Fused
  silica has $P_c\approx4$–5 MW at 1030 nm with the $n_2$ of [13] and [14] and the constant
  of [6], and the cells of [12, 13, 14] run at 0.3 to 60 GW, $x_p$ from about $10^2$ to
  $10^4$. Then $\sigma_p\ll0$ and $\sqrt{\sigma_p}$ has no real meaning. The plate does not
  collapse because it is much thinner than its self-focusing distance, of order
  $z_R/\sqrt{x_p}$. The moment law, in real arithmetic, holds for either sign while $W$
  stays positive, and the design refuses a plate in which $W$ reaches zero.
- *The strength of the Kerr lens of a thin plate.* For a plate of thickness
  $d\ll z_R/\sqrt{x_p}$ the moment law reduces to a thin lens, in the ambient medium, of
  $$
  \frac{1}{f_K}=x_p\frac{\lambda_0^2d}{\pi^2n_pw^4}=\frac{4}{\alpha}\,\frac{n_2P_{\rm eff}d}{\pi w^4},
  \qquad P_{c,p}=\frac{\alpha\lambda_0^2}{4\pi n_pn_2},
  $$
  which, for $\alpha=2$, is exactly the change of the second moment that the phase
  $k_0n_2I_0d\,e^{-2r^2/w^2}$ imprints on a Gaussian. The parabolic expansion of the same
  phase about the axis gives $1/f=8n_2Pd/(\pi w^4)$, $2\alpha\approx3.8$ times stronger: the
  centre of the beam is focused more than its wings, and the second moment follows the
  average. [13] and [14] do not state which lens their designs use; the two give different
  eigenmodes (section 7, Q7).
- *One lens for the whole pulse.* Each time slice sees its own lens, from zero in the
  wings to its full value at the peak, and $P_{\rm eff}$ weights the slices by energy. In
  the gas cells of [4] and [10] $\sigma$ goes from 1 in the wings to 0.4–0.6 at the peak;
  in a plate cell the wings see no lens and the peak slice the full lens of every plate,
  and near the edge of stability, where the mirror radius varies as
  $(2R-L)^{-1/4}$, the spread of the eigenmode across the pulse grows, which is the regime
  of [14]. The design reports the matched mirror radius for $P\to0$ and for the slice at
  the peak.

**Where the plates sit.** A fill scales the radii at the mirrors and at the centre by the
same factor $\sigma^{1/4}$; the lens of a plate moves them in directions set by its
position, and no single $\sigma$ for the cell can follow that. At the centre of a
symmetric cell the eigenmode enters the plate with $U=W/(2f_K)$, so the beam still widens
into the plate and the centre is wider than the linear waist, with a waist on either side;
beside a mirror the lens adds to the mirror's power and brings the cell toward the edge of
stability, so the mirror radius grows. [15] reports both signs (section 3). For a cell of
its geometry ($R=0.2$ m, $L=0.38$ m, 1 mm plates in air) the moment map gives, at
$x_p=10$, −24 % on the mirror radius and +24 % at the centre for five plates within 2 cm of
the centre, and, for two plates 8 cm from the mirrors, +2 % at $x_p=10$ and +16 % at
$x_p=100$, the order of [15]'s second cell. [15] does not state every position and power
behind its matching, so this checks the sign, not the magnitude (section 7).

**Air and polarisation.** In a hybrid cell the ambient gas is its own segment with
$\sigma_a=1-P_{\rm eff}/P_{c,a}$ [13]. For circular polarisation in an isotropic medium
with an instantaneous Kerr response the effective $n_2$ is two thirds of the linear one
[13]; the design uses it when the launch declares circular polarisation, while the run's
Kerr response needs no such factor.

**Validity.** The design is aberrationless in the sense above, monochromatic in its beam
parameters, and neglects the change of the pulse over a pass. In a fill the Kerr phase is
spread over the pass, and [4] finds the simulated B-integral 30 % above the aberrationless
estimate. In a plate cell the Kerr phase of a pass is concentrated in millimetres, about
0.3 rad per plate in [13] and [14], and the beam leaves each plate with an aberration of
that order, which the closure ignores until the cavity reshapes the beam. One published
case can be checked: for [13] (0.75 mJ, 598 fs sech², circular polarisation, two 0.5 mm
plates 280 mm apart in air, $R=300$ mm, $L=550$ mm), the moment closure with $P_{\rm eff}$
and the constant of section 4.3 gives a nonlinear-to-linear waist ratio of 0.926, against
$344/372=0.925$ in [13], while the on-axis lens gives 0.80. [13] does not state every
condition behind its matching, so this is a hint, not a validation. The design sizes the
grid and the launch; the propagated run is the result. The comparison of the two, pass by
pass, is itself an output (section 7) and, for plates, a deliverable of phase 3.

#### 4.5 References, frames, noise and energy

Each pass runs in its own moving frame; the `FieldPlane` reference carries the absolute
delay and carrier phase from pass to pass, so the stored spectrum stays in its window
and `plane.absolute()` is exact over the whole cell. Noise is drawn once, by the
executor, into the first pass's launch, with its seed recorded in the cell record; a
relaunch carries the realised field and draws none, and every runner is built without a
noise model, since a runner refuses an input whose noise model differs. Energies are the
axial flux of the medium (SEP 34), so a plate's Fresnel loss and a coating's reflectance
appear in joules in the ledger.

#### 4.6 Diagnostics

The on-axis B-integral of a pass is estimated from the peak-intensity statistic recorded
between planes, $B_j\approx k_0n_2\int I_{\rm peak}\,dz$ by the trapezoid rule, with its
sampling error stated; it is a diagnostic, not a quantity the run integrates. The
Fourier-limited duration uses `transform_limit` on the beam-integrated spectrum; a
spatio-spectral homogeneity measure is chosen in phase 4 (Q4).

### 5. Proposed interface

#### 5.1 Description and design

```text
from solprop.multipass import CellMirror, MultipassCell, Plate
from solprop.optics import TaylorPhase
from solprop.spatial import SpatialMedium

mirror = CellMirror(radius=0.3, reflectance=0.9995,
                    coating=TaylorPhase((0.0, 0.0, 25e-30), 1030e-9), model_error=1e-3)
cell = MultipassCell(mirror, length=0.595, passes=27, medium=SpatialMedium("Ar", 1.0))
design = cell.design(pulse)        # CellDesign: psi, rayleigh_range, waist(omega),
                                   # mirror_radius, sigma per pass and segment,
                                   # power_ratio, plate Kerr focal lengths,
                                   # b_axis and b_beam per pass, mirror fluence
launch = design.matched_plane(grid, space)   # FieldPlane at the first mirror
```

`MultipassCell(mirror, *, length, passes, medium, plates=(), n2=None)`: `mirror` one
`CellMirror`, or a pair `(first, second)` of equal radius whose coatings differ; `length` in
m with $0<L<2R$, `passes` a positive integer (the passes simulated, not round trips),
`medium` the fill or the ambient medium, `plates` a sequence of `Plate(medium, *,
thickness, position, interface="fresnel" | "matched", n2=None)` with `position` the
distance of the plate's entrance face from the first mirror along a pass. `n2` (m²/W)
overrides the design's index of the fill or ambient medium, and a plate's its own; None,
the default, takes it from optmat (section 4.3). An override reproduces a publication; the
run's responses are the caller's, as on every free-space route, so the run then needs a
`KerrResponse` whose coefficient is scaled to the same index, and the cell record keeps
the design's index beside the run's so that a mismatch is visible.
`CellMirror(radius, *, reflectance, coating=None, model_error)` expands to the elements of
section 4.1. `cell.design(pulse, *, power=None, critical="published" | "variance",
plate_lens="moment" | "on_axis")` returns a frozen `CellDesign`, with the two lenses of
section 4.4 for plates; `power` (W), None by default, replaces the $P_{\rm eff}$ computed on
the pulse by a publication's effective power, $E/(\sqrt2\,\tau)$ for [10]. It refuses
$L\ge2R$, a pair of mirrors of different radii, a non-positive $\sigma$ in a fill or an
ambient gap, a plate in which the moment law reaches $W=0$, and a pulse with no finite
energy.

#### 5.2 Running a cell

```text
from solprop.multipass import MultipassSimulation

simulation = MultipassSimulation(cell, grid=grid, space=RadialGrid(4e-3, 256),
                                 responses=[KerrResponse(...)], settings=settings)
result = simulation.run(launch, output="cell-run/")
result.ledger                     # per pass and per element, in joules
result.pass_output(26)            # an ordinary run output: field_plane, statistics, plots
result.series("mirror_radius")    # per-pass observables against z_abs
```

`MultipassSimulation(cell, *, grid, space, responses, settings, ambient_responses=())`
builds one `Simulation` and one `SimulationRunner` per distinct nonlinear segment of a
pass: the whole pass in a gas cell; each plate, and each ambient gap when
`ambient_responses` is not empty, in a solid-state cell. Linear ambient gaps are
`FreeSpace` elements. Every segment uses the same temporal grid and the same transverse
grid. Each pass's save planes include its centre and its end, so the observables at the
mirrors and at the focus exist. `with_launch(plane)` returns a copy bound to a launch, which
`run` uses when it is given none; scans need it (section 5.5).

#### 5.3 What a run produces

- One ordinary output per segment run, readable by every existing reader, and a cell
  record beside them: the cell, the design, the launch's identity, and per pass the
  configuration hash of each run, its $z_{\rm abs}$ offset, the transport reports of the
  elements that followed it, the run's boundary-monitor totals and the thin-screen
  certificate of each mirror.
- The ledger closes per pass: energy in, coating and interface losses, removed content,
  approximation errors of the screens, the boundary monitor's removal and energy out.
  The cell is the object that owns a chain's ledger, which #238 found missing.
- A rerun with the same output skips the passes whose records match and relaunches from
  the last completed one. An interrupted pass restarts from its launch.

#### 5.4 Refused

The modified noise model (`field_plane` refuses it), TDSE responses (a runner refuses
them), planar and line grids, a medium varying along a pass, different grids between
segments, a plate outside $[0,L]$ or overlapping another, oblique plates, mirrors of
different radii and astigmatic mirrors, each by name.

#### 5.5 Scans of cells

```text
from solprop.driver.scan import Scan, run_scan

def build(point):
    cell = MultipassCell(mirror, length=0.495, passes=27,
                         medium=SpatialMedium("Ar", float(point.values["pressure"])))
    return MultipassSimulation(cell, grid=grid, space=space, responses=responses,
                               settings=settings).with_launch(launch)

scan = Scan("mpc1-pressure", pressure=np.linspace(0.5, 3.5, 7))
result = run_scan(scan, build, output="mpc1-pressure.h5",
                  reductions=[cell_spectral_width(), cell_transform_limit()])
```

`run_scan` accepts a builder that returns a `MultipassSimulation` bound to its launch
(`with_launch`, as `Simulation.with_input`), all points of a scan being of one kind. Each
point runs its cell into its own directory beside the collected file, so every pass
output stays readable; the collected file holds, per point, the identity of the cell
record, the statistics and retained fields of the last pass, and the declared reductions,
which for a cell receive the cell's result rather than one run. A per-pass reduction has
the shape of the passes, so it refuses a scan of `passes`, which is not needed: one cell of
$N$ passes gives every shorter one through its pass outputs, the extraction optics being
out of scope. The point's seed is the cell's single noise draw (section 4.5). Consecutive
points that change only the launch (energy, chirp, duration) reuse the cell's runners, as
consecutive simulation points reuse their backend; a point that changes the medium, the
geometry or the plates rebuilds them. A completed point is skipped on a rerun and an
interrupted one resumes from its last completed pass (section 5.3). `Local` and `Pool` run
a point's passes in one worker and `Batch`, `Queue` and `Slurm` dispatch whole points; a
point's memory admission is that of its largest segment and its retained planes. Whether
the design is recomputed at each point (a matched cell) or kept from a nominal point (a
mismatch study) is the builder's choice.

### 6. Implementation strategies

#### 6.1 Architecture

**A — composition on the host (recommended).** Each pass is one or a few ordinary runs; the
mirror, the coating, the interfaces and the linear gaps are applied with `transport`
between them; the next pass is launched with `couple(..., same_basis)` into the same
runner's port, after declaring the transported plane at the next entrance
(`geometry=port.geometry`). This uses only qualified elements with their certificates and
reports, needs no seam or native change, and runs wherever the free-space route runs. A
mirror is a discontinuity of the field's representation, so a march must stop there and
restart its history in any architecture: the composition loses no accuracy. Its cost is a
host copy of the field and the transform pair of the screen per pass (section 6.4).

**B — mirror events inside the march.** A new leaf of `PropagationProblem` would declare
screens at fixed positions, the march would land on them, apply them on the device and
restart its history; one run would cover the cell. It would keep a cell on the device and
make it an ordinary `Simulation` for scans and the front end, but it is a change of
physics inside a run, which SEP 47 section 5.1 excludes; it needs kernels on both
backends, a rewrite of the frozen tables and an in-run version of the thin-screen
certificate, and a solid-state cell would also need media varying abruptly along z
(#193, #206). Rejected for now; A would be its reference if measurements ever justify it.

**Executor or recipe.** #191 allows a documented loop in user code if that is enough. It is
not: the loop must hold the ledger across runs (#238), the noise identity, the resume, the
frame and reference bookkeeping, the matched launch and the per-pass observables, and
every user would rewrite them. The executor stays thin: a loop over passes and segments,
with no scheduling of its own.

#### 6.2 Layers and modules

- **P** — `src/solprop/multipass.py` (new): `MultipassCell`, `CellMirror`, `Plate`,
  `CellDesign` and the design of section 4, with NumPy and SciPy only; it expands a mirror
  into `ThinLens` and a `SpectralTransfer` of the coating, and builds the matched launch
  with `FieldPlane.from_analytic`. That suffices for a scalar radial cell. For two
  components, and on the Cartesian grid where the image along p matters, `Mirror` gains an
  unfolded form that applies its Jones map and image and keeps the forward axis (phase 3);
  no new element class is added. `PlanarInterface` gains a matched, coated case
  (`interface="matched"`): the medium changes with the field scaled to conserve the axial
  flux, for an antireflection-coated plate. `critical_power` gains its mass as a keyword
  (section 4.3). Registering the module touches `architecture.md` section 4, the figure
  builder behind `structure.md` figure 1 and the package's list of submodules, which the
  structure and package tests check.
- **D** — `src/solprop/driver/multipass.py` (new): `MultipassSimulation`, the pass loop, the
  ledger, the cell record and the resume. `SimulationRunner.run`, in `driver/runner.py`,
  accepts a `CouplingLaunch` bound to its port, as `Simulation.with_input` already does; the
  change is small. The scan driver accepts a
  cell point (section 5.5): the type check, the point's directory, reductions on a cell
  result and the reuse of a cell's runners across points, in the scan modules, which hold
  the logic of a point today; `simulation.py` is not touched for it.
- **X** — `src/solprop/processing/multipass.py` (new): the per-pass series of section 5.2
  from the pass outputs and the cell record, and one plotting function.
- **T**, **B** — none. `PropagationProblem`, `RunRequest`, the stage bundles and the frozen
  tables of `tests/backends/test_driver.py` are unchanged. The new modules enter
  `architecture.md` section 4 and `structure.md` figure 1.

#### 6.3 Backends

No backend code. Every run is an ordinary free-space run on the backend it selects; the
qualification of the cell is that of its runs, with parity between `numpy_ref` and the
native CPU build (ADR-068, ADR-077). A reused runner on a
GPU build compiles but is not qualified today ([backends](../backends.md)); a cell on a GPU
inherits that status, plus one host round trip of the field per pass.

#### 6.4 Cost in time and memory

Per pass the executor adds: the host copy of one field of
$n_r\times n_{\rm pol}\times n_\omega$ complex values (8 MB for 128 × 1 × 4096); the screen's
transform pair on the host, $O(n_r^2n_\omega)$ for the radial transform; and a cold start of
the step controller. A gas cell holds one backend; a solid-state cell one per distinct
nonlinear segment, at most three for a centred plate in a nonlinear ambient gas. Field
retention keeps the fields at the mirrors only unless asked. The overhead per pass against
the propagation of a pass is recorded in a benchmark ledger entry on about ten steps per
pass over three passes.

#### 6.5 Cost in code

Expected, from `poe size`, over the five phases: about 1550 lines of code in `src/solprop`
(design and description 500, with the moment map of section 4.4; executor and record 450;
processing and plotting 250; the runner and interface changes 80; scans of cells 200;
documentation helpers 70), about 520 lines of public docstrings, and about 1350 lines of
tests. Each phase is its own pull request, the largest (phase 2) about 550 net lines in
`src/solprop`, under the size guard's limit of 1500.

#### 6.6 Dependencies

None: the fixed point of section 4.2 uses `scipy.optimize`.

### 7. Verification and validation

**Design (fast tests).**

- $\psi$, $z_R$, $w_0$ and $w_m$ against the closed forms of section 4.2, including the
  near-concentric sensitivity; the numerical fixed point against the closed form for the
  gas cell, to a rounding budget.
- $P_{\rm eff}$ for a Gaussian ($P_{\rm peak}/\sqrt2$) and a sech² ($2P_{\rm peak}/3$) pulse on a
  grid, to the quadrature's error.
- The cell of [4] section 3 (1030 nm, 700 fs, 7 mJ, argon at 1 bar with
  $n_2=0.97\times10^{-23}$ m²/W, $R=1$ m, $L=1.995$ m): $\sigma=0.595$, the matched waist
  $\sigma^{1/4}w_{0,\rm lin}$, $B=4.26$ rad (4.3 in [4]); [4] quotes a linear waist of
  131 µm, which corresponds to $L=1.9945$ m, against 128 µm at 1.995 m.
- MPC3 of [10] in its Fig. 2 (2 mJ, 300 fs, $M^2=1.05$, argon at 2 bar, $R=0.5$ m,
  $L=0.978$ m): the linear mirror fluence of 111 mJ/cm² and the matched fluence of
  170 mJ/cm² with the effective power $E/(\sqrt2\,\tau)$ of [10] (section 4.3), to 1 %;
  the 164 mJ/cm² of the exact $P_{\rm eff}$ is recorded beside it.
- The four cells of [10], its Table 2, with the lasers of its Table 1, MPC3 at 2 bar,
  $n_2=9.7\times10^{-24}$ m²/W per bar and the effective power of [10]: the matched mirror
  fluence, 70, 133, 156 and 145 mJ/cm² against 75, 130, 160 and 140 for MPC1 to MPC4, and
  the focus intensity, within 5 % of the table except for MPC2 (6 to 12 % low, depending on
  the peak power assumed). The tolerance is the table's two significant figures, 5 %, with
  MPC1 and the intensity of MPC2 recorded as exceptions.
- The moment map of section 4.4 against the $\hat q$ form for a fill, to rounding, and
  against the closed form with the plates removed.
- The linear eigenmode of [12] ($R=300$ mm, $L\approx550$ mm, 1037 nm): radii of 0.57 mm on
  the mirrors, 0.17 mm at the centre and 0.34 mm in the plates, which the closed form gives
  to the precision of the quoted values.
- The matched waist of [13]: 372 µm in diameter for the linear match, which the closed form
  gives with the published $M^2\approx1.26$, and 344 µm for the nonlinear one; the ratio
  under both plate lenses (0.926 with the moment lens, 0.80 on the axis) is recorded as
  evidence for Q7.
- The linear mirror radius of [2], 460 µm, which the closed form gives (459 µm) for
  $R=0.35$ m and $L=0.54$ m with the meniscus given to the mirror (section 2).
- The signs of section 4.4 for a cell of the geometry of [15]: the mirror radius falls
  with plates at the centre and rises with plates beside the mirrors, the centre radius
  moving the other way; the magnitudes are recorded, and compared with the 30 % and 15 % of
  [15] only if its positions and powers are known.
- The thin-lens limit of a plate against the moment law, the interface rule of section
  4.4, and the refusals.

**Elements (fast tests).** The unfolded mirror against `Mirror` followed by the rotation of
section 4.1 on a Cartesian plane; two flat, lossless mirrors restore the field; the
ledger of a coating closes bin by bin; the matched interface conserves the axial flux to
the transform's rounding.

**Composition (medium tier, a few passes of about ten steps).** The checks of #191:

- **Identity handoff:** a flat, lossless mirror between two passes of $L$ against one run of
  $2L$, as an accuracy envelope at the tolerance's level, on both CPU backends.
- **Paraxial Gaussian through an ABCD system:** a linear cell launched on its eigenmode keeps
  its second-moment radius at the mirrors and the centre; launched off it, the radius
  oscillates with the period the Gouy phase predicts; tolerances from the thin-screen
  certificate and a grid convergence study.
- **Coherent phase and delay:** the plane's reference after several passes equals the sum
  of the frame integrals and `plane.absolute()` is continuous at each mirror.
- **Energy:** each pass's output energy is its input times the coating's reflectance, per
  bin, to a rounding budget; the ledger closes within the declared approximation errors.
- **Dispersion:** the spectral phase of a linear cell after $N$ bounces is $N$ times the
  coating phase, or for a pair $\lceil N/2\rceil$ times the second mirror's and
  $\lfloor N/2\rfloor$ times the first's, plus the fill's dispersion.
- **Noise and resume:** no draw after the first launch; a resumed cell repeats an
  uninterrupted one within an accuracy envelope.
- **Weak Kerr:** for $P\ll P_c$ the diagnostic B-integral approaches $B_{\rm axis}$ with
  $\sigma\to1$.
- **Radial against Cartesian** on the symmetric on-axis cell, as #191 asks (phase 3).

**Solid state (phase 3).** A glass medium with a Kerr response on the radial route, which
nothing refuses today but no test covers; Fresnel ledger $R+T=1$ through `PlanarInterface`;
flux conservation through the matched interface; a thin plate's B-integral against
$k_0n_2I_0d$; the design's fixed point with a plate against the linear run. The change of
the second moment across one thin plate, for a long pulse at low B-integral, against the
moment lens of section 4.4, which is exact there for $\alpha=2$: this checks the run, not
the closure. Then the closure itself: a plate cell launched on the eigenmode of each lens,
whose mirror radius is followed over the passes; the one that keeps the beam more
stationary, and by how much, is the evidence for Q7.

**Integrated references (galleries, run by hand).** Published simulations and measurements
are witnesses, not truth; each comparison states its tolerance in the phase report.

- [4] section 3: the beam radius at the mirrors and the centre over 16 passes for the
  linear and the nonlinear launch (2.5 → 1.9 mm against 2.18 ± 0.02 mm at the mirrors), the
  peak intensity at the centre and the on-axis B-integral of the first round trip
  (11.2 rad in [4]).
- [10] MPC2 (2 mJ, 320 fs, $R=300$ mm, $L=595$ mm, argon about 1 bar, 27 passes as in the
  simulation of [10], whose Table 2 extracts at pass 25): the spectral width at $1/e^2$, the
  Fourier-limited duration against pressure and its $1/p$ trend, and the compressed
  duration with $-1200$ fs² and $+2000$ fs³ (18.5 fs in [10]); the measured spectra and
  durations of [10] as experimental witnesses.
- [10] MPC1, MPC3 and MPC4 at their optimal pressures: the spectral width at $1/e^2$
  (140–160 nm for all four cells), the Fourier-limited durations (18.3, 19.5 and 20.5 fs
  measured, 17.9 fs for MPC2) and the compressed ones; and the spectral width and
  Fourier-limited duration of MPC1 against pressure at 800 µJ (its Fig. 3), as a scan
  (phase 5). [10] shows that the peaks near 1030 nm in the short cells depend on the
  temporal contrast of the input (its Fig. 6), so spectral shapes carry the uncertainty of
  the launched pulse, and the comparison is on widths and durations.
- [12]: two 2 mm fused-silica plates 12.5 cm from the mirrors, in air, 41 passes, 200 µJ
  at 1037 nm stretched from 170 fs to 600 fs. With a negative pre-chirp the spectrum
  narrows to 3.8 nm at pass 6 and broadens from pass 7 to 49.6 nm (46.7 fs
  Fourier-limited) at pass 41; with a positive one it reaches 63.1 nm (43.3 fs). The
  narrowing pass by pass tests the per-pass observables directly. The $n_2$, the mirror
  dispersion and the plate transmission are not stated in [12]; they enter as declared
  assumptions with their ranges.
- [13]: the cell in air alone (about 100 fs Fourier-limited) and with its two plates
  (42 fs Fourier-limited, 45 fs compressed), in circular polarisation, which exercises two
  field components, the change of handedness at each bounce (section 4.1) and the factor
  2/3 of the design.
- [14]: 24 passes near the edge of stability ($|A+D|/2\approx0.992$), a 0.8 mm plate before
  each mirror in vacuum, 30 mJ, about 460 fs: the width at −10 dB (32.5 nm measured,
  32.2 nm simulated in [14]), the Fourier-limited duration (91.8 fs measured) and the
  transmission. It stresses the design's sensitivity near the edge and the radial grid,
  whose mirror spot is more than 15 times the focus.

**Sensitivity to the nonlinear index (phases 2 to 4).** The comparisons with measurements
use optmat's $n_2$, which is what a user of solprop gets. Each of them, for [4], [10] and
the solid-state reference chosen in phase 3, is run a second time with the publication's
index ($9.7\times10^{-24}$ m²/W per bar of argon for [4] and [10], $2.2\times10^{-20}$ m²/W
for the silica of [14]; for [12] the declared range), through the design's `n2` and a Kerr
coefficient scaled to it, each launch matched by its own design. The phase report gives,
per reference, the change of the spectral width, the Fourier-limited and compressed
durations, the mirror radius over the passes and the B-integral, against the design's
estimate of section 4.3 (+3.9 to +4.6 % on the B-integral for argon). Where the
difference between the two runs exceeds the tolerance against the measurement, the
comparison states both.

**Scans (phase 5).** A scan of three cells reproduces the same cells run alone, identically
on one backend; a rerun skips completed points and resumes an interrupted one from its last
completed pass; consecutive points that differ only in energy reuse the runners; a per-pass
reduction refuses a scan of `passes`; the seeds of cell points are those of simulation
points.

**Convergence.** Radial samples and outer radius, temporal samples, tolerance and the
screen budget, at three levels each.

### 8. Documentation

- A manual page, `docs/guide/workflows/multipass-cells.md`: the unfolded model and its
  limits, the design and its conventions, including the closure for plates and the source
  of the nonlinear index and how to reproduce a publication's, running a gas
  and a solid-state cell, scanning cells, the ledger, resuming, and what is refused; a
  section of [scans](../../guide/workflows/scans.md) on builders of cells.
- Docstrings for every public name; a row in the composition table of
  [coupling simulations](../../guide/workflows/coupling.md).
- A focused tutorial in `docs/tutorials/free_space/`: a short linear-then-Kerr cell of a few
  passes, running in seconds.
- Three worked examples in `docs/examples/`: the matching study of [4]; the four cells of
  [10], with the pressure scan of MPC1; and a solid-state cell, [13] or [14], chosen in
  phase 3 by the completeness of its data (Q5).
- `architecture.md` section 4, `structure.md` figure 1, the capability matrix and a
  `changelog.d/` fragment per phase.

### 9. Compatibility and impact

Additive. No default changes, no change to the run record or the output schema of a single
run, and none to the seam. The cell record is a new file with its own version.
`SimulationRunner.run` gains an accepted input type. `PlanarInterface` gains an option whose
default keeps today's behaviour. Nothing is expected on the differences-from-Luna page.

### 10. Alternatives considered

| Alternative | Assessment |
|---|---|
| Mirror events inside the native march (B) | One run per cell and no host round trip, but a change of physics inside a run (SEP 47 section 5.1), kernels on both backends, a seam change and media varying abruptly along z for plates; no accuracy gain, since a march restarts at a mirror anyway |
| A reduced propagation model (ABCD and a one-dimensional pulse equation, or a hybrid model) | Fast and useful for sizing, which the design keeps; it cannot show aberrations, spatio-spectral coupling or homogeneity, which is why a full propagation is wanted |
| Propagation in the Laguerre–Gauss basis of the cavity | Few modes in the linear regime and mirrors diagonal, but the basis depends on frequency and the mode count grows with nonlinearity; a new route |
| A rescaled transverse coordinate following the beam size | Would shrink the grid of a near-concentric cell; an optimisation for later, measured against the fixed grid |
| One HDF5 file with a group per pass | One file, but a change to the output layer; a directory of ordinary run files and a cell record reuses every reader (Q2) |
| A fresh `with_input` run per pass | Correct but pays the backend's allocation and planning at every pass; the runner pays it once |
| A loop in user code only | Leaves the ledger, the noise identity, the resume and the observables to every user |
| A scan loop over cells in user code | Loses the executors, the resume, the seeds and the collected file that `run_scan` already provides |
| The on-axis parabolic Kerr lens as the only plate lens | Simple, but $2\alpha$ times the second-moment response of the beam and inconsistent with the $\sigma$ of a fill in the same design; kept as an option for comparison (Q7) |
| The $\hat q$ parameter of [4] extended to plates as published | Undefined for $\sigma<0$, which every plate of [12, 13, 14] reaches; the moment law is the same for $\sigma>0$ and holds for both signs |

### 11. Phases

| Phase | Deliverables | Required evidence |
|---|---|---|
| **1 — Cell design** | `multipass.py`: the description, the linear and Kerr-corrected design with the moment map for plates and both plate lenses, the matched launch plane; the design section of the manual page | Closed forms, the numbers of [2], [4], [10] (Fig. 2 and Table 2), [12] and [13], the signs of [15], $P_{\rm eff}$ for two shapes, refusals |
| **2 — Gas-filled cells** | `driver/multipass.py`, the runner accepting a launch, the ledger, the cell record, the resume, minimal per-pass series; the tutorial; the worked example of [4] | The composition checks of section 7 but the radial–Cartesian one; parity on the CPU backends; the benchmark ledger entry; the comparison with [4] and its sensitivity to $n_2$ |
| **3 — Solid-state cells** | Plates, the matched interface, linear and nonlinear ambient gaps, the unfolded `Mirror` for two components, the Cartesian check, the solid-state worked example | The solid-state checks of section 7, including the closure comparison for Q7; radial against Cartesian; the comparison with the chosen reference among [12, 13, 14] and its sensitivity to $n_2$ |
| **4 — Observables and compression** | `processing/multipass.py`, plotting, the homogeneity measure, the worked example of the four cells of [10] at their optimal pressures, the complete manual page | The reference comparisons of section 7 with their tolerances and their sensitivity to $n_2$; convergence studies; final gates |
| **5 — Scans of cells** | `run_scan` accepting cells, reductions on a cell result, the pressure scan of MPC1 in the worked example of [10], the section of the scans page | The scan checks of section 7; the spectral width and Fourier-limited duration of MPC1 against pressure |

The follow-ups go in issues: the spot pattern and re-entrance, the coupling optics and
their nonlinearity, recipes and front-end hooks for cells (SEP 41), astigmatic mirrors on
the Cartesian grid, and mirror events inside the march if a measurement ever calls for
them.

### 12. Open questions

1. **Q1 — The convention of $\sigma$:** resolved by the author on 2026-11-13. The design
   uses the exact $P_{\rm eff}$ of eq. (6) of [4] on the launched pulse and $P_{\rm peak}$
   in the B-integral. Both constants of $P_c$ stay selectable; the default, the published
   $3.77\lambda_0^2/(8\pi n_0n_2)$ as proposed in section 4.3 or the derivation's
   $\lambda_0^2/(2\pi n_0n_2)$, is left to the review. [10] computed with
   $E/(\sqrt2\,\tau)$, which the design reproduces through `power`, and the MPC3 of its
   Table 2 is at 2 bar (section 4.3). The nonlinear index comes from optmat, and its effect
   against the literature's values is measured (sections 4.3 and 7).
2. **Q2 — The form of the output:** a directory of ordinary run files and a cell record
   (proposed), or one file. Left to the review (the author, 2026-11-13). The author
   prefers the directory, which every existing reader and the resume use as they are, and
   asks the reviewers to weigh the number of files it makes: one per nonlinear segment and
   pass, so 27 for the MPC2 of [10], 48 for [14], 82 for [12] with its air linear and 205
   with its air nonlinear, and 189 for the pressure scan of MPC1 at seven points. The
   volume is the same in one file, set by the retained planes (about 8 MB per field on 128
   radial and 4096 spectral samples); the count is what differs, and it matters on file
   systems with a quota of files. A tool that packs a finished cell into one file would be
   follow-up work.
3. **Q3 — Scans of cells:** resolved by the author on 2026-11-12: in scope, phase 5
   (section 5.5).
4. **Q4 — The homogeneity measure** for phase 4. [12] and [14] both report the spectral
   overlap measure of Weitenberg et al.; it is the proposed default, defined and cited from
   its primary source in phase 4. Deferred to phase 4.
5. **Q5 — A solid-state reference:** answered by the author with [2, 12, 13, 14, 15]. Which
   one becomes the worked example, and how the inputs that [12] does not state ($n_2$,
   coating, plate transmission) and the conditions behind the nonlinear match of [13] are
   fixed, is decided in phase 3. The second cell of [15] has a complementary pair of
   chirped mirrors, which the first version admits, and plates at Brewster's angle, which
   it does not (section 2); its first cell, at normal incidence, is within it.
6. **Q6 — Coating data:** whether `TaylorPhase`, `ChirpedMirror` data and a callable with
   its calibration are enough to describe the coatings in use. Deferred to phase 2, where
   the coatings are first applied (the author, 2026-11-13).
7. **Q7 — The Kerr lens of a plate in the design:** the moment lens (proposed default,
   consistent with the $\sigma$ of a fill and with the one published case, [13]) or the
   on-axis lens, $2\alpha$ times stronger (section 4.4). Both stay selectable. Deferred to
   phase 3, settled by the closure comparison of section 7 and the references.

### 13. References

DOIs, titles and authors checked against Crossref on 2026-11-12, and for [15] on
2026-11-13.

1. D. Herriott, H. Kogelnik and R. Kompfner, "Off-axis paths in spherical mirror
   interferometers," *Applied Optics* **3**, 523 (1964). doi:10.1364/AO.3.000523
2. J. Schulte, T. Sartorius, J. Weitenberg, A. Vernaleken and P. Russbueldt, "Nonlinear pulse
   compression in a multi-pass cell," *Optics Letters* **41**, 4511 (2016).
   doi:10.1364/OL.41.004511
3. M. Hanna, X. Délen, L. Lavenu, F. Guichard, Y. Zaouter, F. Druon and P. Georges,
   "Nonlinear temporal compression in multipass cells: theory," *Journal of the Optical
   Society of America B* **34**, 1340 (2017). doi:10.1364/JOSAB.34.001340
4. M. Hanna, L. Daniault, F. Guichard, N. Daher, X. Délen, R. Lopez-Martens and P. Georges,
   "Nonlinear beam matching to gas-filled multipass cells," *OSA Continuum* **4**, 732
   (2021). doi:10.1364/OSAC.418247
5. P.-A. Bélanger and C. Paré, "Self-focusing of Gaussian beams: an alternate derivation,"
   *Applied Optics* **22**, 1293 (1983). doi:10.1364/AO.22.001293
6. G. Fibich and A. L. Gaeta, "Critical power for self-focusing in bulk media and in hollow
   waveguides," *Optics Letters* **25**, 335 (2000). doi:10.1364/OL.25.000335
7. A. Couairon, E. Brambilla, T. Corti, D. Majus, O. de J. Ramírez-Góngora and M. Kolesik,
   "Practitioner's guide to laser pulse propagation models and simulation," *The European
   Physical Journal Special Topics* **199**, 5 (2011). doi:10.1140/epjst/e2011-01503-3
8. S. N. Vlasov, V. A. Petrishchev and V. I. Talanov, "Averaged description of wave beams in
   linear and nonlinear media (the method of moments)," *Radiophysics and Quantum
   Electronics* **14**, 1062 (1971). doi:10.1007/BF01029467
9. A.-L. Viotti, M. Seidel, E. Escoto, S. Rajhans, W. P. Leemans, I. Hartl and C. M. Heyl,
   "Multi-pass cells for post-compression of ultrashort laser pulses," *Optica* **9**, 197
   (2022). doi:10.1364/OPTICA.449225
10. J.-F. Hergott, F. Réau, O. Tcherbakoff, F. Lepetit, N. Lericheux, M. Guer, H. Marroux,
    R. Géneaux, A. Tyagi, R. Cazali, P. Nonnon, D. Bresteau, T. Ruchon, P. Salières and
    T. Auguste, "Experimental and numerical study of direct sub-20fs multi-pass-cell
    post-compression of up to 4mJ, 80W Yb-based lasers," *Applied Physics B* **131**, 56
    (2025). doi:10.1007/s00340-025-08415-0
11. A. E. Siegman, *Lasers* (University Science Books, 1986).
12. L. Shen, J. Song, Y. Peng, G. Luo, Y. Liu, J. Sun and Y. Leng, "Post-compression of the
    GW-level femtosecond pulse in a solid-state multi-pass cell," *High Power Laser Science
    and Engineering* **12**, e80 (2024). doi:10.1017/hpl.2024.67
13. Z. Gao, J. Guo, Y. Huang, J. Li, L. Guo, Z. Tu, J. Xu and X. Liang, "High energy nonlinear
    pulse compression to sub-50 fs by a hybrid bulk-air multi-pass cell," *Optics Express*
    **33**, 13613 (2025). doi:10.1364/OE.549177
14. J. Manz, G. Barbiero, S. Klingebiel, C. Y. Teisset, H. Wang, M. Rampp, J. Brons,
    C. J. Saraceno and T. Metzger, "Bulk-based nonlinear pulse compression of 30 mJ and 150 W
    pulses of an ultrafast thin-disk amplifier," *Optics Express* **33**, 54907 (2025).
    doi:10.1364/OE.580757
15. A.-L. Viotti, C. Li, G. Arisholm, L. Winkelmann, I. Hartl, C. M. Heyl and M. Seidel,
    "Few-cycle pulse generation by double-stage hybrid multi-pass multi-plate nonlinear
    pulse compression," *Optics Letters* **48**, 984 (2023). doi:10.1364/OL.478790

## Part II — Implementation record

## Part III — History

- 2026-11-12: Drafted.
- 2026-11-12: Revised with the author's answers: scans of cells in scope (section 5.5,
  phase 5, Q3); solid-state references [12, 13, 14] (Q5); section 4.4 on what survives of
  the Kerr correction in cells with plates, and Q7; the four cells of [10] in the design
  checks and the galleries.
- 2026-11-13: [2] described as published, with the mirror substrates as the medium, and
  its linear radius in the design checks; [15] added, with the dependence of the Kerr
  correction on the position of the plates (section 4.4) and its check; the cells of
  Table 1 of [9] noted.
- 2026-11-13: Revised with the author's answers: Q1 resolved (the exact $P_{\rm eff}$; [10]
  used $E/(\sqrt2\,\tau)$ and has MPC3 at 2 bar; `power` reproduces a publication); the
  nonlinear index from optmat, its values for argon and silica against the literature's
  and a sensitivity run beside each reference comparison (sections 4.3 and 7); two
  coatings on the two mirrors of a cell (sections 2, 4.1 and 5.1); the context updated for
  the split of `simulation.py` and SEP 49, both merged.
- 2026-11-13: Q6 deferred to phase 2; the default constant of $P_c$ and the form of the
  output (Q2, with the number of files it makes) left to the review (the author).
- 2026-11-13: Tracking issue [#793](https://github.com/LupoLab/solprop/issues/793).
- 2026-11-13: GPT-6 completed the AI review below, incorporating the seven Codex
  findings on PR #794. Part I is unchanged; the findings await the author's response
  and the maintainer's decisions.

### AI review — 2026-11-13

Reviewer: **GPT-6 (AI)**. Reviewed all of Part I in
[PR #794](https://github.com/LupoLab/solprop/pull/794) at
`d1419335614a90148b86f4572c7df43a329e8036`, against the architecture, conventions,
SEP 0, and the existing coupling, runner, spatial-statistics, compression and scan
contracts. These findings are part of the work tracked by
[#793](https://github.com/LupoLab/solprop/issues/793). They are review comments and
questions, not revisions to the proposed model or approval to implement it.

**Assessment.** Composing ordinary free-space runs and existing optical elements is
a sound direction. The separation of a sizing model from the propagated result, and
the explicit comparison of nonlinear-index choices, are useful. Before acceptance,
the proposal needs firmer contracts for the plate approximation, the physical output
plane, restart, spatial compression and scan execution. Several advertised cases
cannot be expressed by the proposed interface. **P1** below means a correctness or
contract issue to settle before accepting the affected scope; **P2** means a necessary
clarification, qualification change or maintainer decision.

**Evidence and limits.** Code paths were read at the commit above. Short Python 3.14.5
probes independently integrated a Gaussian plate's phase kick, checked a cavity
matrix, compared an integrated-spectrum transform limit with two orthogonal spatial
channels, and checked phase-only chirping and the existing medium guard. Their inputs
and results are given below. With optmat 0.9.1, `physdata.n2` gives
`1.0001440357542289e-23` m²/W for argon at 1030 nm, 1 bar and 293.15 K, consistent with
the rounded value in §4.3. The publisher's full texts of
[Hergott et al. [10]](https://link.springer.com/article/10.1007/s00340-025-08415-0)
and [Shen et al. [12]](https://doi.org/10.1017/hpl.2024.67) were checked for the
comparisons discussed in R17. Full text of [4] and [13] was not accessible through
the sources attempted; their detailed numerical claims, the other experimental
reproductions, and the author's corrections to [10] have not been independently
verified. No cell implementation or full-length propagation was run.

**Recommendations on the questions left to review.** For **Q1**, I recommend
`critical="variance"` as the default of the moment design, with the published
constant an explicit reproduction option. The Gaussian phase-kick check is then
consistent with the design's coefficient. Keep the Townes default of the existing
`critical_power` unchanged, distinguish all three powers in the record, and do not
describe the variance scale as a physical no-collapse guarantee (R1–R2). This is a
recommendation for a maintainer decision, not a change to Part I. For **Q2**, I support
ordinary run files plus a versioned cell manifest, provided R7 defines the saved exit
plane and commit/restart contract. A later packing tool is preferable to changing
every run reader for this feature. Count the checkpoint files as well as segment
files when reporting the storage cost.

1. **R1 — P2: Finish the plate-lens derivation in the stated medium
   (§§4.3–4.4, 5.1, Q1, Q7).** The thin-plate formula says that its focal length is
   measured in the ambient medium but omits its index, $n_a$. Combining the proposed
   moment law with the exit rule $U_a=(n_p/n_a)U_p$ gives the nonlinear kick
   $$
   \frac{1}{f_{K,a}}=
   \frac{x_p\lambda_0^2d}{\pi^2 n_a n_p w^4}
   =\frac{4}{\alpha}\frac{n_2P_{\rm eff}d}{\pi n_a w^4}.
   $$
   Here $P_{\rm eff}$ is the power in the plate. The on-axis lens likewise carries
   $1/n_a$. For a continuous Gaussian with
   $w=0.7$ mm, $P=10^8$ W, $n_2=2.53\times10^{-20}$ m²/W, $d=1$ mm and
   $n_a=1.5$, direct radial integration of the phase kick gives
   $1/f=0.00447215889$ m⁻¹ for $\alpha=2$, whereas the printed expression gives
   $0.00670823833$ m⁻¹. This deliberately non-unity index checks the algebra; it is
   not an added cell use case. Either retain $n_a$ or state and bound the
   $n_a\simeq1$ approximation for the admitted gases. Also make clear that substituting
   $\alpha=3.77/2$ rescales this exact Gaussian moment kick: it is a calibrated
   approximation, not the same variance identity. Please retain the independent
   phase-imprint check for $\alpha=2$ rather than validating two uses of the same
   moment formula against each other.

2. **R2 — P1: Specify stability and distinguish moment positivity from physical
   collapse (§§4.2–4.4, 5.1, 7).** $0<L<2R$ certifies the empty symmetric cell,
   not a cell with focusing elements. For example, $R=0.3$ m, $L=0.55$ m and a
   centred thin lens with $f=0.05$ m give half-trip trace/2 $=-1.2916667$ although
   $L<2R$. Define how the physical round-trip map selects a branch, establishes
   linear stability, and tests the Jacobian of the nonlinear $(W,U)$ map; convergence
   of `scipy.optimize` alone is not stability. Require positive $W$ throughout every
   segment, including an interior minimum, and specify the refusal for no stable
   solution or an ambiguous branch. Separately, $W>0$ is not proof that the actual
   field cannot collapse: the proposal's own Gaussian collapse threshold is below
   the zero-Hamiltonian threshold. Nor does $P_{\rm eff}<P_c$ imply
   $P_{\rm peak}<P_c$; a Gaussian at $P_{\rm eff}=0.8P_c$ has
   $P_{\rm peak}=1.13137P_c$. What does the promised peak-slice design report when
   that slice has no stationary fill solution? State the diagnostic's validity and
   failure semantics without making the full propagator's admission depend on an
   unproved collapse criterion. Include the exactly linear/vacuum limit explicitly,
   since `critical_power` refuses a zero susceptibility.

3. **R3 — P1: Define the physical plate layout and its reverse traversal
   (§§4.2, 4.4, 5.1–5.2, 6.1).** Incorporates
   [Codex's plate-order finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018681).
   Record a plate once as the interval $[p,p+d]$ measured from the first mirror.
   A return pass must traverse the plates in reverse order, with entrance
   $L-p-d$, the corresponding reversed gaps, and the appropriate incident/transmitted
   interface media. Use the same sequence in the design and the executor. A pair
   of unequal plates at asymmetric positions is necessary evidence; a centred plate
   or a symmetric pair hides the error. How are a plate touching a mirror and its
   two crossings around that reflection represented without a negative or duplicate
   gap, especially for the substrate geometry mentioned in §2?

4. **R4 — P1: Separate a beam-quality design parameter from a realizable launch
   (§§4.2–4.4, 5.1, 7).** Incorporates
   [Codex's beam-quality finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018691).
   The specified Gaussian has $M^2=1$, whereas the acceptance numbers use 1.05 and
   approximately 1.26. Define the measured radius and beam-quality convention and
   derive its place in the second-moment invariant and nonlinear closure. Merely
   enlarging a coherent Gaussian waist does not produce $M^2>1$: it changes its
   Rayleigh range while keeping its beam quality at one. An $M^2$ value alone also
   does not determine a unique coherent field or its Kerr response. Please choose
   between a qualified field family/measurement for such launches and an explicitly
   approximate design-only parameter whose output does not claim to realize it.
   Keep ideal-Gaussian numerical checks separate from experimental comparisons with
   measured beam quality, and test the reported quality of any generated launch.

5. **R5 — P1: Bind nonlinear responses to each segment and distinguish electronic
   and delayed indices (§§4.3–4.4, 5.1–5.2, 7).** A cell admits multiple bulk media
   and a design `n2` for each plate, but the executor has one `responses` sequence
   for all plates and one for the ambient. A `SpatialMedium` supplies dispersion and
   density; it does not choose the plate's nonlinear response. Two different glasses
   therefore cannot be assigned their respective Kerr/Raman responses through this
   API. Specify a per-segment binding, or narrow the admitted plate family explicitly.
   The quoted silica index includes a delayed Raman share, while `KerrResponse` is
   instantaneous: setting its coefficient from the total index and then adding Raman
   counts that share twice; using only the electronic share gives a different design.
   Define the effective index used for a finite pulse, the polarization convention,
   and what scaling a literature index changes. The instantaneous circular factor
   $2/3$ must not silently scale a delayed tensor response. Require the recorded
   design/run comparison to identify coefficients and response models per segment,
   and demonstrate one glass example with consistent units and normalization.

6. **R6 — P2: Bound the fixed-pulse approximation when coatings or plates change
   the pulse (§§4.1, 4.3–4.4, 5.1).** The recurrence
   $P_{{\rm eff},j}=R_c^jP_{{\rm eff},0}$ assumes an unchanged temporal shape and
   identical, frequency-independent losses. Different coatings, spectral filtering,
   plate dispersion and chirped mirrors are all admitted. A phase-only coating can
   preserve energy while changing $P_{\rm eff}$: an initially transform-limited
   Gaussian of 300 fs intensity FWHM with $10^5$ fs² GDD has an effective-power
   ratio of approximately 0.308747 at unchanged energy, as checked by Fourier
   transformation. Please define whether the design
   propagates a temporal surrogate through these known linear elements or reports
   a fixed-pulse estimate with an explicit limitation. In particular, distinguish
   instantaneous matched radii at reduced power from the actual drift of one launch
   propagated through a changing map; substituting a new $\sigma_j$ into the
   stationary formula does not compute that drift. Give an unequal-coating example
   and state which result is compared with the propagation.

7. **R7 — P1: Define the cell's output plane and make restart a transaction
   (§§4.1, 5.2–5.3, 5.5, 6.4, Q2).** A gas pass has one run output; a plate pass
   has several, and its last run may end before a linear gap and the final mirror.
   Consequently `pass_output(j)`, the scan's retained last-pass field, and the next
   launch are not necessarily the same plane. State whether an $N$-pass result is
   before or after the $N$th reflection, how many coating factors it contains, and
   how callers access both sides and each segment. Retain or reproducibly reconstruct
   the actual terminal `FieldPlane`, including its medium, geometry, calibrated
   reference and transport reports; an ordinary final-segment output alone is
   insufficient. Also define the terminal result when no segment has nonlinear
   responses.
   Define a manifest commit point after the required files are durable, the lock
   covering a cell directory, and the identity covering launch, seed, ordered
   elements, media/responses, settings and calibration. Specify conflict handling
   and removal/replacement of only an incomplete attempt. Test interruption after
   segment completion, after applying the mirror, and before publishing the pass
   record: restart must apply each element exactly once and preserve the ledger.
   Mandatory handoff/checkpoint retention must survive `fields="none"` in a scan,
   and boundary stops must not be marked as completed passes.

8. **R8 — P2: Expose noise selection and its single-draw identity
   (§§4.5, 5.2–5.5).** Incorporates
   [Codex's noise finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018703).
   Add the public choice of supported input noise or no noise, an explicit seed,
   and the scan reseeding contract. A `FieldPlane` contains samples, not the
   metadata needed to decide whether to add a fresh draw. State how a caller supplies
   an already noisy plane without adding another floor, when an unseeded draw is
   persisted, and how restart before the first completed pass recovers it. The
   existing scan derives a seed from its name and index when `ScanPoint.seed` is
   absent; preserve that behavior as well as explicit-seed checks. Test noiseless,
   seeded, initially unseeded, supplied-realization and interrupted-first-pass cases.
   Every reused segment runner should continue to carry the proposed no-noise model.

9. **R9 — P1: Admit every simultaneously live runner and account for handoff
   allocations (§§5.5, 6.3–6.4).** Incorporates
   [Codex's memory finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018698).
   Keeping one backend for every nonlinear segment makes the workspace cost their
   sum, not their maximum. Add the simultaneously held translated tables, retained
   planes, transport/coupling workspaces and any required host/device copies, avoiding
   double counting genuinely shared storage. Otherwise the scan can admit more cells
   than memory can hold. Either adopt that lifecycle explicitly or document a
   bounded-lifetime alternative and its planning cost. All retained runners need
   deterministic cleanup on construction failure, run failure and cancellation.
   A solid-state cell also pays a handoff at every segment boundary, not only once
   per pass as §6.4 suggests. Benchmark a short gas case and a short multi-segment
   case, reporting those actual allocations and transfers.

10. **R10 — P1: Rebuild a pressure-scan launch in its point's medium
    (§5.5).** Incorporates
    [Codex's scan-launch finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018694).
    The example reuses a nominal `launch` after changing `SpatialMedium`. The
    `same_basis` path calls `coupling.port.check_same_medium`, which rejects a
    different admittance; a direct call with otherwise identical metrics 1.0003
    and 1.0006 confirms that refusal. For a matched study, construct the design and
    plane in each point. For a mismatch study, keep the intended nominal beam
    parameters but sample and normalize the plane in that point's medium. State
    what is held fixed: energy/peak axial power, field amplitude, radius and
    curvature are not interchangeable constraints. Do not relabel the old metric
    or insert an artificial interface simply to satisfy the guard. Test both kinds
    of pressure scan through the actual public builder.

11. **R11 — P1: Specify the scan execution contract, especially `Batch`
    (§§3, 5.5, 6.2, 6.5, 11).** This requires more than widening the builder's
    type check. `_scan_batch.py` constructs one backend per `Simulation`, starts
    backend group sessions, and consumes their observations. It is not an arbitrary
    whole-task dispatcher, and a `MultipassSimulation` has no single kernel problem
    to submit there. Define how cell points integrate with it without violating
    the proposed absence of seam/backend changes, or explicitly defer/refuse that
    executor. `SimulationRunner.run` also has no public `stop` argument, whereas
    Local/Pool scans promise interruption of a running point within a step. State
    how cancellation reaches an active segment and how reporters, failures, leases,
    reduction schemas and retained-field choices work for each supported executor.
    Energy/chirp scans must rerun the existing transverse checks and preserve
    material/rate admission when they reuse a runner. Include short tests of every
    admitted execution path and update the cost estimate for the chosen design.
    Finally, settle the public import: §5.2 imports `MultipassSimulation` from
    `solprop.multipass`, but §6.2 assigns that module to P and the class to D.
    Use a driver import or a deliberate public facade without introducing a P-to-D
    dependency.

12. **R12 — P2: Materialize center observations and distinguish center from focus
    (§§2, 4.4, 5.2, 6.4, 7).** Incorporates
    [Codex's center-plane finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018708).
    A `FreeSpace` element has no internal run save planes. If $L/2$ lies in a
    linear ambient gap, split the transport there and retain or measure that
    intermediate plane. If it lies inside a nonlinear segment, translate the
    cell coordinate to that segment's local save position. Specify the observation
    side if it is exactly an interface. Also, §4.4 itself describes two waists
    around a central plate, and asymmetric layouts need not focus at $L/2$.
    Call the fixed observation the center; a claimed focus observable needs a
    defined search and sampling error. Define whether `mirror_radius` uses the
    existing fluence second-moment radius or the $1/e^2$ radius, which diverge for
    an aberrated beam.

13. **R13 — P1: Define physical spatial compression before promising compressed
    durations (§§3, 4.6, 7–8, phase 4).** The integrated spectral energy
    $S(\omega)$ determines a conventional spectral transform limit obtained from
    $\sqrt{S}$, but not the pulse a common compressor can produce when the beam
    has spatially varying spectra or phase. A Fourier probe with two orthogonal
    spatial channels on 4096 FFT-ordered bins, with amplitudes
    $\exp[-(k\mp70)^2/(2\times20^2)]$, gives a synthetic $\sqrt{S}$ peak
    1.99922 times the summed physical peak at the same energy.
    `processing.compression._require_modal` explicitly
    rejects spatial outputs; its array route merely sums squared amplitudes and
    does not supply the missing physical-flux objective. `apply_transfer` takes
    an output, not an arbitrary spectrum as §3 states. Please distinguish the
    integrated-spectrum diagnostic, any on-axis/aperture diagnostic, and compression
    by one common spectral transfer applied to the full field. Define the objective
    and duration from the spatial axial flux, including its frequency-dependent
    admittance, and say which measurement each literature comparison uses. Test a
    separable beam and a beam with spatially varying chirp; the latter must expose
    the difference between the spectral limit and attainable compression.

14. **R14 — P2: Measure the named B-integral and define its sampling error
    (§§4.3, 4.6, 7).** Spatial `peak_intensity` is the maximum over transverse
    nodes and time (`backends/spatial.py`), not the maximum in time on the axis.
    Those disagree for a ring or an aberrated beam. Use `on_axis_intensity` for
    the proposed axis diagnostic and name $\int \max_t I(0,t,z)\,dz$ precisely:
    it is not generally the nonlinear phase of one temporal slice. State the
    effective nonlinear coefficient for vector polarization and delayed responses,
    or restrict this diagnostic to a specified instantaneous scalar model. Join
    sample intervals within each medium, preserving interface sides and avoiding
    trapezoids across a coefficient jump. The center and end alone do not resolve
    a narrow focus, and an adaptive step history alone supplies no quadrature error
    estimate. Specify the sampling/refinement rule and verify its convergence on
    the analytic Gaussian integral before using it to compare matching models.

15. **R15 — P1: Preserve unknown coating phase/delay and test the actual mirror
    map (§§4.1, 4.5, 6.2, 7, Q6).** `ChirpedMirror` data omit constant phase and
    delay; `coupling.transport._reference` marks these terms uncalibrated, and
    `FieldPlane.absolute()` then raises. Thus the unconditional promise that
    `absolute()` is exact over every admitted cell is incompatible with an existing
    supported coating. Specify the required calibration, or retain the unknown
    terms and limit absolute-phase assertions to calibrated models. Restart must
    preserve that status. Nor is the field continuous across a nontrivial mirror
    screen: the coherent check should assert the stated screen/Jones map between
    the two sides, with consistent references. A single flat vector/Cartesian
    mirror still changes handedness and the transverse image; the identity
    handoff test must restrict itself to the scalar symmetric case or undo that
    known map before comparison. Use two reflections and an asymmetric test field
    to check the full convention, and require scalar/vector agreement for a
    linearly polarized input, including the coating's constant phase convention.

16. **R16 — P2: Make the qualification claims match the numerical policy
    (§7).** Restrict the per-frequency energy equality to a linear, lossless
    propagation case with only the specified coating losses: Kerr broadening
    redistributes energy between frequency bins, and absorbers/interfaces need
    their own ledger terms. Across separately integrated runs use an accuracy
    envelope plus transform/transport error bounds, not an unconditional rounding
    budget; use rounding bounds only for matched algebraic operations. Replace
    the scan's promise of identical numerical results by SEP 25's appropriate
    numerical comparison. Add narrowband field/envelope parity through a mirror
    and a plate, and short deliberately mismatched/aberrated launches as well as
    matched Gaussians. Name the tolerance construction for each test class before
    its reference result is known. Pass-by-pass ledgers should distinguish actual
    joule losses from approximation bounds and specify how bounds and cumulative
    boundary removals are combined over the whole cell, rather than resetting
    the effective budget on every relaunch.

17. **R17 — P2: Separate published data, author corrections and model uncertainty
    (§§4.3–4.4, 7, 13).** The public text of [10] uses peak-power wording, places
    Fig. 2 at 2 bar, and separately reports MPC3 compression at 1.2 bar in §2.4.
    The co-author's correction may be right, but the draft's statement about Table 2
    is not independently established by Fig. 2. Please publish a small input table
    distinguishing the published conditions, the explicitly attributed corrections,
    and the conditions selected for each numerical reproduction, with the author's
    confirmation linked in the review response. A discrepancy of 5 % is not a
    general consequence of printing two significant figures. Derive experimental
    envelopes from input/measurement uncertainty separately from numerical error;
    [10] itself quotes an uncertainty of $1.2/9.7$ on its argon nonlinear index.
    A small fit residual in optmat is not automatically the total experimental
    uncertainty. [12] reports transmission with and without its plates and beam
    quality, and the authors have a
    [public dataset](https://doi.org/10.57760/sciencedb.09731); inspect whether these
    constrain the missing inputs before fitting an arbitrary loss or pulse. Choose
    the comparison metric and tolerance before adjusting those inputs, and label
    a tuned reproduction separately from an independent validation. Missing
    information in [13] must not let agreement of one waist ratio alone choose
    the plate closure for Q7.

18. **R18 — P1: Keep the implementation in one PR and budget its total cost
    (§§6.5, 11).** Incorporates
    [Codex's implementation-process finding](https://github.com/LupoLab/solprop/pull/794#discussion_r4563018675).
    SEP 0 requires one feature branch and one draft implementation PR, with phase
    reports accumulated there until `Final`. Make the five phases qualified
    milestones on that branch unless a maintainer explicitly grants an exception;
    splitting them solely to stay below the size budget is not the established
    process. `.github/workflows/size.yml` measures the net Python-file line growth
    against the merge base, including docstrings and other lines, whereas the
    proposal quotes code lines separately. Its 1550 code lines plus 520 docstring
    lines already exceed that PR budget before comments/blanks. Re-estimate the
    whole implementation after the response, scan and output contracts are fixed,
    identify reuse/removals, and leave any `size-ok` decision to the maintainer.

The seven linked Codex threads are incorporated above as R3, R4, R8–R10, R12 and
R18. Resolving those GitHub threads records their transfer into this AI review;
it does not mean that the proposal has already addressed the findings. Please
answer each R-number in the separate response-to-review section required by SEP 0,
with the resulting Part I changes or the reason for a different decision.
