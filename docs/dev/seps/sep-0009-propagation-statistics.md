# SEP 9 — Broadband beam measurements and propagation statistics

| | |
|---|---|
| **SEP** | 9 |
| **Title** | Broadband beam measurements and propagation statistics |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra; GPT-6 (2026-10-22 coordination with SEP 31) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Feature |
| **Created** | 2026-09-27 |
| **Resolution** | — |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Add beam measurements to modal propagation, including both pulse-integrated fluence
widths and time-resolved widths, with a frequency-dependent reconstruction and an
explicit measurement domain. Complete the propagation-statistics catalogue with
per-mode pulse durations, filtered total peak power, general spectral filters,
species pressures and separately named nonlinear-resolution diagnostics. Preserve
existing statistics, defaults and propagation trajectories. The reference is the
reconstructed physical field and its declared flux convention, with analytic tests,
independent spatial and temporal refinement, and parity on NumPy, Numba CPU and
Kokkos CPU. A width is a measurement of a specified distribution, not an effective
area converted into a radius or a universal measure of beam quality.

### 2. Motivation and scope

A multimode fibre can change its transverse light distribution without changing its
total energy or even its modal energies: relative modal phases alone change the
interference pattern. A mode's profile can also change across a broadband spectrum,
and different parts of a pulse can occupy different transverse regions. The useful
questions are therefore distinct:

- How wide is the energy distribution integrated over the complete pulse?
- How wide is the power distribution at each resolved time within it?
- How do those answers change when a spectral band is selected?
- Is the field centred, elliptical, hollow or split into several lobes?
- Is an apparent change physical, or a consequence of too few modes, too few
  integration nodes, insufficient temporal sampling or a clipped measurement domain?

Both fluence and time-resolved widths are required deliverables. Spectral selection
and time resolution are independent: a band-limited width is not a time slice.
The location is the guide cross-section at the stated propagation position. These
are not measurements after outcoupling, imaging optics or free-space propagation.

The proposal covers ordinary modal routes with an actual transverse profile and the
existing general modal route with its frequency-dependent E/H reference. The latter
is qualified only on combinations the propagation route itself admits; this SEP does
not add general-modal envelope propagation. A dispersion-only GNLSE declaration has
no spatial shape. It can supply a declared effective-area intensity, but cannot
supply a beam width. Scalar and vector profiles, taper motion, material regions,
modal coherence and saved-field amplitude conventions are in scope.

Existing radial, Cartesian, line and hybrid planar statistics remain supported;
their common flux measure follows [SEP 31](sep-0031-simulation-coupling.md)'s
uniform-medium correction. Adding further time-resolved spatial measurements
is outside this SEP. SEP 31 also owns transported observation-plane fields;
the measurements here describe fields inside the propagation route.
No new detector material model, polarisation analyser, finite-exposure response,
noise ensemble, beam-quality propagation ratio $M^2$, mode solver, response model or
adaptive numerical controller is proposed. A width along a fibre is not necessarily
a free-space beam waist. No new dependency is required.

### 3. Context

#### 3.1 Capability audit

The following inventory separates absent measurements from existing equivalents and
deliberate choices. It is an inventory of capabilities, not a request to reproduce
another implementation's function names or reductions. The current definitions are
in [statistics](../../guide/results/statistics.md), [power and intensity](../../guide/results/power-and-intensity.md),
`src/solprop/stats.py`, `src/solprop/spatial.py` and the backend statistics methods.

| Measurement or facility | Present behaviour | Disposition in this SEP |
|---|---|---|
| Spectral centroid | `omega0`, per mode and polarisation, with power/metric weighting | Keep; no missing centroid statistic |
| Energy | `energy`; `physical_energy` on a general modal basis | Keep the route-specific meanings |
| Energy in wavelength/frequency intervals | `energy_window`, half-open angular-frequency intervals | Keep exact tiling; add a wavelength constructor for explicit filters |
| Energy through an arbitrary spectral window | Hard intervals only in run statistics | Add a labelled complex amplitude-transfer specification and `energy_filtered` |
| Peak power per mode and of the simultaneous sum | `peak_power`, `peak_power_total` | Keep; sum traces before taking the total peak |
| Filtered peak power | `peak_power_window` per mode for hard intervals | Add `peak_power_total_window` and general filtered counterparts |
| Inner and outer temporal FWHM of the total pulse | `fwhm_t`, `fwhm_t_outer` | Keep their scalar shapes |
| Inner and outer temporal FWHM per mode | No catalogue entry; the total widths do not replace these | Add `fwhm_t_modes`, `fwhm_t_outer_modes` |
| Effective-area average peak intensity | Can be computed from peak power and an area, but has no distinct statistic | Add explicit `effective_peak_intensity` for one mode with a declared area |
| On-axis modal peak intensity | `peak_intensity`, using the current analytic Poynting closure | Keep; do not silently change it to a transverse maximum |
| Transverse maximum modal intensity | Absent; an informal roadmap item | Add `peak_intensity_transverse` on the declared diagnostic domain |
| Fibre beam widths | `fwhm_r` and `beam_radius` exist for spatial geometries, not modal reconstruction | Add modal fluence and time-resolved width families |
| Electron density and ionisation rate | Backend statistics already exist, with additional ordered species statistics | Keep their population and sampling conventions |
| Gas density and total pressure | Profile statistics; ordered `species_density` also exists | Keep |
| Pressure of each constituent | No ordered species-pressure statistic | Add `species_pressure` with an explicit EOS interpretation |
| Core radius and ZDW | Existing opt-in profile and modal statistics | Keep; multiple-ZDW root selection remains a separate known issue |
| Position and step length | Saved-plane positions, periodic-row positions and indexed step history | Already present; no duplicate `z`/`dz` statistics |
| Relative transverse integration error | `quadrature_error`, a cached embedded-rule alarm | Keep; it does not measure modal truncation |
| Absolute and band-selected integration residuals | No catalogue entries | Add fresh, per-member nonlinear quadrature diagnostics with stated norms |
| Nonlinear source reconstruction in a retained modal space | Not implemented | Add an opt-in spatial residual; do not equate it to quadrature error |
| Number of transverse evaluation points | Fixed rule extents are available from the problem | Record rule/probe counts as diagnostic metadata, not a fictitious adaptive error |
| Collector, custom closures, device traits, analytic transform plans | `StatsSpec`, one catalogue, backend seam and periodic channel | Infrastructure already exists; arbitrary executable callbacks are not new observables |

The ionisation comparison needs particular care. Ordinary multimode plasma is
sampled at the nonlinear nodes; the single-channel density/rate pair refers to the
densest final-population node. The mixture path additionally exposes each species'
largest rate over nodes and time. These are deliberately different from measuring
only on the guide axis. They are not missing statistics to be replaced. Likewise,
the established peak intensity is not $P/A_{\rm eff}$.

#### 3.2 Existing reconstruction and execution

The ordinary `Mode.field(points, z)` interface has no frequency argument.
`modes.quadrature.mode_matrix` uses a vacuum-normalised transverse profile.
In particular, `StepIndexMode` declares a profile wavelength: its dispersive
propagation constant does not make that frozen transverse profile dispersive.
A diagnostic of this route must retain that approximation and name it. It must
not solve new profiles at other frequencies and apply them to amplitudes propagated
in a different basis.

The [general modal route](../../guide/waveguides/general-modal.md) already carries tracked,
complex, frequency-dependent E/H fields through
`translate.modal_tables.ModalTransformTables`. `processing.modal` checks the guide,
gauge and table identities and reconstructs a plane, including a moving guide's
nodes. Its `reconstructed_flux` uses the reconstructed cross-frequency E/H product.
This is different from the general route's run statistic `analytic_flux` (the two
shared that name until the accessor was renamed), which is formed from
metric-normalised amplitudes. Their energy integrals agree;
their time traces need not. New beam measurements must use the reconstructed
product and must not reinterpret the existing statistic.

`backends.numpy_ref.backend`, `backends.numba_cpu.backend`, `backends.spatial` and
`backends.driver.StepDriver` own the reference, compiled and periodic evaluation.
In `solprop-kokkos`, `src/solprop_kokkos/backend.py` packs/unpacks native statistics;
`src/general.cpp` owns the modal statistics pass, and `src/general_modal.cpp` the
general modal execution. The current Kokkos temporal FWHM path transfers a total
power trace for host crossing interpolation; a new family of per-mode/time-resolved
measurements must budget that transfer rather than assume all statistics are scalars.

The design builds on ADR-010 (quadrature), ADR-012 (statistics scheduling), ADR-017
(Fourier boundary), ADR-030 (power normalisation), the
[kernel contract](../kernel-contract.md) and the capability gates in
`output.capability` and `stats.GENERAL_MODAL_ACCESSORS`. It requires none of the
other draft physics SEPs. Statistics observe the accepted state after boundary
treatments; cached stage arrays are reusable only when they represent that same
state, position, noise convention, filter and sampling grid.
Beam measurements use the propagating field, including input noise when it is
part of that field. They do not silently add the modified-noise drive that is
present only in a nonlinear-response evaluation. A source diagnostic, in contrast,
uses the run's declared nonlinear-drive noise convention and records that scope.

### 4. Theory

#### 4.1 The measured field and the spectral filter

Use SI and the optics convention of [conventions](../conventions.md): inverse
transforms carry $e^{-i\omega t}$. At a fixed $z$, write the analytic fields as

$$
\widetilde{\mathbf E}(\mathbf r,\omega)
 =\sum_m a_m(\omega)\mathbf e_m(\mathbf r,\omega),\qquad
\widetilde{\mathbf H}(\mathbf r,\omega)
 =\sum_m a_m(\omega)\mathbf h_m(\mathbf r,\omega).
$$

The channel index here includes any declared polarisation component. The E/H
profiles have units V/(m $\sqrt{\mathrm W}$) and A/(m $\sqrt{\mathrm W}$) on a
unit-power reference. In an ordinary modal route the internal amplitude is instead
the existing vacuum-normalised $a$; its magnetic equivalent is described below.
Complex profiles and amplitudes must be expressed in the same Fourier convention
and tracked gauge. Conjugating a saved amplitude without converting its profile
reference is not a physical reconstruction.

Let $T_j(\omega)$ be a dimensionless scalar amplitude transfer, applied to **both**
fields before reconstruction. The unfiltered request is $T_0=1$ over the represented
active spectrum. The filtered flux is

$$
S_j(\mathbf r,t)=\frac12\operatorname{Re}
 [\mathbf E_j(\mathbf r,t)\times\mathbf H_j(\mathbf r,t)^*]
 \cdot\hat{\mathbf z},\qquad [S_j]=\mathrm{W/m^2}.
$$

This is the analytic Poynting flux, with the real instantaneous sum-frequency term
excluded. It is not $|E|^2$ in an arbitrarily chosen refractive index and does not
include a detector's finite response time. Only transverse components enter the
axial cross product, although the full vector reference must retain its longitudinal
components for consistency and nonlinear-source diagnostics.

For ordinary profiles $\mathbf g_m=\mathbf e_m/\sqrt{N_m}$, where
$N_m=\varepsilon_0c\int|\mathbf e_m|^2dA/2$, retain the existing closure

$$
\widetilde{\mathbf E}_k=\sum_m a_{mk}\mathbf g_m,\quad
\widetilde{\mathbf E}_{w,k}=\sum_m w_{mk}a_{mk}\mathbf g_m,\quad
S_j=\frac{\varepsilon_0c}{2}\operatorname{Re}
 [\mathbf E_j\cdot\mathbf E_{w,j}^*],\quad
w_{mk}=\operatorname{Re}n_{{\rm eff},m}(\omega_k,z).
$$

The route's explicit index override remains authoritative. Reconstructing a saved
true-power amplitude requires $a=b/\sqrt w$ first. Even at one frequency, local
intermodal terms carry the arithmetic mean $(w_m+w_n)/2$; replacing $a$ with $b$
in the vacuum synthesis would give geometric-mean weights. This is an observable
difference in an interference pattern. The ordinary closure is an approximation of
that propagation model, not a substitute for the general route's actual H fields.

#### 4.2 Full-pulse fluence and exact discrete normalisation

The fluence through the plane is

$$
F_j(\mathbf r)=\int S_j(\mathbf r,t)\,dt
 =\frac{1}{4\pi}\operatorname{Re}\int
 |T_j(\omega)|^2
 [\widetilde{\mathbf E}(\mathbf r,\omega)
 \times\widetilde{\mathbf H}(\mathbf r,\omega)^*]_z\,d\omega,
\qquad [F_j]=\mathrm{J/m^2}.
$$

The continuous expression uses analytic positive-frequency spectra. In an envelope
representation the same integral uses the represented detuning band, with physical
frequency retained for profiles and filters. Parseval removes cross-frequency terms
only after integration over the complete temporal window; coherent cross-mode
terms at each frequency remain. Thus common spectral phase changes do not change
fluence, but can change time-resolved widths. Relative modal spectral phase can
change both.

For the raw DFT arrays, let $N=n_t$ and let $d_k=1$ on an envelope grid. On a field
grid $d_k=2$ for interior positive-frequency bins and $d_k=1$ at DC and Nyquist,
with the endpoint reality constraints of the analytic-signal construction. Then

$$
\eta_k=\frac{\Delta t}{N}d_k^2,\qquad
F_j(\mathbf r)=\frac12\sum_k\eta_k|T_{jk}|^2
 \operatorname{Re}[\mathbf E_k(\mathbf r)\times\mathbf H_k(\mathbf r)^*]_z.
$$

This is the discrete definition. The ordinary closure replaces the half cross
product by $\varepsilon_0c\operatorname{Re}(\mathbf E_k\cdot\mathbf E_{w,k}^*)/2$.
No Simpson weights or frequency-centroid approximation are added. The integration
measure is the physical area, including $r\,dr\,d\theta$ on polar rules. A radial
line profile itself is **not** multiplied by $r$ before a half-maximum search.

Integrating the reconstructed fluence over the full represented domain must recover
the route's energy to the independently qualified spatial/table error. For a
general basis that energy is
$\sum_k\eta_k a_k^\dagger W_k a_k$, not a sum of coordinate powers.
This is the central normalisation test. A core-only aperture is a different
measurement and reports its captured energy; it must not be renormalised and
presented as the whole beam. Evanescent cladding tails matter especially to second
moments, so energy convergence alone does not certify a width.

#### 4.3 When a coherency-matrix reduction is exact

At each frequency the local flux is a Hermitian quadratic form
$a_k^\dagger Q_k(\mathbf r)a_k$. If
$Q_k(\mathbf r)=\sum_{\ell=1}^L q_{\ell k}Q_\ell(\mathbf r)$ with a certified
separation, spectral accumulation can be performed first:

$$
C_{j\ell}=\sum_k\eta_k|T_{jk}|^2q_{\ell k}a_ka_k^\dagger,
\qquad F_j(\mathbf r)=\sum_\ell\operatorname{tr}[Q_\ell(\mathbf r)C_{j\ell}].
$$

For the ordinary fixed-profile closure an exact special case is

$$
C_{j,mn}=\sum_k\eta_k|T_{jk}|^2
 \frac{w_{mk}+w_{nk}}2\,a_{mk}^*a_{nk},\qquad
F_j=\frac{\varepsilon_0c}{2}\sum_{mn}
C_{j,mn}\,\mathbf g_m^*\cdot\mathbf g_n.
$$

The second expression deliberately uses the conjugate orientation to the first
matrix convention; its indices state the contraction explicitly. Tests must use
complex profiles and nontrivial phases, since real profiles can conceal a misplaced
conjugation. All quantities are evaluated at the measurement's $z$.

A single unweighted matrix is valid only when the spatial kernel is independent
of frequency up to a known scalar. Frequency-dependent profile shapes, mode-index
weights or detector response cannot be discarded. In particular, evaluating every
profile at the pulse's centroid frequency is not broadband reconstruction. The
general implementation synthesises at each frequency; a low-rank approximation
needs its own certified error and is not required by this SEP.

#### 4.4 Time-resolved measurements

For a requested time, use $D_j(\mathbf r)=S_j(\mathbf r,t)$ in the same spatial
measurement operator used for $D_j=F_j$. In continuous notation the temporal
product includes the kernel

$$
S_j(\mathbf r,t)=\frac{1}{2(2\pi)^2}\operatorname{Re}
\iint T_j(\omega)T_j(\omega')^*
[\widetilde{\mathbf E}(\mathbf r,\omega)
 \times\widetilde{\mathbf H}(\mathbf r,\omega')^*]_z
e^{-i(\omega-\omega')t}\,d\omega\,d\omega'.
$$

Do not replace this by independent spectral intensities or a same-frequency metric
square root. Implement it by synthesising E/H spectra and inverse-transforming
them, not by materialising the double-frequency kernel. A frequency-independent
profile allows temporal modal synthesis first; a dispersive profile generally does
not. On envelope grids the removed carrier cancels in the Hermitian product.

Times are in seconds in the run's co-moving frame, with its existing time origin.
An integer oversampling factor selects a zero-padded temporal grid, preserving
physical amplitudes and the treatment of field-grid endpoints. Requested samples
are an explicit subset of that grid; no periodic wrapping of an out-of-window time
is allowed. Omitting the subset means all times on that grid when a temporal
statistic is requested; a fluence-only request allocates no temporal output.
Zero padding refines an already represented field and cannot recover
unrepresented optical bandwidth. Sampling must resolve the difference frequencies
of the flux, not merely a convenient set of pulse times. Temporal convergence is
qualified independently of propagation error and transverse sampling.

In general,

$$
\operatorname{width}\!\left(\int S_j\,dt\right)
\ne\int \operatorname{width}(S_j)\,dt,
$$

and neither side is a width measured at the instant of peak total power. All these
operations are nonlinear once centring, normalisation or a threshold is introduced.
The delivered time series must therefore include the time axis and the power through
the measurement aperture so that a width during a negligible tail is identifiable.

#### 4.5 Centroids, second moments and half-maximum widths

For an admissible distribution $D$ over the declared aperture $A$, set

$$
U_D=\int_A D\,dA,\quad
\boldsymbol\mu_D=\frac{\int_A\mathbf rD\,dA}{U_D},\quad
\Sigma_D=\frac{\int_A(\mathbf r-\boldsymbol\mu_D)
(\mathbf r-\boldsymbol\mu_D)^T D\,dA}{U_D}.
$$

$U_D$ is energy for fluence and power for a time slice. Centroids are in m and the
covariance is in m². With eigenvalues $\lambda_1\ge\lambda_2$, report second-moment
diameters $d_i=4\sqrt{\lambda_i}$, and the optional equivalent radial moment
$r_2=\sqrt{2\operatorname{tr}\Sigma_D}$. This is the usual second-moment beam-width
convention [1, 2], applied here to the explicitly specified energy or power
distribution; no claim of a complete ISO measurement procedure is made.
Keep the covariance itself so a circular beam does not need an arbitrary orientation.

For the axis-aligned Gaussian
$D=D_0\exp[-2(x^2/w_x^2+y^2/w_y^2)]$, direct Gaussian integration gives
$\Sigma_{xx}=w_x^2/4$, $\Sigma_{yy}=w_y^2/4$, and $d_i=2w_i$.
On a circular Gaussian $r_2=w$ and the radial full width at half maximum is
$w\sqrt{2\ln2}$. Radius, diameter and FWHM must never share an unlabeled output.

For a general modal beam define `fwhm_r` from the **azimuthal average about the
fluence centroid**, or the slice's own centroid for `fwhm_r_time`:

$$
\overline D(r)=\frac1{2\pi}\int_0^{2\pi}
D(\boldsymbol\mu_D+r(\cos\theta,\sin\theta))\,d\theta.
$$

Use half the maximum of this averaged profile over the certified sampled domain.
Starting at $r=0$, take the first outward bracketed crossing and return twice its
radius. No search goes beyond the aperture's largest complete centred circle.
If the central value is below the threshold, as in a sufficiently hollow beam,
the central FWHM is undefined. This is a radialised width, explicitly labelled as
such; it is not a line cut at one azimuth and is not a principal-axis diameter.
Centroid and covariance expose the information this scalar removes. The existing
spatial Cartesian width uses a related radial average; its implementation and
defaults are not changed here.

Use linear interpolation between ordered bracketing samples and refine the
diagnostic radius/angle samples independently of the nonlinear quadrature. A
resolved ring can have finite second moments and undefined central FWHM. Multiple
crossings, tangent crossings and shallow slopes need explicit resolution flags;
a root finder alone does not establish that the correct lobe was resolved.

Analytic flux, including local fluence, is not guaranteed positive everywhere for
an arbitrary coherent modal superposition. The existing convention retains its
sign. A width interprets $D/U_D$ as a positive spatial distribution, so an appreciable
negative region, nonpositive total, unresolved domain or a dark time slice cannot
silently become a beam width. Return NaN for the affected width and a reason code;
retain the signed aperture energy/power. Nonfinite samples raise instead. Do not clip
the flux or replace it by $|E|^2$. An operation-count bound distinguishes round-off-sized
negative values from physical signed flux; sums still use the signed samples.
Only eigenvalues consistent with zero within the propagated round-off bound may be
rounded to zero for a square root. A user-visible relative signal floor can mark
weak temporal slices; its value is stored and does not alter the field or energy.

#### 4.6 Remaining scalar statistics

Per-mode temporal widths reuse the current ordinary-route trace
$P_m(t)=\sum_p\operatorname{Re}[a_{mp}(t)a_{w,mp}(t)^*]$ and its existing inner/outer
crossing rules. They sum polarisations within a spatial mode. No basis-independent
per-mode energy or duration is promised for a general nonorthogonal basis; these
names are refused there, preserving the existing capability distinction.

Apply a filter before forming either member of the flux product. For ordinary
modal filtered peaks,

$$
P_{j,m}^{\rm peak}=\max_t P_{j,m}(t),\qquad
P_{j,\rm total}^{\rm peak}=\max_t\sum_m P_{j,m}(t).
$$

Taking the largest mode peak or adding peaks attained at different times is wrong.
An amplitude transfer $T$ weights spectral energy by $|T|^2$, not $T$; phase affects
filtered temporal peaks even though it does not affect filtered energy. Hard
half-open `energy_windows` retain their exact-tiling semantics. Arbitrary overlapping
filters have no such sum rule. Wavelength intervals map monotonically to reversed
frequency endpoints, $\omega_{\rm lo}=2\pi c/\lambda_{\rm hi}$ and
$\omega_{\rm hi}=2\pi c/\lambda_{\rm lo}$. No wavelength-density Jacobian belongs
in a transfer function multiplying a spectrum already sampled in frequency.

`effective_peak_intensity` is explicitly $\max_t P(t)/A_{\rm eff}(z)$, W/m²,
only for a single mode with a finite positive declared effective area. It is a
nonlinear effective-area diagnostic, not the spatial maximum. An area alone gives
neither the field profile nor its width; for a broadband frozen-profile model the
area retains that model's reference-wavelength approximation.

`peak_intensity_transverse` is $\max_{q,t}S(q,t)$ on the recorded diagnostic nodes
and temporal grid, with the position/time of the sampled maximum available in
metadata or companion outputs. It is a sampled maximum, whose convergence must be
tested. Its temporal search covers the complete reconstructed grid, independently
of a subset selected for width output. It does not rename or change modal
`peak_intensity`.

`species_pressure` reports an ordered vector in Pa, following the SI rule for new
quantities. The existing `pressure` output in bar remains unchanged. For a shared
fill, report the partial-pressure paths already declared by that mixture's EOS
construction. For independently supplied number-density paths, evaluate each
constituent's own EOS at its local density and temperature and label the result
`partial_eos`. Do not assume room temperature or claim that the sum is a
thermodynamic mixture pressure outside the mixture model's domain. The ideal-gas
limit is $p_s=\rho_s k_BT$; this SEP introduces no new mixing law or constants.

#### 4.7 Quadrature resolution and modal truncation are different

The existing `quadrature_error` compares the fine and embedded coarse projections
of a stage's source and retains a cached alarm. It is deliberately not an on-axis
source reconstruction, not an error of the accepted field and not a controller
input. Preserve that meaning, including undefined cases with no embedded rule.

New plane diagnostics evaluate the accepted state with the same physical response,
noise treatment and temporal/spectral windows as the run, using diagnostic buffers.
Let $N_f$ and $N_c$ be fine/coarse nonlinear drives in the **lab-frame internal
amplitude derivative**, after physical response factors and spectral masking, with
the linear term and numerical absorber excluded. For filter $T_j$, define

$$
\|v\|_{j,W}^2=\sum_k\eta_k|T_{jk}|^2v_k^\dagger W_kv_k,\quad
R_{j,\rm quad}=\|N_f-N_c\|_{j,W},\quad
e_{j,\rm quad}=R_{j,\rm quad}/\|N_f\|_{j,W}.
$$

Use the ordinary diagonal power weight or the general route's flux metric as
appropriate. $R_{j,\rm quad}$ has units $\sqrt{\mathrm J}/\mathrm m$, since the
drive is an amplitude derivative. This is a norm, not an RMS of arbitrarily scaled
DFT coefficients. A vanishing source has zero absolute residual and undefined
relative residual; missing coarse quadrature is separately unavailable. A general
guide currently has no embedded coarse rule: it must refuse these requests by name
until one is supplied and independently qualified. Do not fabricate a zero error.
Band-selected residuals reveal poorly resolved weak emission hidden by a large pump.

For modal truncation use a separately defined source-space projection. The first
implementation covers instantaneous and delayed **polarisation** responses, ordinary
and general modal. At every retained frequency, sample the physical nonlinear
polarisation $p_k$ (C/m²) on a diagnostic rule $V$ of positive area weights. If $B_k$
contains the electric profile columns in the propagated span, the weighted least
squares reconstruction is

$$
\widehat p_k=B_k(B_k^\dagger VB_k)^{-1}B_k^\dagger Vp_k,\qquad
e_{j,\rm span}^2=
\frac{\sum_k\eta_k|T_{jk}|^2\|p_k-\widehat p_k\|_V^2}
     {\sum_k\eta_k|T_{jk}|^2\|p_k\|_V^2}.
$$

Compute the projection with a rank-revealing weighted QR/SVD or an equivalently
conditioned factorisation, not an explicit inverse. Use a separate validation rule
to qualify sampling; projecting and testing only at interpolation nodes can
conceal missing spatial structure. The result measures representability of the
polarisation in the retained electric-profile span under a stated norm. It is
invariant under a nonsingular change of coordinates within that span, but it is
not an emitted-energy fraction, not a UPPE reciprocal-source projection, and not
an error bound on propagation. A pointwise on-axis residual would miss modes
vanishing there and is not the recommended diagnostic.

Plasma/current responses require a distinct source-space convention and are
explicitly deferred; a polarisation-only residual must refuse a request to label
it as the total nonlinear source in such a run. Converting current to polarisation
by dividing by frequency is not introduced implicitly. Adding it requires a
defined treatment of DC, response factors and the measured norm. Quadrature
diagnostics can still use the run's complete assembled nonlinear drive, including
plasma. Source-evaluation node counts, projection-rule counts and validation-rule
counts are recorded separately; none is evidence of convergence on its own.

### 5. Proposed interface

The names below are proposed, not available APIs. Extend the existing frozen
`StatsSpec` rather than add a second callback/collector system:

- `filters`: an ordered tuple of frozen `SpectralFilter` records, with a unique
  label and finite complex amplitude-transfer samples on the run's coarse frequency
  axis. Preserve samples and their grid identity. A wavelength-band constructor
  takes positive finite endpoints in m and an explicit transition width in rad/s;
  zero means a hard half-open band, positive means the documented Planck taper.
  No arbitrary callable crosses the backend seam. Transfers need not be passive;
  values above unity are recorded as gain rather than called an energy fraction.
  On field grids a transfer must preserve real DC/Nyquist values where those bins
  are nonzero; inactive bins stay zero. Reject an incompatible endpoint transfer.
- `beam`: an optional frozen `BeamMeasurement` describing the physical aperture,
  integration and profile-sampling rules, full-domain or finite-aperture intent,
  refinement/error settings, optional temporal sample indices, integer temporal
  oversampling factor $\ge1$, and relative signal floor in $[0,1)$. The default
  floor is zero apart from the derived numerical floor. The modal reference comes
  from the assembled route; the record cannot substitute an unrelated profile.
- `nonlinear_diagnostics`: an optional frozen specification of the fresh plane
  diagnostics, independent validation rule and response scope.

Use the first filter slot for the explicit unfiltered transfer. Shapes always keep
the filter axis, including a one-filter request; do not squeeze singleton axes in
the stored statistics. Here `n_filters` and `n_stat_times` are new documented axes;
batch and saved-plane axes are added by the existing infrastructure.

| Proposed statistic | Shape per plane | Units / meaning |
|---|---|---|
| `fwhm_t_modes`, `fwhm_t_outer_modes` | `(n_modes,)` | s; ordinary modal routes |
| `peak_power_total_window` | `(n_windows,)` | W; existing hard windows |
| `energy_filtered` | `(n_filters, n_modes, n_pol)` | J; ordinary modal routes |
| `physical_energy_filtered` | `(n_filters,)` | J; general modal metric |
| `peak_power_filtered` | `(n_filters, n_modes)` | W; ordinary modal trace |
| `peak_power_total_filtered` | `(n_filters,)` | W; ordinary modal trace |
| `effective_peak_intensity` | `()` | W/m²; one declared effective area |
| `species_pressure` | `(n_species,)` | Pa; ordered, with EOS interpretation |
| `beam_energy` | `(n_filters,)` | J through the declared aperture |
| `beam_centroid` | `(n_filters, 2)` | m; transverse Cartesian coordinates |
| `beam_covariance` | `(n_filters, 2, 2)` | m²; central second moments |
| `beam_diameters` | `(n_filters, 2)` | m; descending principal $4\sigma$ diameters |
| `beam_fwhm_r` | `(n_filters,)` | m; central radialised fluence FWHM |
| `beam_power_time` | `(n_filters, n_stat_times)` | W; signed aperture power |
| `beam_centroid_time`, `beam_diameters_time` | `(n_filters, n_stat_times, 2)` | m; each slice centred independently |
| `beam_covariance_time` | `(n_filters, n_stat_times, 2, 2)` | m² |
| `beam_fwhm_r_time` | `(n_filters, n_stat_times)` | m; central radialised slice FWHM |
| `peak_intensity_transverse` | `(n_filters,)` | W/m²; sampled spatial/temporal maximum |
| `nonlinear_quadrature_residual` | `(n_filters,)` | $\sqrt{\mathrm J}/\mathrm m$; fresh accepted-state norm |
| `nonlinear_quadrature_error` | `(n_filters,)` | dimensionless; fresh relative norm |
| `polarisation_span_error` | `(n_filters,)` | dimensionless; retained-span residual |

Each beam width/moment family stores validity codes with matching filter/time axes;
FWHM has a distinct validity code because it can fail when moments remain valid.
The vocabulary includes `valid`, `empty`, `below_signal_floor`, `signed_flux`,
`no_central_lobe`, `unbracketed`, `unresolved` and `degenerate_orientation` where
orientation is requested. Nonfinite input or missing required physics is an error,
not a successful measurement carrying an undefined-width code. The time axis is
stored once as `stats_axes/time`, and the filter order/labels once in metadata.

`beam_fwhm_r` is the explicit filtered family; §4's modal meaning of `fwhm_r` is
exposed as an unfiltered scalar convenience using that same evaluator. An ordinary
modal `beam_radius` request similarly returns the existing two-element layout
(second-moment radius and central $1/e^2$ radius), from the same radial profile.
These two legacy names remain plane-only. All existing spatial shapes and meanings
remain unchanged. No two separate implementations of a width are introduced.

Example sketch, with the proposed convenience constructors:

```text
StatsSpec(
    names=("energy", "fwhm_t_modes", "peak_power_total_window",
           "beam_diameters", "beam_fwhm_r", "beam_diameters_time",
           "beam_fwhm_r_time", "beam_power_time"),
    energy_windows=frequency_intervals,
    filters=(SpectralFilter.all(grid),
             SpectralFilter.wavelength_band(grid, (700e-9, 900e-9),
                                            label="pump", transition=2e13)),
    beam=BeamMeasurement(
        aperture=measurement_aperture,
        quadrature=measurement_rule,
        profile_sampling=radial_and_angular_samples,
        time_indices=selected_indices,
        time_oversampling=2,
    ),
)
```

Run-time statistics and offline processing share the same measurement definition.
Offline entry points require the saved amplitude/power convention and the route's
validated reconstruction reference, using the existing guide/table identity checks.
They can return nodal fluence/flux for inspection; the run normally stores only
reductions. A general guide cannot pass an ordinary `modes=` tuple and bypass
these checks. Unsupported routes, filters, missing profiles, unresolved modal
cutoffs, incorrect gauges or out-of-domain samples fail before propagation.

Cheap scalar additions can join the existing periodic subset after qualification.
The beam families, time series, transverse maxima and fresh nonlinear diagnostics
are opt-in at save planes in this proposal. They are not silently enabled at every
accepted step. Users can request closer save planes; a separately benchmarked
between-plane admission can be proposed later. This preserves the current cost
contract and avoids treating a potentially large time series as a cheap scalar.

### 6. Implementation strategies

#### 6.1 Recommended layers and data contract

Keep specifications, named reference reductions and validity semantics in
`stats.py` and the spatial measurement domain in `spatial.py`, with modal physics
owned by `modes/`. Translation compiles a separate diagnostic reconstruction record:
physical coordinates/area weights, aperture identity, E/H maps or certified fixed
profile factors, temporal sampling, filters and their validity domain. It must not
reuse a one-node effective-area transform as if it were a spatial field.
General modal propagation tables remain the authority for channel identity and
gauge. If diagnostic nodes differ, translation solves/evaluates profiles there and
aligns them to that authority; a fresh provider solve without alignment is invalid.
For a moving guide both profile and physical node coordinates belong to the plane.

Backends evaluate only validated arrays, in node chunks, and return host reductions.
No new backend dependence on a mode solver or EOS is introduced. Driver/profile
statistics own constituent pressures. Processing uses the same definition and
identity checks as the run, with offline arrays available for independent inspection.
Capability metadata must publish these new names only where the reference and
backend actually support them; update all admission doors, including general-modal
files, scans and batches.

Adding the diagnostic record and statistics parameters changes the seam. Bump
`SCHEMA_VERSION` from the then-current value in the first implementation phase
that carries them, update the literal contract tests and the Kokkos consumed-field
tuples, and record it in the changelog. Do not preselect a version number in this
Draft. Any later phase changing protected driver hooks or stage bundles repeats
that rule. The new fixed axes, filter data, validity codes and reference/aperture
identities require an output-schema revision with unsupported files refused and resume
hashes accounting for every measurement-defining field.

#### 6.2 Alternatives and costs

| Strategy | Cost and limitation | Decision |
|---|---|---|
| Reconstruct after saving the full field | Easy to inspect; misses runs that retain only statistics and transfers/stores the full state | Provide offline parity, but not as a hidden device fallback |
| One modal coherency matrix | About $O(n_\omega M^2+Q M^2)$ per filter; exact for the certified separable kernel | Use for ordinary fluence when cheaper than direct synthesis |
| Frequency-resolved node synthesis | $O(Q M n_\omega)$ per filter, plus reductions; retains arbitrary tracked profiles | NumPy oracle and required general-modal path |
| Time reconstruction in node chunks | Add $O(Q n_t'\log n_t')$ per filter, where $n_t'$ is the padded grid | Required for time-resolved widths and transverse peak intensity |
| Precomputed moment matrices | Can avoid node/time fields for some integrated moments; frequency-pair matrices cost $O(M^2n_\omega^2)$ | Optional separable optimization only; never materialise the general pair kernel |
| Approximate spectral low rank | May reduce synthesis cost; introduces a new approximation near cutoffs and cancellations | Defer pending an independent error certificate |

For fixed profile maps, persistent storage can be $O(QM)$ plus a coherency matrix.
Frequency-resolved maps can cost $O(QMn_\omega)$; do not copy a dense reference a
second time merely for statistics. Budget streamed/table-evaluated maps and a node
chunk's E/H buffers explicitly. Temporal reconstruction scratch scales with
$Q_{\rm chunk}n_t'$, not the full cross-section times the full time window.
Filters may be processed sequentially with shared maps. Reductions of moments
over nodes can accumulate one time vector per moment; radial FWHM needs a separately
planned profile pass or bounded profile buffers. Recomputing that pass is an honest
time/memory tradeoff, not an unlisted allocation.

Returned time-series storage grows as $O(n_z n_{\rm filters}n_{\rm stat\_times})$
times the selected moments. Native crossing reductions are preferable when large
per-mode or temporal traces would otherwise cross the device seam. Small reduced
radial profiles may be transferred for the one reference interpolation algorithm
if their size is admitted in the plan. Record actual transfer volumes. Neither the
full state nor full nodal E/H fields are copied to implement an advertised native
run statistic.

The fresh source diagnostics can cost another nonlinear evaluation at a plane,
and the span test adds source sampling and a conditioned projection. Keep them
opt-in and separate from the cached alarm. Partial reductions within a rounding
bound fixed before measurement, no kernel allocations, no fast-math, no hidden response-state mutation, and complete
memory-ledger entries apply to all three backends. No cost percentage is promised
before measurements on the required short windows.

### 7. Verification and validation

Use analytic fields and independently evaluated spatial integrals as the first
reference. Backend agreement alone is insufficient. Each test states its error
budget before comparing results; no tolerance is selected from a measured failure.

1. **Parseval and flux normalisation.** Directly integrate reconstructed time-domain
   E/H flux and compare to the same-frequency fluence sum on both grids, then
   integrate spatially against the route's energy. Exercise isolated bins, legal
   DC/Nyquist endpoints, complex profiles and index weights differing by mode and
   frequency. Use deterministic accumulation/FFT round-off bounds plus separately
   measured modal-table and transverse-integration errors.
2. **Gaussian widths.** Compare centroid, covariance, $4\sigma$ diameters and radial
   FWHM with §4.5 for circular and rotated elliptical Gaussians. Translation of a
   beam moves its centroid without changing its covariance eigenvalues. Scale the
   amplitude over a wide finite range without changing widths. Linear crossing
   interpolation is second order for a smooth nonzero-slope crossing: test three
   spatial refinement levels with the expected order, allowing the independently
   derived sampling error.
3. **A single capillary mode and a taper.** Independently integrate the squared
   Bessel profile with adaptive SciPy quadrature; bracket the central crossing
   independently. With a self-similar taper, lengths scale with core radius and
   covariance with its square. Compare at the actual plane, not the entrance.
4. **Coherent multimode beating.** Two orthogonal modes with a fixed spectrum and
   a changing relative phase have constant modal energies and a changing spatial
   pattern. Compare direct synthesis with the coherency contraction, including
   complex polarisation partners and a mode that vanishes on axis. Verify that
   discarding cross terms fails the constructed case.
5. **Frequency-dependent profiles.** A synthetic two-band E/H family with distinct
   Gaussian radii has a broadband covariance equal to the energy-weighted
   combination of its band covariances when their centroids coincide. With distinct
   centroids include the covariance of band centroids. This gives an analytic
   counterexample to a centroid-frequency profile. Also qualify a real general
   modal provider near, but within, its declared confinement domain, with widened
   cladding aperture and refined frequency tables.
6. **Time dependence and spectral phase.** Construct two modes with different
   delayed envelopes and a known spatial cross term. Verify widths and centroids
   at several times by direct summation; full fluence is not an average of widths.
   A common spectral phase changes temporal slices but leaves full fluence fixed.
   A pure delay translates the time axis. Refine temporal sampling at three levels;
   compare exact Fourier evaluation at selected times with zero-padded synthesis.
7. **Signed flux and undefined widths.** Test destructive interference with negative
   local flux, a dark slice, a user signal floor, an annulus, a double lobe, an
   unresolved crossing and a truncated aperture. Assert specific reason codes,
   finite raw signed totals where defined, NaNs only in the affected reductions,
   and exceptions for nonfinite inputs. Equal eigenvalues cannot create unstable
   principal-axis labels. No clipping may change the energy identity.
8. **Gauge and convention invariance.** Apply frequency-dependent channel phases
   and compensated amplitudes, and a declared unitary mixing of a general basis.
   Fluence and time-resolved reconstruction agree with the original to their
   contraction/table bounds. Exercise saved optics spectra, internal spectra,
   `b`-to-`a` conversion, an explicitly rebuilt reference and wrong-identity refusals.
9. **Scalar completeness.** Use temporally separated mode peaks to distinguish a
   simultaneous total peak from a sum of maxima. Check per-mode inner/outer widths,
   empty modes, unit/zero/complex filters, hard-window tiling, wavelength endpoints,
   and a phase-only filter. Check the effective-area statistic against its declared
   definition without equating it to the true spatial peak. Species pressures use
   their actual temperature paths and agree with $\rho_s k_BT$ in the ideal limit.
10. **Two independent resolution diagnostics.** In an orthonormal analytic spatial
    basis, construct a source with a known component outside the retained span:
    the quadrature residual can converge to zero while the span residual remains.
    Adding the missing basis function removes the latter. Separately under-resolve
    the quadrature while keeping the modal space fixed. Test a weak selected band,
    rank deficiency, zero-source denominators, plasma refusal for the
    polarisation-only span test, and general-guide missing-coarse-rule refusal.
    Use conditioned least-squares/summation bounds; compare on an independent rule.
11. **Execution and storage.** Statistics only read the state: they add no
    operation to the march and do not change its schedule or stage kernels, a
    structural contract. Within one backend, a replayed run with statistics keeps
    the same steps (exact) and its propagated arrays within a rounding bound fixed
    before measurement of the run without them; an adaptive run keeps its arrays
    within the case's accuracy envelope against a tighter or stored reference, with
    its accepted/rejected counts in a loose band. Cover split absorbers, rejection,
    replay and modified noise. Per-member diagnostics must
    not broadcast the largest batch error to every member. Verify partial chunks,
    thread-count agreement within a rounding bound fixed before measurement,
    periodic/plane equality for admitted cheap names (exact where the periodic row
    reuses the data the plane stores, otherwise within a rounding bound),
    memory/HDF5 round trips, resume hash mismatches, schema compatibility and
    scans retaining statistics without fields. Existing between-plane row-count
    defects are repaired with regression tests before qualifying new periodic names.

Qualify NumPy first, then Numba CPU and Kokkos CPU. On identical sampled data derive
parity tolerances from accumulation length, FFT errors, conditioning and the local
crossing slope; do not ask for relative agreement of two undefined or vanishing
widths. For a crossing, propagate a profile error as
$|\delta r|\lesssim(|\delta D|+|\delta L|)/|D'(r)|$; a nearly tangent crossing
is unresolved rather than arbitrarily accurate. Record convergence in aperture,
transverse rule, profile samples, frequency tables and time sampling separately.

Benchmark the default statistics, each new family, and combined requests on roughly
ten accepted steps or a fibre short enough to finish in seconds. Include ordinary
few-mode and frequency-dependent general-modal cases, time-resolved requests with
several output sizes, batch members, chunk sizes, persistent/scratch bytes and host
transfer bytes. Record the window, warm-up and measured distributions in
[benchmarks](../benchmarks.md); do not extrapolate to a full-length propagation.
GPU qualification remains the separately exercised device lane, never inferred
from CPU compilation.

### 8. Documentation

Extend [statistics](../../guide/results/statistics.md) with the complete capability table,
equations, units, axis shapes, cadence, validity codes, apertures and two worked
examples: fluence widths and time-resolved widths of a coherent two-mode field.
Extend [power and intensity](../../guide/results/power-and-intensity.md) and
[general modal propagation](../../guide/waveguides/general-modal.md) with the distinction
between the actual E/H product, ordinary frozen-profile flux and metric-envelope
surrogates. Document saved-reference identity requirements in
[processing](../../guide/results/processing.md) and filter/time-axis persistence in
[output](../../guide/results/output.md).

Add a short tutorial showing unchanged modal energies with changing beam width,
then a delayed two-mode pulse whose temporal slices differ from its fluence width.
Keep numerical examples short; a full worked example is optional and, if added,
must be executed and its generated output committed under the gallery rules.
Every public constructor and reduction gets a validated docstring with units,
refusals and approximation limits. Update architecture, conventions, seam and
backend documentation when the implementation changes them, and the changelog
in each phase. At Final, the manual links back to the archived SEP.

### 9. Compatibility and impact

The new measurements do not change an existing default statistic, units, reduction,
temporal trace or output shape. SEP 31 separately changes the shared uniform-medium
spatial flux measure. All new measurements here are opt-in. In particular,
total temporal FWHM stays scalar,
`pressure` stays in bar, general-basis coordinate statistics remain restricted,
`peak_intensity` stays on-axis on modal routes, and `quadrature_error` remains the
cached stage alarm. Existing spatial `fwhm_r` is not evidence that a modal request
can run without profiles. New ordinary modal widths describe the frozen-profile
model when that is what propagated; a frequency-dependent diagnostic does not
retroactively improve that model.

The output and seam revisions are deliberate breaking changes, with clear
unsupported-file and incompatible-resume refusals; no old-file migration is required.
Filter transfer samples, temporal
selection, signal floors, aperture, profile approximation, guide/gauge identities,
source scope and diagnostic norms all enter provenance. A backend missing any
requested family refuses it by name and names a backend that supports it; no full
field host fallback is hidden inside a native statistics call.

When implemented, [differences from Luna](../../guide/workflows/luna.md) must describe the coherent
frequency-resolved fluence and time-resolved definitions, centroid-based radial
average instead of an assumed on-axis maximum on one cut, true flux weighting,
per-mode versus total widths, explicit filters, spatial residual instead of an
on-axis reconstruction surrogate, and preserved plane/periodic scheduling.
The broader inventory does not imply numerical identity with a differently
normalised or differently sampled statistic.

The roadmap's transverse-maximum intensity item is fully covered by this SEP.
Multiple-ZDW selection, SEP 31's shared spatial flux, and the remaining ordinary
spatial accessor issues remain separate. The between-plane memory-output
row-count defect is a prerequisite fix for qualification, not a statistics redesign.

### 10. Alternatives considered

- **One width from effective area.** Effective area encodes a nonlinear overlap and
  does not fix a profile, its centroid or its intermodal interference. Refuse this
  shortcut even when it happens to reproduce a Gaussian calibration.
- **Only a radial cut through the guide axis.** This assumes the maximum, symmetry
  and centre that a multimode field can lose. Prefer covariance plus an explicitly
  radialised central width, with an undefined result for a hollow central profile.
- **Only pulse-integrated statistics.** These conceal spatial changes within a
  pulse. Both fluence and time-resolved measurements are required here.
- **Only $|E|^2$ or $|b(t)|^2$.** These are useful quantities with different physical
  meanings. They are not the adopted broadband flux and cannot silently replace it
  when its sign makes a conventional width undefined.
- **Incoherent modal intensities.** This removes phase-dependent spatial beating.
  An incoherent ensemble requires an explicitly defined ensemble, not discarding
  cross terms of one deterministic propagated field.
- **A single frequency-independent matrix everywhere.** Useful when the kernel
  separates, wrong when the profiles vary. Its domain is an optimization contract.
- **Rename the quadrature alarm as reconstruction error.** Integration accuracy
  and modal completeness are independent; two distinct diagnostics are required.
- **Always compute every statistic or accept arbitrary closures.** This defeats
  bounded device execution, meaningful cadence and serialisable provenance. Keep
  the catalogue and explicit capability refusals.

### 11. Phases

1. **NumPy scalar catalogue and measurement contract.** Implement the per-mode
   durations, hard-window total peak, filters, effective-area diagnostic and
   species pressures with tests and manual coverage. Introduce and validate the
   immutable measurement records, capability vocabulary, output metadata and seam
   revision. Backends explicitly refuse new unsupported families. Keep the default
   path the same code path, so default results are unchanged (an exact no-op
   contract); resolve API/norm questions before accepting the proposal.
2. **NumPy beam measurements, including temporal slices.** Deliver the chunked
   reference reconstruction, ordinary and general modal fluence, centroids,
   covariance/diameters, radialised widths, temporal series, signed-flux policy,
   transverse peak, offline parity and the short tutorial. Qualify E/H identities,
   modal phase/gauge, taper motion, aperture and independent sampling convergence.
   Both fluence and time-resolved measurements are required to close this phase.
3. **NumPy source diagnostics.** Deliver fresh filtered quadrature norms where an
   embedded pair exists, and the polarisation-span residual with an independent
   validation rule. Document current-source and missing-rule refusals. Prove
   separation of quadrature error from modal truncation and read-only evaluation.
4. **Numba CPU parity.** Implement admitted requests with bounded preallocated
   buffers, deterministic shared bodies and reference comparisons. Qualify batch,
   thread count, replay and periodic scalar additions; record the short-window
   cost and memory ledger. Do not hide a reference-backend fallback.
5. **Kokkos CPU parity and completion.** Update consumed contract fields, native
   reconstruction/reductions, packed values/validity and Python adapter in
   `solprop-kokkos`. Qualify the same definitions without full state observation,
   including general-modal magnetic reconstruction and temporal output volume.
   Run both repositories' applicable gates, publish measurements, reconcile open
   questions with issues/roadmap and archive only after all required CPU parity.

Each phase is independently mergeable, with the full gate, slow and coverage lanes
for implementation, the appropriate documentation, and its Part II report. Until
a family is implemented on a backend, its refusal remains tested. GPU work follows
the standing separate qualification policy. No phase changes the propagation model
to make a diagnostic easier to compute.

### 12. Open questions

- **Q1 — acceptance, maintainers:** confirm the proposed centroid-centred radial
  average for modal `fwhm_r`, and the policy that signed local flux can make widths
  undefined while preserving raw signed energy/power. The draft recommends these
  rules; they are new measurement conventions, not assumptions already approved
  for implementation. Both fluence and time-resolved families are in scope.
- **Q2 — acceptance, maintainers:** approve the fresh drive norm and the separate
  electric-profile polarisation-span diagnostic, including its limited physical
  interpretation and explicit deferral of current-source span measurements.
- **Q3 — phase 1:** settle exact public constructor names, status encoding and
  persistence of the time/filter axes. Retain the shapes, semantics and resume
  requirements above. Choose a measured default diagnostic-rule recipe only after
  qualifying the ordinary and general-modal domains; do not inherit nonlinear
  nodes as an unexamined width-resolution default.
- **Q4 — phases 2 and 4:** choose the bounded strategy for centroid-centred radial
  resampling, moving-guide diagnostic tables and general-modal H synthesis.
  Performance may choose recomputation or additional maps, but cannot freeze the
  frequency dependence or substitute a new gauge. Record interpolation and
  aperture error estimates, including shallow-crossing failure criteria.
- **Q5 — phase 3 / future roadmap:** a missing embedded quadrature on general
  guides remains a named refusal. A future current-source span diagnostic needs a
  separate norm/response convention; neither extension is required for Final here.
- **Q6 — phase 5:** confirm periodic scalar qualification after the existing
  between-plane row-count issue is fixed. Large beam/time-series families remain
  plane-only; a later periodic extension needs its own measured cost admission.

### 13. References

1. A. E. Siegman, “How to (Maybe) Measure Laser Beam Quality,” in *DPSS (Diode
   Pumped Solid State) Lasers: Applications and Issues*, OSA Trends in Optics and
   Photonics **17**, MQ1 (1998).
   [doi:10.1364/DLAI.1998.MQ1](https://doi.org/10.1364/DLAI.1998.MQ1).
2. ISO, *ISO 11146-1:2021: Lasers and laser-related equipment — Test methods for
   laser beam widths, divergence angles and beam propagation ratios — Part 1:
   Stigmatic and simple astigmatic beams*.
   [Official standard record](https://www.iso.org/standard/77769.html).
3. F. Poletti and P. Horak, “Description of ultrashort pulse propagation in
   multimode optical fibers,” *Journal of the Optical Society of America B*
   **25**, 1645–1654 (2008).
   [doi:10.1364/JOSAB.25.001645](https://doi.org/10.1364/JOSAB.25.001645).
   Context for broadband multimode/polarisation modelling, not a source of a
   beam-width formula or an implementation to reproduce.
4. solprop, [conventions](../conventions.md),
   [statistics](../../guide/results/statistics.md),
   [power and intensity](../../guide/results/power-and-intensity.md),
   [general modal propagation](../../guide/waveguides/general-modal.md),
   [kernel contract](../kernel-contract.md), and the modules named in §3.

**Source verification at drafting.** The publisher records confirm the titles,
authors, dates and DOIs of [1] and [3]; their accessible abstracts establish scope,
but their full text was not available through the consulted public endpoints.
ISO's official record confirms [2]'s edition and restricted beam-class scope; the
full normative measurement procedure was not inspected. No material coefficient
or empirical beam-width constant is transcribed. The flux, Parseval, Gaussian
width and projection formulas above are derived here from the package's declared
conventions; [1–3] provide context rather than unverified equation authority.
An independent synthetic complex-profile calculation on both DFT layouts checked
the fixed-profile contraction, temporal/spectral Parseval equality, integrated
modal energy and invariance of fluence under common spectral phase; the largest
relative discrepancy was $2.9\times10^{-16}$ for 64 time samples and three modes.
This checks the algebra, not provider accuracy or a propagation implementation.
Implementation must check any additional external formula it adopts against its
primary source and report that check. This draft does not certify ISO compliance.

## Part II — Implementation record

No implementation has started.

## Part III — History

- 2026-09-27: Drafted after a capability and reconstruction audit. Scope includes
  both full-pulse fluence and time-resolved beam widths, explicit spectral filters,
  and separately defined quadrature and modal-span diagnostics.
- 2026-10-17: Restated for [SEP 25](archive/sep-0025-rounding-level-reproducibility.md).
  §7 item 11's "propagated arrays bitwise unchanged" by statistics becomes a
  structural rule (statistics only read the state and change no march operation)
  plus a rounding bound fixed before measurement for replayed runs and the case's
  accuracy envelope for adaptive runs, with step counts exact on replay and in a
  loose band otherwise; its thread-count invariance and §6.2's deterministic
  partial reductions become agreement within a rounding bound. Phase 1's
  "defaults bitwise unchanged" (§11) stays exact as a no-op contract: the default
  takes the same code path. Periodic/plane equality stays exact only where the row
  reuses the data the plane stores; HDF5 round trips and resume hashes stay exact.
- 2026-10-22: Assigned shared uniform-medium spatial flux and transported
  observation planes to SEP 31. This proposal retains its in-route modal
  measurement scope; schema changes require no old-file compatibility layer.
