# Conventions

The units, signs, normalisations, grids and array layouts that every module assumes and
every docstring defers to. A function that departs from any of these says so in the first line
of its docstring and explains why. Sign errors escape review; these belong in one place.
The layer letters P, T, B, D and X that docstrings use are defined in
[architecture.md §3](architecture.md) and ADR-001, not here.

## 1. Units

SI exclusively inside the package: metres, seconds, joules, watts, volts per metre, pascals,
kelvin, radians, radians per second for angular frequencies, m⁻³ for number densities.
One documented exception at the **user interface**: pressures are accepted in bar
(`constants.BAR` converts) because the field works in bar, and pressures are reported in
bar too: the `pressure` statistic in an output file is in bar, read straight off the
run's pressure profile. Wavelengths are always metres.
Temperatures default to 293.15 K (20 °C, Luna's room temperature;
`constants.ROOM_TEMPERATURE`) where a default is needed. Constants come from
`solprop.constants` (CODATA via `scipy.constants`), never typed by hand.

## 2. Fourier transform and the sign of propagation (ADR-017)

Two conventions, and one rule for where each lives.

**Optics convention — the interface.** Every physical expression in the documentation, every
user-facing complex spectrum or phase, the offline propagators and compressors, the
processing outputs and the HDF5 files use

```
Ẽ(ω) = ∫ E(t) e^{+iωt} dt,        E(t) = (1/2π) ∫ Ẽ(ω) e^{−iωt} dω
```

so a plane wave is `e^{i(βz − ωt)}`, propagation multiplies the spectrum by `e^{+iβz}`, a delay
by τ multiplies it by `e^{+iωτ}`, positive group-velocity dispersion is the positive parabola
`½β₂(ω − ω₀)²`, and a positive Taylor phase coefficient gives a positively chirped pulse —
as in any textbook.

**FFT convention — inside.** The kernel state, the linear operator, the noise field, the
translation layer and everything behind the backend seam use the transform that NumPy,
pocketfft and FFTW implement directly:

```
Ẽ(ω) = ∫ E(t) e^{−iωt} dt,        E(t) = (1/2π) ∫ Ẽ(ω) e^{+iωt} dω
```

so the plane wave is `e^{i(ωt − βz)}`, a delay is `e^{−iωτ}`, and the linear propagation of
the stored spectrum is `Ẽ(z) = e^{D z} Ẽ(0)` with

```
D(ω, z) = −i (β(ω, z) − β₀(z) − β₁(z)(ω − ω₀)) − α(ω, z)/2 − α_abs(ω)/2,
β = (ω/c) Re n_eff,   α = 2 (ω/c) Im n_eff,   Im n_eff > 0 is loss,
```

with `α_abs` the spectral absorber rate of ADR-007 (zero where there is no collar). This is
the envelope-grid form. On a field grid `β₀` is kept, because a constant phase on the
half-spectrum of a real field rotates its carrier-envelope phase, and the frame term is
`β₁ω` through the origin; an envelope grid with a third-harmonic term keeps `β₀` too, and
with it measures the frame term `β₁ω` from the origin, because `β₁(ω − ω₀)` would still
remove the constant `−β₁ω₀`, which `A³ e^{2iω₀t}` does not share
(`docs/guide/numerics/linear-operator.md`, "Which terms the frame removes"). These are Luna's
internal signs, so kernels and stored spectra compare directly with Luna's.
The stored spectral state is the **raw forward DFT of the sample array**,
`Ẽ_k = Σ_n E_n e^{−2πi kn/N}` (NumPy forward: no `1/N`, no δt), inverted by
`E_n = (1/N) Σ_k Ẽ_k e^{+2πi kn/N}`. Its phase origin is therefore the **first sample**,
`t = −n_t δt/2`, not the window centre: relative to `t = 0` the stored phase carries the
linear ramp `e^{−iω_k n_t δt/2} = (−1)^k`, `k` being the storage index, on both grids.
Nothing inside the seam cares — the linear operator, the noise, the responses and every
delay are origin-independent — and neither do energies and densities (§4); only absolute
spectral phases do, and those are an interface matter (below).

**The boundary is a conjugation.** For a real field the optics spectrum is the complex
conjugate of the FFT spectrum, bin by bin. For an envelope the same holds in both domains once
each convention uses its own analytic signal: the optics envelope is the conjugate of the FFT
envelope. The conversion is exact and involutive, and it appears in exactly three places,
always through the named helpers `fourier.to_optics` and `fourier.to_internal`:

1. **pulse builders and offline propagators**: user-supplied spectral phases, Taylor
   coefficients and transfer functions are conjugated into the internal arrays;
2. **output handlers and processing accessors**: internal arrays are conjugated on the way
   out; HDF5 datasets are written in the optics convention and carry the attribute
   `spectral_convention = "optics"`; resuming a run conjugates back;
3. **processing accessors with `convention="fft"`**: return internal-convention arrays for
   direct comparison with Luna files, which store the FFT convention.

For **spectra** the boundary also moves the phase origin to the window centre, so that a
user-facing spectral phase is referenced to `t = 0`:
`Ẽ_optics,k = (−1)^k · conj(Ẽ_k)`. `fourier.spectrum_to_optics` / `spectrum_to_internal`
do both steps; `to_optics` / `to_internal` do the conjugation alone and are the right tool
for time-domain arrays and wherever only the sign matters. A Taylor phase about `ω₀` or a
pure delay commutes with the origin shift and may be applied to a stored spectrum directly
(Luna's `prop_taylor!`); an *absolute* spectral phase — measured data, a group delay read
as an arrival time — must go through the spectrum helpers. The shift is `(−1)^k` on the
storage index for both layouts and nothing else: a pulse's `phase[0]` is therefore its
carrier-envelope phase at its own peak on either grid, independent of the window length
(Luna's envelope-grid constant `exp(−iω₀ n_t δt/2)` is not carried). No other
module conjugates anything. Energies, spectral densities, masks and windows are
convention-free (`|Ẽ|²`). A round-trip test guards the sign: build a pulse with positive
group-delay dispersion, read the spectral phase back, delay it by τ and check that it arrives
later.

The amplitude conversion is a separate, named boundary beside these Fourier helpers:
`fourier.power_amplitude` maps internal `a` to saved `b`, and its inverse maps a resumed
plane back. It is called by `driver.run._power_observation` and the batch scan path
before the output handler's conjugation. It never changes the Fourier sign. Input
assembly and `Simulation.with_input` divide at the entrance. These are the amplitude
conversion sites; no propagation kernel converts amplitudes.

## 3. Grids

**Time axis**: `t_n = (n − n_t/2) δt`, `n = 0 … n_t − 1`, centred on zero; `n_t` even and
2-3-5-smooth. `fft_size="auto"`, the default, takes the cheapest such size by the transform-cost
model (`grid.fft_cost`) between the requirement and the next power of two; `"pow2"` and `"smooth"`
take the next power of two or the next such size.

**Field grid** (`FieldGrid`, real `E(t)`): the state is the one-sided spectrum on
`ω_k = k δω`, `k = 0 … n_t/2`, so `n_omega = n_t/2 + 1` and `δω = 2π/(n_t δt)`. Construction
from the wavelength limits `(λ_min, λ_max)` and the requested window `trange` (Luna
`Grid.jl:61–110`):

- `ω_min = 2πc/λ_max`, `ω_max = 2πc/λ_min`, `ω_max_win = 1.1 ω_max`;
- the **fine** grid is sampled at `ffac × f_max` (`δt_fine = 1/(ffac f_max)`, or the user's
  `δt` if finer); `ffac = 6` by default so that a cubic response (`E³`, reaching `3ω_max`)
  is resolved; `ffac = 4` suffices for a response without third harmonic; `ffac ≥ 2`;
- `n_to` = the transform size ≥ `trange/((1 − 2 collar_fraction) δt_fine)`, room for the
  temporal collar outside the request (`collar_fraction = 0.1` by default; `trange/δt_fine`
  under `collar_placement="inside"`); the realised window is `trange_even = n_to δt_fine` and
  `δω = 2π/trange_even` on both grids;
- the coarse grid keeps the fine bins up to the first `ω > ω_max_win`, its size `n_t` rounded
  up to the smallest even divisor of `n_to` (so `2^k + 1` bins on a power-of-two grid; clamped
  to the fine grid); `δt = π/ω[−1]`, `n_t = 2 (n_omega − 1)`;
- invariants (asserted): `δt/δt_fine = n_to/n_t = ω_fine[−1]/ω[−1]`, and the coarse time
  samples coincide with fine samples.

Note that `ffac` changes the whole grid (δω, window length, `n_omega`), not only the fine one.

**Envelope grid** (`EnvelopeGrid`, complex `A(t)`): the state is the full spectrum
`Ã(ω)` on `ω = ω₀ + v`, `v = 2π·fftfreq(n_t, δt)` **in FFT order** (no shifts in kernels;
processing shifts for display), with `ω₀ = 2πc/λ_ref`. Construction (Luna `Grid.jl:170–252`):

- the relative span is measured symmetrically, `max(|ω_hi − ω₀|, |ω₀ − ω_min/2|)`, so the
  grid reaches the bottom of the active band even when the carrier sits near the blue edge;
- without third harmonic: `ffac = 2` and `ω_hi = 1.1 ω_max`, so the fine and coarse grids
  coincide (no oversampling) unless the user asks for a finer `δt`;
- with third harmonic: `ffac = 6` and `ω_hi = ω_max`, with a coarse grid cropped to the active
  band through `ω_max_win` (its size an even divisor of `n_to`, not necessarily a power of
  two) and the same invariants as the field grid, plus coincidence of the time samples.

**Active band and windows** (both grids): the active mask is `ω_min/2 < ω < ω_max_win`
(Luna `sidx`; the dispersion relation, the noise and the error norm live there); the
in-band region is `[ω_min, ω_max]`. The spectral collar is a Planck taper rising over
`[ω_min/2, ω_min]` and falling over `[ω_max, 1.1 ω_max]`; the temporal collar is a Planck
taper rising over `[t_min, −t_flat]` and falling over `[t_flat, t_max]`, with
`t_flat = min(trange/2, (T/2)(1 − 2 collar_fraction))`. The grid realises
`T ≥ trange/(1 − 2 collar_fraction)`, so `t_flat = trange/2`: the whole request is flat and
each collar, at least a fraction `collar_fraction` of the window (Luna's power-of-two grids had
no minimum), lies outside it (SEP 21, ADR-057). Only `collar_placement="inside"`, which
realises the request rounded up and is kept to reproduce frozen witnesses, can put the
collar inside the requested window. In
solprop these profiles become **rates** (ADR-007); the hard band limit outside the active
mask is a 0/1 mask.

**Oversampling scale factors** (Luna `NonlinearRHS.jl:19–56`): zero-pad the coarse spectrum
into the fine one with scale `(n_omega_fine − 1)/(n_omega − 1) = n_to/n_t` (field) or
`n_to/n_t` (envelope, copying the leading and trailing halves); crop back with the inverse.
The factors cancel the different `1/N` of the two inverse transforms.

## 4. Field amplitudes, power, energy and spectra

**True-power amplitudes at the interface (ADR-030).** Pulse builders and saved modal
spectra carry `b(omega,z) = sqrt(w(omega,z)) a(omega,z)`, where `a` is the internal
vacuum-normalised amplitude and `w = Re n_eff`. The prefactor's explicit index override
also defines `w`: the GNLSE mapping uses `index=1.0`, so that route's conversion factor is one.
Assembly divides both the launch and noise by `sqrt(w(omega,0))`. Observations multiply
by the plane's square root through `fourier.power_amplitude`; resume divides again.
Outside the active band the stored weight is one and the state is masked to zero.

**Energy.** On an envelope grid the energy is `dt/n_t sum_k |b_k|²`. On a field grid
it is `dt/n_t (|b_0|² + 4 sum_{0<k<n_t/2} |b_k|² + |b_Nyquist|²)`, with the real
endpoints of the analytic signal. Equivalently replace `|b_k|²` by `w_k |a_k|²`.
The time integral of the analytic signal's squared modulus gives the same energy.
The raw DFT scaling and the spectral-density Jacobian are unchanged. A requested
launch energy is therefore true energy in joules on both grids.

**Temporal Poynting flux.** Run statistics use
`P_m(t) = sum_pol Re[a_m(t) conj(a_w,m(t))]`, where `a_w` is the analytic signal
synthesised from `w a`. `peak_power`, `peak_power_total`, the windowed peaks and both
pulse widths use this same trace. Its unequal-frequency terms carry
`(w(omega)+w(omega'))/2`. The instantaneous squared modulus of the saved amplitude
`b(t)` instead carries `sqrt(w(omega) w(omega'))`; both have the same time integral,
but are different broadband quantities. The difference is second order in relative
index variation. Flux is not clipped: negative excursions are possible in principle.
These are cycle-averaged analytic-signal quantities; the real instantaneous
flux's sum-frequency term is outside this statistic's scope.

**Physical electric field and intensity.** The internal basis retains
`N_m = eps0 c/2 integral |e_m|² dA` and the physical synthesis
`E = sum_m a_m e_m/sqrt(N_m)`. With the magnetic-field equivalent `E_w` synthesised
from `w a`, the axial intensity statistic is
`S = eps0 c/2 Re[E dot conj(E_w)]`. For a monochromatic single mode this reduces to
`Re n_eff eps0 c |E|²/2`. Gas rate models use the physical electric-field amplitude
in V/m, not this Poynting intensity; plasma synthesis and the response bodies take no
power weight.
The effective area is `(integral |e|² dA)² / integral |e|⁴ dA`. Thus the Kerr
coefficient per true watt is `3 omega chi3/(4 eps0 c² (Re n_eff)² A_eff)`;
ADR-026's GNLSE calibration gives `n2 omega/(c A_eff)`.

**Varying index.** Translation adds the exact cumulative term
`Phi_w = -1/2 [ln w(omega,z) - ln w(omega,0)]`. Constant fills have an identically
zero term, which leaves the Phi coefficients as they are. Parameter expansions append
endpoint basis products to the integrated weights. The z-table representation adds
the independently resolved logarithmic term. Noise retains ADR-008's unitary phase:
this real term does not enter its transport.

**Shot noise.** The stored realisation is the true-power draw: one photon per
mode per active bin, fixed modulus and random phase. Its field-grid modulus is
`sqrt(hbar omega/domega) sqrt(2 pi)/(2 dt)`, and its envelope-grid modulus is
`sqrt(hbar omega/domega) sqrt(2 pi)/dt`. The internal draw is divided by the entrance
square root. It is not a circular-Gaussian draw.

**Spatial geometry** ([SEP 34](seps/archive/sep-0034-medium-flux.md)). Radial, Cartesian and
line states retain physical V/m or V m units and have no modal power-weight row. They are
measured with the scalar axial admittance of the medium at the plane,
`Y_c = eps0 c² k_z/omega` per transverse column and bin, zero on nonpropagating,
inactive and nonpositive-frequency bins (`solprop.spatial.SpatialFlux`). Energy, beam
and angular-port spectra, absorbed powers and the noise draw use `Y_z = Re Y_c`:
`U = dt/(2 n_t) sum_{j,p,k} v_j gamma_k Y_z |E_jpk|²`, with `gamma_k` the endpoint
weights above. Local fluence, intensity and on-axis quantities synthesise `E` and
`G = Y_c E` with the **complex** admittance and form `S_z = 1/2 Re[E_a dot conj(G_a)]`;
they are signed and never clipped, and widths refuse a fluence negative beyond its
rounding budget. `peak_field` is the named physical-field diagnostic, `max |E_a|` in
V/m, which ionisation uses. A launch normalises physical energy or peak axial power at
its plane in the medium there; a plane at a declared index jump names its side
(`before` or `after`). The hybrid planar guide uses the slab metric
`w = c Re q/omega`, `q = sqrt((omega n_eff/c)² - k_y²)`, on its vacuum-normalised slab
amplitudes, without a second `eps0 c/2`.

**Varying radial medium.** A radial gas path adds the endpoint term
`-1/2 ln[Y_z(omega,k_r,z)/Y_z(omega,k_r,z0)]` to the cumulative linear exponent, the
spatial counterpart of `Phi_w`, so `Y_z |E|²` is invariant under the linear flow. It is
admitted only when every propagating column meets the adiabatic certificate:
`Lambda = TV(ln Re k_z)/2` over the run path, `R* = tanh² Lambda` and
`delta* = expm1(Lambda tanh Lambda)` within `AdiabaticAccuracy` (default 1e-4 each;
`solprop.adiabatic`).

**Field planes and coupling** ([SEP 31](seps/archive/sep-0031-simulation-coupling.md)). A
`FieldPlane` holds a stored spectrum exactly as the output does: the optics-convention
raw DFT of each column, referenced to the window centre, in √W (ordinary and general
modal coordinates), V m (the Hankel spectrum of a radial field) or V/m (the DFT of a
Cartesian one).

- *Plane reference.* The absolute spectrum is
  `E_abs(omega) = E_stored(omega) exp[i(phi + omega tau)]` with `omega` the absolute
  angular frequency of the bin, `tau` the absolute time of the record's `t = 0` and
  `phi` a carrier phase. A run's reference is the integral of the affine term its
  linear operator removed, `phi + omega tau = integral_0^z R dz`,
  `R = beta0 + beta1 (omega - omega_o)`, with `omega_o` the frame's reference
  frequency on an envelope grid that removes `beta0` and zero on a field grid or an
  envelope grid with a third harmonic (where `beta0` is kept in the operator and
  contributes nothing here). The run record stores the exact
  Chebyshev antiderivatives of the frame tables, so the pair is rebuilt from the
  file; a launched run inherits its source plane's pair as its origin.
- *Affine phase and delay.* An operation that removes a known affine term adds it to
  `(phi, tau)`, so `E_abs` is unchanged and a pulse is never translated out of its
  window: a free-space gap removes `R L` of its medium (or `beta1(omega0) L` without
  a frame), and `regrid(delay=, phase=)` moves the record against the reference. A
  spectral element that drops its own constant phase or group delay (a grating or
  prism pair, measured mirror data) clears that term's entry in
  `PlaneReference.calibrated`, and `absolute()` then refuses rather than invent it.
- *Spectral-density scaling.* A raw DFT bin of a localised record is
  `q E(omega)/dt`, with `E(omega)` the optics-convention transform in V s/m (times
  the transverse unit) and `q = 1` on a field grid, 2 on an envelope grid, whose
  envelope is the analytic signal. Moving a record between lattices keeps the
  continuous transform, not the bin values: target bin `m` receives
  `(dt/(dt' q'_m)) exp(i dphi) sum_j a_j exp(i omega'_m (t_j + Delta))`
  (`coupling.regrid`), with `q'` the target's factor (2 on a field grid's interior
  bins), so equal-bin copying is exact only on an identical lattice.
- *Energy.* Each representation keeps its native metric: the identity for
  true-power coordinates, `conj(W)` for a declared general basis, the axial admittance
  `Y_z` above for a spatial plane. The scalar coupling model fits in the common
  paraxial measure `M0 = n_ref eps0 c dA/2`, `n_ref` the receiving port's core index;
  the captured and uncaptured energies are reported in it, and the launch in the
  target's native metric. Nothing is rescaled to a requested energy.

## 5. Modes and polarisation

- `n_eff(ω, z)` complex with `Im n_eff > 0` for loss; `n_eff − 1` computed directly for gases.
- Marcatili (full): `n_eff = sqrt(ε_co − (u/(ka))² (1 − i v_n/(ka))²)`, `k = ω/c`,
  `u = j_{n−1,m}` (HE) or `j_{1,m}` (TE/TM), `v_n` as in Luna `Capillary.jl`; reduced:
  `n_eff = 1 + (ε_co − 1)/2 − c²u²/(2ω²a²) + i c³u²v_n/(a³ω³)`.
- The **coherent wall** (`CoherentWall`) condenses a contiguous HE_1m set of
  the full Marcatili model onto one Robin impedance and shares that row's `v_n`: its
  `k_eff = k/v_n` uses the same vacuum `k = ω/c`, while the paraxial carrier it is written
  about is the in-core `k√ε_co`. It is the loss model of its channels, so it is
  declared instead of a `LossSpec`, never beside one. `CoherentWall(impedance="te_tm")`
  condenses the wall's **two** grazing impedances instead of their average — `1/k_TM` for
  the field normal to the wall and `1/k_TE` for the field tangential to it, this package's
  own `v_TM` and `v_TE` over `k_0` — rotated into the radial families of one total
  azimuthal order, which is exactly conserved: a block per order, not per polarisation,
  in which `TE_0m` and `TM_0m` keep their own `v_n` and the two hybrid families of an order
  `ν ≥ 1` are **coupled** by the half-difference of those impedances.
  Its basis is attached to the core
  radius, so a **varying** guide is written in a moving basis and costs the connection
  `C_mn = (a'/a) 2 s_m s_n u_m u_n/(u_m² − u_n²)`, real and antisymmetric, derived and
  measured in `docs/guide/waveguides/modes.md`. For a solid guide the same connection is `Γ = W⁻¹S`
  with `S_mn = ⟨ψ_m|∂_zψ_n⟩` on the flux metric and its self-terms kept, anti-Hermitian
  rather than antisymmetric, its diagonal the normalisation's own rate plus a gauge phase
  (`solprop.modes.modal_connection`, `docs/guide/waveguides/general-modal.md`); the Bessel form is its
  real exactly-normalised special case. The cumulative `Φ` beside such a block must be a
  multiple of the identity in the channel axis, so its normalisation term is the lowest
  channel's, and the flow's diagonal carries the rest of each channel's own,
  `−½ d ln(w_n/w_1)/dz`. The taper connection is stored antisymmetric in the true-power
  amplitude, `W^(-1/2) C W^(1/2)` in the channel amplitude, and the wall block in the field
  amplitude: one
  forward-Riccati reduction of the moving-basis equation yields both, and the diagonal.
- Polarisation components are the transverse Cartesian components `(x, y)` in the waveguide
  frame. The polarisation axis has length 1 for scalar runs (the single component is the
  mode's declared linear polarisation) or 2 (`x`, `y`) for vector runs; circular states are
  `(x ± iy)/√2`; Jones vectors are `[E_x, E_y]`.
- **The unrotated member of a degenerate pair is the x-polarised one** (ADR-031). A
  hybrid mode's `even` parity is the `cos((n−1)θ)` transverse pattern about the x axis,
  x-polarised for `n = 1`, and `odd` is its quarter-turn partner; every family obeys the
  one definition in `ChannelLabel`. A Marcatili mode reaches it through `phi`, a rotation
  of the pattern about the axis **from x towards y** — the sense `solprop.polarisation.rotate`
  already documents — so `phi = 0` is `even`, `phi = π/(2n)` is `odd`, and for `HE_1m`
  `phi` is the polarisation angle measured from x. `translate.problem.DEFAULT_COMPONENTS`
  is therefore `("x",)`. Luna's `ϕ = 0` is the y-polarised member and its scalar interface
  resolves `y`; the two differ by a quarter turn and by nothing else (Luna's
  `Capillary.jl`).
- Quadrature weights include the Jacobian (`a² ξ w_ξ w_θ` in polar coordinates); the
  single-node radial rule carries the factor `2π`.

## 6. Array layout and axis names

Kernel state and canonical storage: `(n_batch, n_modes, n_pol, n_omega)` complex128, ω
contiguous (C order). Under `precision="single"` (SEP 33, ADR-054) a backend holds the
state as complex64 at the run's exact power-of-two scale `s`, `a / s`; what crosses the
seam to the host, and every saved field, is the physical state widened to complex128.
Saved fields: `(n_z, n_modes, n_pol, n_omega)`, with a leading `n_batch`
for batch runs. Space-domain buffers: `(n_nodes_chunk, n_pol, n_to)`. Convenience accessors
may squeeze trivial axes; the seam and the files never do.

Standard axis names, reused verbatim in every jaxtyping annotation and docstring:

| name | meaning |
|---|---|
| `n_batch` | batch members |
| `n_modes` | modes (1 for mode-averaged) |
| `n_pol` | polarisation components of the kernel state (1 or 2) |
| `n_components` | Cartesian components of a local physical field (1, 2 or 3); a general modal guide carries `n_pol = 1` and `n_components = 3` |
| `n_omega` | coarse spectral bins |
| `n_omega_fine` | fine spectral bins |
| `n_t` | coarse time samples |
| `n_to` | fine time samples |
| `n_nodes` | transverse quadrature nodes |
| `n_species` | ordered labelled mixture components, including non-ionising components |
| `n_channels` | active plasma channels, in mixture order |
| `n_points` | arbitrary points of a cross-section handed to `Mode.field` |
| `n_z` | positions along z: saved planes in outputs, sample nodes in the translation layer |
| `n_kinks` | hard kinks of a piecewise representation in z |
| `n_breakpoints`, `n_edges` | interval edges of a piecewise representation or a step schedule (`n_intervals + 1`, `n_steps + 1`) |
| `n_stages` | stages of a Runge–Kutta tableau |
| `n_steps` | steps of a fixed schedule |
| `n_intervals`, `n_deg` | Chebyshev intervals and coefficients per interval |
| `n_flow_terms` | the flattened `(n_intervals, n_deg)` weight axis a stage combines a coupled flow's coefficient block over (`StageScalars.flow_weights`) |
| `n_table` | knots of a rate table |
| `n_windows` | energy windows in statistics |
| `n_apertures` | far-field apertures of a statistics request (`StatsSpec.apertures`, SEP 23) |
| `n_regions` | regions of transverse wavevector of a statistics request (`StatsSpec.spectral_regions`, SEP 23) |
| `n_to_doubled`, `n_omega_doubled` | the doubled, zero-padded convolution grid of a Raman kernel (`2 n_to`) and its one-sided spectrum (`n_to + 1`) |
| `n_terms` | independently damped terms of a Raman response |
| `n_levels`, `n_lines` | rotational levels of a Raman ladder and its `J → J + 2` transitions (`n_levels − 2`) |

A self-validating dataclass whose `__post_init__` relates its arrays gives each array
its own name (`n_energies`, `n_populations`, `n_amplitudes` beside `n_levels` and
`n_lines` on `RotationalLines`; `n_history_*` on `StepHistory`), so that the relation
is enforced in production and not only by the test session's runtime checker
(`AGENTS.md`, "Types and arrays"). Such names are not listed here individually.

## 7. Naming

`snake_case` functions and variables, `PascalCase` classes, `UPPER_SNAKE` constants; Greek
letters spelled out (`omega`, `beta1`, `gamma3`, `chi3`, `alpha`); wavelengths are
`wavelength` (m), never `lambda`; `n_<axis>` for lengths; `z` for the propagation coordinate;
`dz` for a step; `phi` for the cumulative operator; established abbreviations (`fwhm`, `zdw`,
`rdw`, `gnlse`, `uppe`, `ppt`, `adk`, `bsi`) are fine.

Three public names break these rules on purpose. `wavelength`, the reference wavelength of
every interface, and `tools.lambda_rdw` with `lambda_rdw_all` keep the names of Luna's
`λ0` and `Tools.λRDW`, which a reader coming from Luna types first
([coming from Luna](../guide/workflows/luna.md)). `boundary_N`, how many times the
temporal collar's window is applied over the run ([ADR-007](decisions.md)), keeps the
capital of the count it is written as.
