# Statistics catalogue

Use this reference to find the saved statistic, shape and units for your geometry. Select statistics with a [StatsSpec](statistics.md#the-spec), and interpret them using the [physical definitions](statistics-definitions.md). A missing statistic need not mean its physical quantity is zero.

## The catalogue

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 18 16 7 8 51

   * - name
     - shape
     - units
     - between planes
     - definition
   * - ``omega0``
     - ``(n_modes, n_pol)``
     - rad/s
     - yes
     - Centroid of the spectral energy density, :math:`\sum_k \omega_k S_k / \sum_k S_k`.
   * - ``energy``
     - ``(n_modes, n_pol)``
     - J
     - yes
     - Modal column energies, or scalar whole-beam spatial energy with transverse
       spectral weights and the factor :math:`\varepsilon_0 c/2`.
   * - ``energy_window``
     - ``(n_windows, n_modes, n_pol)``
     - J
     - yes
     - The same on a hard half-open spectral gate, one per window.
   * - ``peak_power``
     - ``(n_modes,)``
     - W
     - yes
     - :math:`\max_t P_m(t)` for each mode.
   * - ``peak_power_window``
     - ``(n_windows, n_modes)``
     - W
     - yes
     - The same on the gated spectrum: one scalar per mode per window.
   * - ``peak_power_total``
     - ``()``
     - W
     - yes
     - :math:`\max_t \sum_m P_m(t)`.
   * - ``physical_energy``
     - ``()``
     - J
     - yes
     - The general modal basis's energy: the same-frequency product of the state with
       the basis metric, integrated over frequency. A scalar, because a coordinate
       energy is not additive on a basis that is not orthonormal; on the identity
       metric it is :math:`\sum_m` ``energy``.
   * - ``spectral_flux``
     - ``(n_omega,)``
     - J/(rad/s)
     - no
     - The general modal basis's spectral energy density: the same same-frequency
       metric product published per bin instead of integrated,
       :math:`\mathrm dU/\mathrm d\omega`. Its sum times the angular-frequency
       spacing is ``physical_energy``, exactly on every interior bin. An array where
       the rest of the family are scalars, so it is admitted but not a default and
       is not available between save planes.
   * - ``analytic_flux``
     - ``()``
     - W
     - yes
     - The general modal basis's peak power: :math:`\max_t \mathbf{a}^{\mathsf H}W\mathbf{a}`,
       the same-frequency flux. On the identity metric it is
       ``peak_power_total``. It is **not** the cross-section flux of the
       reconstructed fields, which carries the guide's cross-frequency kernel as
       well; that is
       :py:func:`solprop.processing.modal.reconstructed_flux`.
   * - ``fwhm_t``
     - ``()``
     - s
     - yes
     - Width of the central feature of the total power at half its maximum.
   * - ``fwhm_t_outer``
     - ``()``
     - s
     - yes
     - Total extent of the power above half its maximum.
   * - ``density``
     - ``()``
     - m⁻³
     - yes
     - Number density of the filling gas at the plane.
   * - ``pressure``
     - ``()``
     - bar
     - yes
     - Pressure of the filling gas at the plane.
   * - ``core_radius``
     - ``()``
     - m
     - yes
     - Core radius of the waveguide at the plane.
   * - ``zdw``
     - ``(n_modes,)``
     - m
     - no
     - Zero-dispersion wavelength of each mode at the plane.
   * - ``on_axis_intensity``
     - ``()``
     - W/m²
     - yes
     - Largest analytic intensity in time on the spatial beam axis.
   * - ``on_axis_spectrum``
     - ``(n_omega,)``
     - J s/m²
     - no
     - On-axis spectral fluence density, summed over polarisation. Its sum times
       the angular-frequency spacing is the on-axis fluence.
   * - ``beam_radius``
     - ``(2,)``
     - m
     - no
     - Second-moment radius and interpolated central 1/e² radius of the
       time-integrated fluence, in that order. Undefined radii are NaN.
   * - ``fwhm_r``
     - ``()``
     - m
     - no
     - Twice the first half-maximum crossing of the central fluence profile.
       Empty profiles, profiles hollow at the threshold (axis fluence below it)
       and profiles that never cross it return NaN.
   * - ``aperture_spectrum``
     - ``(n_apertures, n_omega)``
     - J/(rad/s)
     - no
     - The spectrum collected through each far-field aperture of the spec's
       ``ApertureSet``: the stored spectrum weighted, per frequency, by the profile
       at each column's paraxial image ``z c kappa / omega`` in the mask plane
       (``W**2`` under amplitude weighting, ``W`` at the corrected diameter under
       quadrature weighting). Cartesian and line grids.
   * - ``aperture_on_axis_spectrum``
     - ``(n_apertures, n_omega)``
     - J s/m²
     - no
     - The on-axis spectral fluence behind each aperture placed in a Fourier plane
       of focal length ``distances``, summed over polarisation; with every weight
       one it is ``on_axis_spectrum``.
   * - ``region_spectrum``
     - ``(n_regions, n_omega)``
     - J/(rad/s)
     - no
     - The spectrum in each half-open box of physical transverse wavevector of
       ``spectral_regions``; regions that tile the plane sum to ``beam_spectrum``.
   * - ``beam_spectrum``
     - ``(n_omega,)``
     - J/(rad/s)
     - no
     - The whole-beam spectral energy density of a Cartesian or line run. Its sum
       times the angular-frequency spacing is ``energy``.
```

`pressure` is in bar, not pascals: it is read straight off the
{py:class}`~solprop.profiles.PressureGradient` the run carries, and a pressure path is a
user-interface object, which is the one documented exception to SI
([conventions.md](../../dev/conventions.md) §1).

### Backend statistics

The entries below and the species entries later on this page carry
`backend=True`, because a saved plane does not hold what they are about. The
far-field `poynting_spectrum` and `absorbed_power` need the run's linear operator
([far-field apertures](statistics.md#far-field-apertures-and-angular-regions)). The electron
density, the peak ionisation rate and the plasma integration diagnostics need the field
at the quadrature nodes and the run's rate table; the peak intensity needs the mode profiles on
the axis, which is never a node; the quadrature error needs the two projections of a
stage's own polarisation. A backend forms them from its own buffers in one pass,
where the state already lives, and
{py:func}`~solprop.stats.collect` refuses them by name rather than pulling the state back
across the seam to reconstruct them.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 24 10 9 9 48

   * - name
     - shape
     - units
     - between planes
     - definition
   * - ``electron_density``
     - ``()``
     - m⁻³
     - yes
     - :math:`\rho(z)\,f(t_\text{end})`: the density times the ionised fraction at the
       **end** of the window -- what the pulse leaves behind, not what it reaches in
       passing -- at the node of largest fraction, which for a mode-averaged run is the
       one node.
   * - ``peak_ionisation_rate``
     - ``()``
     - s⁻¹
     - yes
     - :math:`\max_t W(\lvert E\rvert)` at that same node, so the two describe one
       column and not two.
   * - ``on_axis_electron_density``
     - ``()``
     - m⁻³
     - no
     - The density left behind **on the axis**, :math:`r = 0`, summed over the species:
       the axis field is synthesised from the mode profiles and scanned with the node
       scan's own rate tables, preionisation and running-integral rule (SEP 19). Opt-in,
       on resolved modal routes.
   * - ``on_axis_peak_ionisation_rate``
     - ``()``
     - s⁻¹
     - no
     - The largest rate on the axis over species and time; cycle-averaged on an
       envelope grid.
   * - ``on_axis_species_electron_density``
     - ``(n_species,)``
     - m⁻³
     - no
     - Each species' density left behind on the axis, in mixture order; zero for a
       species without an ionisation channel.
   * - ``on_axis_species_peak_ionisation_rate``
     - ``(n_species,)``
     - s⁻¹
     - no
     - Each species' largest rate on the axis; cycle-averaged on an envelope grid.
   * - ``peak_intensity``
     - ``()``
     - W/m²
     - yes
     - Cycle-averaged analytic Poynting flux
       :math:`\tfrac{1}{2}\varepsilon_0 c\max_t\operatorname{Re}[E(0,t)\cdot E_w(0,t)^*]`,
       from the mode profiles at :math:`r = 0`. The real instantaneous sum-frequency
       term is excluded; the magnetic equivalent is synthesised from :math:`w a`.
   * - ``quadrature_error``
     - ``()``
     - dimensionless
     - yes
     - The relative difference between the projections of one stage's node polarisation
       with the Kronrod rule and with its embedded Gauss subset, taken once per accepted
       step. ``nan`` when the rule has no embedded pair, because the difference of a rule
       with itself is zero and that is not convergence.
   * - ``plasma_undershoot``
     - ``()``
     - dimensionless
     - no
     - The plasma column's band integration (SEP 18):
       :math:`\max(-\Delta S) / \max \Delta S` of the exposure increments, each maximum
       over every sample, node and population, so that a node the field barely ionises
       does not set it. The filtered rate can dip below zero beside a spike,
       so the fraction is non-decreasing only to within this. A large value says the
       rate varies within a sample faster than the grid resolves; recorded, not acted
       on. Zero under the plain trapezoid.
   * - ``plasma_work_defect``
     - ``()``
     - dimensionless
     - no
     - :math:`\max\lvert\int W(1-f)\,dt - (f_\text{end} - f_0)\rvert / \max(f_\text{end}
       - f_0)` over nodes and populations: the loss current's work per ionisation
       potential against the electrons it freed, normalised by the most-ionised node.
       Second order in :math:`\delta t`; recorded, not acted on.
   * - ``poynting_spectrum``
     - ``(n_omega,)``
     - J/(rad/s)
     - no
     - The axial-flux spectrum of a Cartesian or line run: ``beam_spectrum``
       weighted by the obliquity :math:`c\operatorname{Re}k_z/\omega` of the run's
       own operator, zero on evanescent and inactive bins. Its ratio to
       ``beam_spectrum`` is the beam-averaged obliquity.
   * - ``absorbed_power``
     - ``(4,)``
     - J/m
     - yes
     - The power the spectral, transverse, evanescent and material channels of the
       linear operator remove, in that order: together the whole energy decrement
       of the linear flow on a Cartesian or line grid; on a line per metre of y, in J/m².
   * - ``plasma_boundary_level``
     - ``()``
     - dimensionless
     - no
     - The free-current integrand :math:`fE` at its largest in the :math:`2K` samples
       the band rule's filter reads past the window's start, relative to its largest
       anywhere, both over every node and population:
       non-zero only for a pre-ionised run, and the size of the start-up term of the
       finite-window budget. Zero under the plain trapezoid.
```

The three `plasma_*` diagnostics are recorded by default wherever the field plasma
integrates by the band rule
({py:class}`~solprop.nonlinear.responses.PlasmaResponse`'s `integration`), at each
save plane, on every CPU backend; a native device build refuses them by name.

For spatial runs, `peak_intensity` is the maximum over physical transverse samples
and time; `on_axis_intensity` reconstructs the axis explicitly. For modal runs,
`peak_intensity` retains the on-axis definition above. A radial saved spectrum can be
analysed by passing `space=radial_grid` to `collect`; this supplies the transverse
measure and prevents a Hankel column from being interpreted as a modal amplitude.
For example, `collect(default_spec("radial"), grid, plane, space=radial_grid)` returns
whole-beam energy, both intensities, both radii, radial FWHM and the on-axis spectrum.
The radial second-moment radius is
`sqrt(2 sum(w_r r² F) / sum(w_r F))`, where `F` is intensity integrated over the
whole time window. Interpolated widths use linear interpolation from the separately
reconstructed axis to the radial samples. They describe the central lobe only;
a hollow profile has a valid second moment but no central width at a threshold
its axis fluence lies below (the thresholds are fractions of the peak fluence).

The quadrature error is **an alarm, never a controller input** (see the
[design decisions](../../dev/decisions.md)): above $10^{-6}$ the backend logs a warning once
per plane, because a rule too coarse for the integrand is a modelling error that no step
size can fix. It is the embedded estimate, which costs one extra projection per accepted
step. A check that reconstructed the on-axis polarisation from its modal expansion and
compared it with the polarisation evaluated directly there would cost a whole extra
transform and response pass instead, and is not provided.

**Node maxima and the axis.** `electron_density` and the species densities are read at
the node of largest total electron density, and the peak rates are maxima over the
nodes: values *sampled* at the rule's nodes. The default rule has a node at
$r/a \approx 10^{-3}$, so for one HE₁₁ mode they are the axial values to about $10^{-5}$.
A few-node field-scale rule ([radial nodes](../waveguides/radial-quadrature.md)) has its
innermost node well off the axis -- at 92% of the axial intensity for five Gauss nodes
-- so a steep rate's node maximum is far below the axial peak, and the run warns
(`SampledPeakWarning`). The four `on_axis_*` statistics read the axis itself, at the
cost of one extra row's rate scan per saved plane, or none when the rule's own
Gauss-Radau node is on the axis. They are opt-in and saved-plane only, and refused by
name on the spatial routes, a general modal guide, the mode-averaged route, a problem
without plasma and the TDSE route (which never evaluates an atom for a statistic).
Every backend uses the same host routine for an extra axis scan, or reuses a Radau
axis row from its own node scan. CPU parity is qualified. On a GPU build of the native backend these
statistics require a host copy of the field at each requested save plane; GPU
qualification and a device-only axis scan remain follow-up work
([SEP 19](../../dev/seps/archive/sep-0019-field-scale-quadrature.md)). A multimode
field's largest density need not lie on the axis, so the node maxima keep their
meaning beside them.

The ionised fraction behind `electron_density` and the species statistics is the plasma
response's own, with the pre-ionised part inside the exponential, so the density a run
reports is the density its polarisation saw.

### Spatial width requirements

`fwhm_r` is implemented for radial geometry. It requires the transverse coordinates
and physical field reconstruction; a temporal modal spectrum alone cannot supply a
radial width. Missing geometry is reported explicitly.

## Species populations

Mixture capillary defaults include all five species statistics. Radial and Cartesian
mixture construction adds them when no explicit `StatsSpec` was supplied. Species
order always follows the mixture, including zero rows for non-ionising components.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 24 18 9 9 40

   * - name
     - shape per plane
     - units
     - between planes
     - definition
   * - ``species_ionisation``
     - ``(n_species, n_nodes)``
     - dimensionless
     - yes
     - Final ionised fraction, including preionisation.
   * - ``species_density``
     - ``(n_species,)``
     - m⁻³
     - no
     - Component number density.
   * - ``species_electron_density``
     - ``(n_species,)``
     - m⁻³
     - yes
     - Electron densities at the node with the largest total electron density.
   * - ``species_peak_ionisation_rate``
     - ``(n_species,)``
     - s⁻¹
     - no
     - Largest instantaneous rate over time and nodes.
   * - ``species_ionisation_energy``
     - ``(n_species,)``
     - J/m
     - no
     - Final ionised inventory, integrated with physical transverse weights.
```

The total electron-density statistic sums the species at that same node; it does
not sum independently selected species maxima. Periodic species fractions and electron
densities use these same definitions between save planes. Pulse work subtracts the
initial inventory. The [mixture guide](../media/mixtures.md) gives the formulas and processing
accessors, including node fluence and the separately attributed linear absorption.
