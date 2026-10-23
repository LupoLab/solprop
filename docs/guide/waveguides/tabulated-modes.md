# Modes from tabulated data

Measured or independently calculated effective indices can supply dispersion and loss
without an analytic guide model. First decide whether a table describes an evacuated
hollow guide, a filled guide, or a solid-core fibre: adding gas to an already filled
table would count the material twice.


When the effective index comes from a measurement or from a separate mode solver, a
table of it can stand in for an analytic model. There are three ways to use one:

- an **evacuated hollow table** plus a declared gas,
  {class}`~solprop.modes.tabulated.TabulatedHollowMode`, which adds the gas's
  susceptibility to the table and so follows a pressure change or gradient
  ([below](#an-evacuated-hollow-table-and-its-gas-fill));
- a **filled hollow table**, {class}`~solprop.modes.tabulated.TabulatedMode`, whose
  table is the whole index of one particular fill
  ([below](#a-filled-hollow-table-and-choosing-area));
- a **solid-core fibre**: the bare {class}`~solprop.modes.tabulated.IndexTable` given to
  `prop_gnlse` as its `fibre=`
  ([below](#solid-core-input-and-checking-the-interpolant)).

The two hollow classes keep Marcatili's transverse field, so they run on the capillary
route as a supplied `modes=[...]` set. A table never extrapolates, and interpolation
between its rows is the main source of error; the last subsection says how to check it.

{class}`~solprop.modes.tabulated.IndexTable` takes loaded arrays of angular
frequency and complex effective index, with an optional effective-area column.
There is no file-format convention: read the rows with NumPy, h5py or a loader
of your choice, then call `IndexTable(omega, neff_values)` or
`IndexTable.from_wavelength(wavelength, neff)`. All units are SI. Both
constructors accept the rows in either order and sort them into increasing
frequency. The second argument of the plain constructor is `neff_values`, not
`neff`, because a field and a method cannot share one name:
`table.neff(omega)` evaluates the interpolant.

The table copies its arrays and makes them read-only. It needs at least four
distinct positive frequencies, finite indices with nonnegative imaginary part,
and positive finite areas when present. It interpolates real refractivity
$\operatorname{Re}(n-1)$ with a cubic spline in angular frequency,
and differentiates that spline analytically. Default loss interpolation is a
cubic spline of $\log\operatorname{Im}n$, which stays positive between knots
when loss spans many decades. It requires strictly positive imaginary rows.
For data with zero-loss rows, or to interpolate the imaginary index itself rather than
its logarithm, use `loss_interpolation="linear"`: a cubic spline of $\operatorname{Im}n$.
If that spline undershoots zero, both its value and its derivative are clamped to zero,
with one warning log per table.

`IndexTable.from_loss(wavelength, index, loss_db_per_m, ...)` converts power loss
using $\alpha=\ln(10)\,L_{\rm dB}/10$ and
$\operatorname{Im}n=c\alpha/(2\omega)$. The table's `alpha` is in m⁻¹;
its `beta`, `dbeta_domega`, `refractivity`, `dneff_domega` and
`effective_area` use the same interpolants. Area interpolation is cubic;
a nonpositive interpolated area is refused, rather than clipped.

## An evacuated hollow table and its gas fill

For a separated longitudinal and transverse wavevector,

$$
\beta^2=n_g^2 k^2-\kappa^2,\qquad k=\omega/c.
$$

An evacuated table gives $\kappa^2/k^2=1-n_v^2$. Holding that transverse
quantity fixed while adding gas susceptibility $\chi_g=n_g^2-1$ gives

$$
n_{\rm eff}^2=n_v^2+\chi_g.
$$

{class}`~solprop.modes.tabulated.TabulatedHollowMode` applies this rule through
the full Marcatili evaluation. For $r=n_v-1$ it supplies
$w=-r(2+r)$ and $w'=-2(1+r)r'$; these preserve the small refractivity and
analytic derivative. It evaluates the declared gas, and never samples silica.
The rule is exact for Marcatili's fill-independent waveguide term: at the
table's own knots it reproduces the same `MarcatiliMode` to rounding, on a mixture's
rationalised arithmetic as well as on a single gas's. Between knots the error is the
spline's: doubling the knots from 50 to 100 to 200 reduces the index error by factors of
15.98–16.26 and the derivative error by 8.13–8.36, fourth and third order. The
transverse fields, normalisation and polarisation partners are Marcatili's.

```{code-block} python
import numpy as np
from solprop.constants import C
from solprop.modes import IndexTable, MarcatiliMode, TabulatedHollowMode

omega = 2 * np.pi * C / np.linspace(200e-9, 2e-6, 200)
vacuum = MarcatiliMode(15e-6, cladding=1.45)
table = IndexTable(omega, vacuum.neff(omega))
mode = TabulatedHollowMode(15e-6, "Ar", 10.0, table=table)
```

A pressure gradient varies the gas contribution while keeping the table and
radius fixed; a taper is refused. `model="reduced"` and a supplied cladding
are also refused. The usual loss policy acts after the table model: keep,
remove, scale or replace its total attenuation. Hollow-table species
absorption is still attributable to the declared gas.

The rule neglects changes in wall contrast and the gas-driven resonance shift: the
filled table keeps its wall resonances where the evacuated guide has them. A gas lowers
the wall's index contrast, and the antiresonance condition
$\lambda_m \propto t\sqrt{n_w^2-n_g^2}$ moves every resonance by
$\Delta\lambda/\lambda \approx -\chi_g/[2(n_w^2-1)]$: about a quarter of a per cent
towards shorter wavelengths (−0.24 % at 800 nm) for 10 bar of argon in a silica wall.
Close to a resonance the filled guide's index and loss therefore differ from the
table's by more than the smooth-row figures below. The rule's error
is measured on the antiresonant fibre whose finite-element data are described in
[accuracy against a finite-element solution](antiresonant.md#accuracy-against-a-finite-element-solution), with the evacuated FEM rows as the
table and the 10-bar argon rows (293 K, `kingston` source) as the answer. Rows within
3 % of a wall resonance are counted as near it; the split is descriptive, not a
validity region.

- **Real index.** On the nine smooth rows away from resonance (690–946 nm) the rule
  predicts the filled index to **2.24544e-8**, against **6.3e-7** from simply adding the
  gas index to the evacuated one. Near resonance the largest residual is
  **2.82871e-5**.
- **Loss.** The ratio of FEM to table loss is **0.99261–0.99843** on the nine smooth
  rows, but **0.85867–0.99843** over all fourteen rows away from resonance: it falls to
  0.86 at the short-wavelength end. The rule's transfer of loss is therefore trusted on
  the smooth rows only.

Those numbers describe one fibre.

A second diagnostic fills a table built from the Vincetti model itself (a 15 µm core,
300 nm walls, eight tubes of 6 µm outer radius, cladding index 1.45, 400 knots over
400–1200 nm; nominal first wall resonance at 630 nm) and compares it with the Vincetti
model evaluated at the filled pressure. At 10 and 30 bar the largest real-index
differences reach 9.19e-6 and 2.69e-5; near-resonance loss ratios span 0.527–1.673
and 0.161–2.558. Part of that is interpolation rather than the fill rule: even the
evacuated table, where the rule does nothing, has $\beta_2$ interpolation residuals up
to 4564 fs²/m, because a 400-knot cubic does not resolve every fitted narrow
resonance, and the largest of them lie *outside* the nominal resonance window, at the
fitted tube-mode cutoffs (inside it the maximum is 2888 fs²/m). A wall-contrast
refinement of the rule is not implemented.

## A filled hollow table, and choosing area

{class}`~solprop.modes.tabulated.TabulatedMode` takes the table as the
**whole** complex index. It adds no gas susceptibility and returns no
species absorption contribution: that loss is already in the table.
Marcatili fields still supply the transverse profile. Declare gas or a
mixture for its nonlinear responses, with constant material paths. **A table
is one fill:** a 10-bar argon table paired with a 20-bar nonlinear material
is a modelling error the code cannot detect. Varying material paths are
refused because this table cannot describe a changing fill.

```{code-block} python
from solprop.modes import TabulatedMode

# filled_table contains the measured or computed index of this same 10-bar fill.
# mode = TabulatedMode(15e-6, "Ar", 10.0, temperature=293.0,
#                      table=filled_table, reference_wavelength=800e-9)
```

When an area column exists, both hollow classes require
`reference_wavelength`, inside the table. Propagation uses the table area
at that fixed reference; `mode.effective_area(omega)` exposes its spectral
variation. With no area column, omit `reference_wavelength` and the inherited
Marcatili area is used. Normalisation and transverse fields remain Marcatili's
even when the table supplies propagation area. On the finite-element fibre above, the
evacuated and filled areas differ by at most 1.02e-4 relative on the smooth rows, so an
area column is insensitive to the fill to that level. The radius still defines the field
shape, not the loaded dispersion.

## Solid-core input and checking the interpolant

Pass an `IndexTable` as `fibre=` to `prop_gnlse` or `Simulation.from_gnlse`:
`prop_gnlse(None, length, fibre=table, n2=..., ...)`. The table area at
`lambda0` resolves $\gamma=n_2\omega_0/(cA_{\rm eff})$; a table without area
requires `gamma` and uses the GNLSE convention 1 m². A supplied
Mode is wrapped by
{meth}`~solprop.nonlinear.gnlse.SpecifiedMode.from_mode`:
its dispersion, analytic first derivative, loss and area are retained, but
its fill is ignored as a nonlinear material because `gamma` or `n2`
defines that response. A mode whose geometry moves along z is carried as a
pair of declared paths -- dispersion and area together -- rather than frozen
at the entrance; see the [GNLSE page](gnlse.md). The record nests the original mode
under the GNLSE mode kind. Use the capillary route when the fill should drive nonlinear
responses. See [running a simulation](../setup/running.md) and the
[executed examples](../../auto_tutorials/waveguides/plot_02_tabulated_modes.rst).

Extrapolation is always refused. A propagation's active band includes the
spectral collars outside `wavelength_limits`, and dispersion probes such as
`beta2` evaluate neighbouring frequencies. Inspect
`grid.omega[grid.active_mask]` before running; provide additional support for
any derivative probes. The refusal gives both requested and table ranges in
rad/s and nm. Narrow the grid or obtain wider data rather than extending an
unverified spline.

Knot agreement checks data conversion, not resolution between rows. Leave
one interior row out, rebuild, and compare the prediction at that row; do
not leave an endpoint out and then extrapolate. On the evacuated finite-element rows
above, the largest leave-one-out real-index error away from resonance is
**9.15112e-6**, much larger than the smooth fill-comparison residual. Compare
`beta2` with unequal-step second differences of the raw propagation constants
as well. On the filled rows the two differ by up to **7.2699 %** over the interior of
the nine first-band rows above 580 nm, whose 29.5–33.4 nm spacing averages curvature
that the derivative samples locally. This is a sparse-data diagnostic, not a physical
error bar. A linear spline of the *evacuated* rows' imaginary index crosses zero in 20
intervals between 200 and 456 nm; log interpolation avoids that particular defect but
cannot recover missing spectral detail. Refine data around resonances and ambiguous
zero-dispersion wavelengths before using them for predictions.
