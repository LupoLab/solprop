# Power, energy and intensity

This page defines the units every route presents at its interface: what a modal amplitude
in √W means, how a run's energy, power and intensity statistics are measured, and how to
convert a saved amplitude back to the one the propagation used internally. Read it before
comparing a statistic with a measurement, or before reconstructing a field from a saved
spectrum yourself.

## True-power amplitudes

A modal input and a saved modal spectrum use **true-power amplitudes**, in √W: $|b|^2$
integrated over time is the energy the mode carries, in joules, so a one-joule launch has
one joule in the model. If the internal amplitude is $a$, the saved amplitude is

$$b(\omega,z)=\sqrt{w(\omega,z)}\,a(\omega,z),\qquad w=\operatorname{Re}n_{\rm eff},$$

or the explicit constant index a run declares in its place.

Internally the propagation basis is **vacuum-normalised**: the modes, and the synthesis
matrices that turn amplitudes into fields at the quadrature nodes, are normalised with the
vacuum admittance rather than with the mode's own index. Assembly divides the pulse and the noise by $\sqrt{w(\omega,0)}$, output
multiplies each saved plane by $\sqrt{w(\omega,z)}$, and resuming a run divides again. For
a lossless gradient the cumulative linear exponent gains
$-\tfrac12\ln[w(\omega,z)/w(\omega,0)]$ ([the linear operator](../numerics/linear-operator.md)), which
conserves the true energy while the internal squared norm changes. The conversion touches
only the amplitudes and the linear exponent: the nonlinear response models and the
propagation kernels are the same with or without it.

Three routes keep the weight at one:

- the **GNLSE** mapping uses a unit-index calibration, $w=1$, so its internal and saved
  amplitudes are the same ([the GNLSE](../waveguides/gnlse.md));
- the **general modal route** normalises every channel to unit Poynting flux, so its
  amplitude is a true-power amplitude from construction and its power weight is the
  constant one ([general modal propagation](../waveguides/modal-projection.md#normalisation-and-why-the-power-weight-is-one));
- the **free-space** grids propagate a physical field rather than modal amplitudes, and
  have no modal weight at all ([below](#spatial-geometries)).

A planar guide uniform in y, and a free-space line, integrate over x per metre of y: their
powers are in W/m and their energies in J/m (the slab's amplitudes in √(W/m)), and the run
records `transverse_measure = "per_unit_width"`
([spatial decompositions](../setup/spatial-decompositions.md#planar-hollow-guides-uniform-in-y)).

## Power traces and Poynting flux

Energy and spectral windows integrate $w|a|^2=|b|^2$ with the grid's Parseval weights
([grids](../setup/grids.md#energy-and-spectral-density)). A broadband instantaneous power needs a
more careful definition, because $w$ depends on frequency. With $a_w$ obtained by inverse
transforming $w a$, run statistics use

$$P_m(t)=\sum_p\operatorname{Re}[a_{mp}(t)a_{w,mp}(t)^*].$$

`peak_power`, `peak_power_total`, `peak_power_window`, `fwhm_t` and `fwhm_t_outer`
all use this analytic-signal Poynting trace. On the axis, synthesise the electric field
$E$ from $a$ and the magnetic-field equivalent $E_w$ from $w a$:

$$S(t)=\frac{\varepsilon_0c}{2}\operatorname{Re}[E(t)\mathbin{\cdot}E_w(t)^*].$$

`peak_intensity` is the largest value of this cycle-averaged flux; the real instantaneous
flux's sum-frequency term is not included. For a monochromatic single mode it is exactly
$\operatorname{Re}n_{\rm eff}\,\varepsilon_0c|E|^2/2$. Gas ionisation rate models use the
physical electric-field amplitude in V/m; converting their argument to this intensity
would change the model.

The squared modulus $|b(t)|^2$ has geometric-mean cross terms
$\sqrt{w(\omega)w(\omega')}$, whereas the flux has arithmetic-mean cross terms
$[w(\omega)+w(\omega')]/2$. Their difference is second order in the relative index
variation, and their time integrals agree exactly by Parseval. Standalone array functions
such as `stats.peak_power(grid, b)` measure $|b(t)|^2$, because they are not given a
mode's index; the statistics recorded during a run measure the flux. Negative excursions
of the flux are possible in principle and are not clipped.

The general modal route does not use these names: its statistics are `physical_energy`
and `analytic_flux`, functionals of the flux metric of its own basis
([noise, statistics and what comes back](../waveguides/general-modal.md#noise-statistics-and-what-comes-back)).

## Working with a saved amplitude

```{code-block} python
from solprop.fourier import power_amplitude
from solprop.stats import energy
from solprop.translate.linop import power_weight_at

problem = simulation.translate()
weight = power_weight_at(problem, 0.0)
b = power_amplitude(problem.initial.field0[0], weight)
launch_energy = energy(simulation.grid, b).sum()
```

Here `simulation` is any modal simulation (for example one built by
`Simulation.from_capillary`). The saved spectrum uses the optics Fourier convention
([grids](../setup/grids.md#fourier-signs-and-the-phase-origin));
`output.fields("fft")` changes its sign convention but keeps true-power units. To
reconstruct a physical modal electric field, first divide by the plane's $\sqrt w$, then
apply the vacuum-normalised synthesis matrix. The file records
`power_convention="poynting"`.

A saved run carries its own weight, so a file needs no problem and no mode object:
{py:func}`solprop.processing.mixtures.get_power_weight` returns the weight at every saved
plane, shape `(n_z, n_modes, n_pol, n_omega)`, bit for bit what the translation
evaluated there.

```{code-block} python
import numpy as np

from solprop.output.hdf5 import open_file
from solprop.processing import get_power_weight

with open_file("run.h5") as run:
    weight = get_power_weight(run)  # dimensionless Re n_eff
    internal = run.fields("fft") / np.sqrt(weight)  # vacuum-normalised a
```

The run record stores that weight factored rather than as one array per plane: a few
frequency-resolved terms and one short basis row per plane (see [output](output.md)). A
radial, Cartesian or other free-space run — which propagates a physical field and has no
modal weight — raises
{py:class}`~solprop.exceptions.ProcessingError` instead.

## Spatial geometries

Free-space runs propagate the physical field in V/m, with no modal weight. Their energy,
power, fluence and intensity are the axial Poynting flux of the medium at the measured
plane ([energy and flux in a medium](../free-space/flux.md)). Each transverse column is
weighted by the scalar axial admittance $\mathcal Y_c=\varepsilon_0c^2k_z/\omega$, the
spatial counterpart of the modal weight $w$: it depends on frequency and, through the
exact $k_z$, on the column's angle. Integrated energy uses its real part. Local
quantities synthesise $E$ and the magnetic closure $G=\mathcal Y_cE$ separately and form
$\tfrac12\operatorname{Re}(E_a\cdot G_a^*)$, exactly as the modal trace above pairs $a$
with $a_w$. For a paraxial beam in a transparent medium this is $n_0\varepsilon_0c|E_a|^2/2$,
so a launch of given energy in a medium of index $n_0$ has a field $\sqrt{n_0}$ lower than in
vacuum. The local flux is signed and not clipped, and the electric field that drives
ionisation is a separate diagnostic, {py:func}`~solprop.spatial.peak_field`.

A radial run through a pressure gradient adds the same endpoint term
$-\tfrac12\ln[Y_z(z)/Y_z(z_0)]$ to its linear exponent as a modal gradient does, so the
energy of each column is conserved while its field amplitude changes; the path must meet
an [adiabatic certificate](../free-space/flux.md#the-adiabatic-certificate).
