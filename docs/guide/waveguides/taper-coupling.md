# Mode conversion in a capillary taper

A changing bore changes the local mode profiles. Expressing the pulse in those profiles
introduces a connection that transfers amplitude between modes, even in a linear,
lossless calculation. The [worked
comparison](../../auto_examples/waveguides/plot_02_tapered_capillary.rst) shows its
effect and separates taper and wall contributions. The connection is derived in [the varying coherent-wall
model](coherent-wall.md#a-guide-that-varies-along-z) and is also used without a wall.

(the-taper-coupling-without-a-wall)=
## The taper coupling without a wall

The connection belongs to the moving basis, not to the wall, so a multimode capillary
taper **without** a wall carries it too, by default. One reason this matters: a lossless
capillary is often used as a simple model of a low-loss antiresonant fibre, and a
small-core taper of such a fibre must be able to convert between modes. In the
true-power amplitude of the local modes the equation is

```{math}
b' = \bigl(iB(z) - C(z)\bigr)\,b,
```

the walled generator with the wall term $H$ zero, except that $B$ is each channel's
**own** propagation constant $\beta_n = n_{\rm eff}\,\omega/c$ — the Marcatili index of
whatever `model=` and `loss=` the modes declare, lossless or not — rather than the wall
block's paraxial condensation of it, so that with $C$ dropped the flow propagates
exactly what the adiabatic route propagates. The assembly is the varying wall's: the
cumulative $\Phi$ carries the lowest channel for every channel, the flow's diagonal the
rest, $\overline{i(\beta_n - \beta_1)}$ (zero for the lowest channel) with each
channel's normalisation term, frozen per section of the operator's own table with the
variation and $\sqrt{w_n/w_m}\,C_{mn}$ in the explicit residual. It applies to the
contiguous $\mathrm{HE}_{11}$ to $\mathrm{HE}_{1N}$ set of plain capillary modes,
$N \ge 2$, whose radius varies
({py:func}`~solprop.modes.capillary.taper_coupling_block`); a pressure gradient moves no
mode shape and needs none. `taper_coupling=False` on
{py:func}`~solprop.driver.interface.prop_capillary`,
{py:func}`~solprop.driver.interface.prop_waveguide`,
{py:meth}`~solprop.driver.simulation.Simulation.from_capillary` and
{py:meth}`~solprop.driver.simulation.Simulation.from_waveguide` is the explicit opt-out
to the adiabatic route; beside a wall it is refused, because the wall carries the
connection. The coupling is refused by name, with the opt-out named in the message, on
an envelope grid (the coupled flow is tabulated on a field grid's increasing frequency
axis), beside a second field component or the `modified` noise model, and for a radius
profile without an analytic slope. The `input` noise model is admitted: its draw is
part of the launch, one photon per channel and bin in the entrance's
$\mathrm{HE}_{1m}$ set, and the coupling propagates it like any field. The record's
`modal.flow.model` is `"taper coupling"`, with the flow's certificate rows and no wall
ratios, so a coupled run is not resumed as an adiabatic one.

These comparisons illustrate the model's accuracy and the need to converge the retained mode set:

- **The adiabatic limit.** A slow lossless taper (30 to 29 µm over a millimetre)
  converts within the Dyson remainder of the first-order coupled-mode amplitude and
  below its integration-by-parts (Love-type) bound, and its $\mathrm{HE}_{11}$ is the
  adiabatic route's to the same remainder, phase included.
- **Power conservation.** The lossless flow conserves $\sum|b_m|^2$ to its step error,
  converging at fourth order or better.
- **A finite-element solution.** Against an independent scalar paraxial
  finite-element solution of the tapered bore in mapped coordinates — a 125 to
  62.5 µm taper over 1 cm and over 10 cm between 5 mm straight ends, lossless, at
  800 nm, launched in $\mathrm{HE}_{11}$ — twenty channels reproduce the energy in every
  channel holding more than $10^{-3}$ of it to 0.22 % (1 cm, where 36 % of the energy
  leaves $\mathrm{HE}_{11}$) and 0.016 % (10 cm, where 0.27 % does), every weaker channel
  to $5\times10^{-7}$ of the launch, and the total power to $2\times10^{-9}$. Eight
  channels are too few for the 1 cm taper, and at 400 nm, where the conversion reaches
  further up the ladder, thirty are needed.
- **The backends.** `numba_cpu` and the native backend run it within the flow-parity bounds, and a
  lossless multimode taper runs on `backend="auto"`.

**A strongly lossy taper at the default step bound.** The coupling carries each
channel's loss inside its block, and with the default `dz_max` of half the length the
flow's conditioning guard can refuse a strongly lossy multimode taper. The refusal says
to cap `dz_max`, or to pass `taper_coupling=False` for the adiabatic route.
