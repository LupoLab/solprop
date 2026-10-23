(primer)=
# From Maxwell to solprop: a long-form primer

This primer is a long, continuous read about the physics and numerics behind solprop. It
is written for someone who is comfortable with computation and who knows nonlinear and
ultrafast optics, but who wants to see, in one place and in order, how the pieces fit
together. It starts from Maxwell's equations, derives the unidirectional pulse propagation
equation (UPPE) and the generalised nonlinear Schrödinger equation (GNLSE), and then works
through waveguide modes, multimode and free-space propagation, the material responses, and
the numerical methods that solve the equation and the checks that tell you when to believe
the answer.

It is different from the rest of the manual. The [user guide](../guide/index.md) is
reference material for people who already know what they want: it states equations,
conventions and evidence compactly and precisely, spread over many pages. Here the same
material is unpacked slowly, with the reasoning behind each choice, the relevant
literature, and small worked numbers. Much of it repeats what the guide says, on purpose,
gathered into one narrative. Each chapter ends by pointing to the guide pages it
consolidates, so that you can move on to the precise statements once the ideas are
familiar.

This is not a code tutorial. There are no code blocks to run. For runnable calculations,
see the [introduction](../introduction/index.md) and its lessons, and the
[examples](../examples/index.md).

## How to read it

The chapters are meant to be read in order. The whole primer is a day or two's reading.
Later chapters lean on earlier ones, and the numerical chapters assume the equation of
Chapter 2. If you already know a topic, skim its chapter for the notation and the "What to
remember" box at the end.

| Part | Chapters | What you will understand |
|---|---|---|
| Foundations | [1. Maxwell's equations to a propagation problem](maxwell.md), [2. The UPPE](uppe.md), [3. From the UPPE to the GNLSE](gnlse.md) | Conventions, the one approximation behind the UPPE, the moving frame, envelopes, and how $\gamma$, the shock term and the Raman fraction arise |
| Guided light | [4. Waveguide modes](modes.md), [5. Many modes, and guides that change](multimode.md) | Capillary, antiresonant, planar and solid-core modes; normalisation and true power; multimode projection, coherent walls, tapers and vector modes |
| Free space | [6. Beams in free space](free-space.md) | The exact $k_z$, Hankel and Cartesian grids, critical power, filamentation, and what symmetry excludes |
| The medium | [7. Linear response and material data](linear-media.md), [8. The Kerr effect](kerr.md), [9. Raman scattering](raman.md), [10. Ionisation, plasma and noise](ionisation.md) | What optmat supplies, the cubic and its factors, dispersive Kerr, Raman conventions, rates, plasma currents and the noise seed |
| Numerics | [11. Putting the field on a grid](grids.md), [12. Marching in z](stepping.md), [13. Knowing when to believe a result](accuracy.md) | Grids and aliasing, the interaction picture, the exact varying operator, the integrators and their control, and convergence |
| Synthesis | [14. Putting it together](synthesis.md), [References](references.md) | The whole chain on one page, how to choose a model, a glossary, and the literature |

## Notation and conventions

The primer uses SI units throughout, like the code. Its equations are written in the
**optics Fourier convention** of solprop's interface,

$$
\tilde E(\omega)=\int E(t)\,e^{+i\omega t}\,\mathrm dt,\qquad\text{plane wave }e^{i(\beta z-\omega t)},
$$

whereas solprop's internals and many equations in the guide's numerical chapters use the
conjugate FFT sign. [Chapter 1](maxwell.md) gives the dictionary between the two.

| Symbol | Meaning |
|---|---|
| $E(t)$, $\tilde E(\omega)$ | Real electric field and its spectrum |
| $E_a$, $A$ | Analytic signal; envelope, with $E=\operatorname{Re}E_a=\operatorname{Re}\{Ae^{-i\omega_0t}\}$ |
| $\omega_0$, $\omega_{\rm ref}$ | Carrier and reference frequencies |
| $n_{\rm eff}$, $\beta$, $\alpha$ | Effective index; $\beta=(\omega/c)\operatorname{Re}n_{\rm eff}$; power loss $\alpha=(2\omega/c)\operatorname{Im}n_{\rm eff}$ |
| $k_z$, $k_\perp$ | Longitudinal and transverse wavenumbers in free space |
| $\beta_0$, $\beta_1$ | Phase and group delay removed by the moving frame |
| $D$, $\Phi$ | Linear operator and its integral along $z$ |
| $P$, $J$ | Nonlinear polarisation and free-electron current |
| $\chi^{(3)}$, $n_2$, $\gamma$ | Third-order susceptibility, nonlinear index, GNLSE coefficient |
| $\gamma^{(3)}$, $\gamma_T$ | Per-particle third-order coefficient (m⁵/V²); optmat's Taylor-convention hyperpolarisability |
| $A_{\rm eff}$ | Effective area |
| $\rho$, $N$ | Number density |
| $f_R$, $h$ | Raman fraction; Raman response function |
| $W$, $f$ | Ionisation rate; ionised fraction |
| $w$ | Power weight $\operatorname{Re}n_{\rm eff}$ in the true-power amplitude $b=\sqrt w\,a$ |

Some letters are overloaded in the literature. In particular, $\gamma$ is the GNLSE coefficient
here. The Keldysh parameter is written $\gamma$ only in Chapter 10, where no confusion arises,
and the per-particle coefficient always carries its superscript.

## Figures

The schematic figures in `docs/primer/figures/` are drawn by hand. The computed figures are
written by `docs/primer/figures/primer_figures.py` from small calculations with solprop's
public API, and each caption names its source and settings. Regenerate them with

```sh
uv run python docs/primer/figures/primer_figures.py
```

They illustrate ideas. None of them is a convergence or qualification study.

```{toctree}
:maxdepth: 1

maxwell
uppe
gnlse
modes
multimode
free-space
linear-media
kerr
raman
ionisation
grids
stepping
accuracy
synthesis
references
```
