# Waveguide modes

Choose a mode model from the guide geometry and the field information the
calculation needs. The table links each family to its equations, construction,
assumptions and validity limits. Start with the practical workflows for
[capillaries](capillaries.md) or [solid fibres](solid-fibres.md), and use
[running a simulation](../setup/running.md) to choose an entry point.

A **mode** here is an object that supplies a complex effective index
$n_{\rm eff}(\omega)$ (its real part the dispersion, its imaginary part the loss), the
analytic frequency derivative of that index, an effective area, and, for multimode runs,
a transverse field profile. Every class in the table lives in {mod}`solprop.modes`.

| model | class | use it for | run it with | section |
|---|---|---|---|---|
| Marcatili capillary | {class}`~solprop.modes.capillary.MarcatiliMode` | gas-filled hollow capillaries: HE, TE and TM modes, tapers, pressure gradients, mixtures | `prop_capillary` | [the Marcatili model](capillary-model.md#the-marcatili-model) |
| coherent wall | {class}`~solprop.modes.capillary.CoherentWall` | a multimode capillary whose wall couples the modes, straight or tapered; separate TE and TM wall impedances | `prop_capillary(..., wall=...)` | [the coherent wall](coherent-wall.md#the-coherent-wall) |
| antiresonant tube lattice | {class}`~solprop.modes.antiresonant.VincettiMode` | single-ring antiresonant hollow-core fibres: loss and dispersion from published fits | `prop_capillary(..., lattice=...)` | [the Vincetti model](antiresonant.md#antiresonant-tube-lattice-fibres-the-vincetti-model) |
| tabulated index | {class}`~solprop.modes.tabulated.IndexTable`, {class}`~solprop.modes.tabulated.TabulatedHollowMode`, {class}`~solprop.modes.tabulated.TabulatedMode` | measured or separately computed $n_{\rm eff}(\omega)$: an evacuated hollow guide filled with gas, a filled guide as a whole, or a solid-core fibre | `prop_capillary(modes=[...])`, `prop_gnlse(fibre=...)` | [tabulated data](tabulated-modes.md#modes-from-tabulated-data) |
| step-index fibre | {class}`~solprop.modes.stepindex.StepIndexMode` | solid-core fibres and nanowires, exact vector modes | `prop_gnlse(fibre=...)`; several modes on the general modal route | [step-index fibres](step-index.md#step-index-fibres) |
| hollow rectangle | {class}`~solprop.modes.rectangular.RectMode` | hollow rectangular guides with dielectric or metal walls | `prop_waveguide` | [rectangular guides](rectangular.md#hollow-rectangular-waveguides) |
| hollow planar (slab) | {class}`~solprop.modes.slab.SlabMode`, {class}`~solprop.modes.slab.PlanarGrid` | planar compressors: per unit width when uniform in y, or with the y direction resolved | `prop_waveguide`, `prop_planar` | [planar guides](planar.md#hollow-planar-waveguides) |
| full vector fields | `StepIndexMode.electromagnetic_field`, {class}`~solprop.modes.guide.ModalGuide` | several solid-guide modes with their E and H fields, on the [general modal route](general-modal.md) | `prop_waveguide(..., provider=guide)` | [full fields](vector-fields.md#full-electric-and-magnetic-mode-fields) |
| radial graded index | {class}`~solprop.modes.graded_index.GradedIndexProvider` | graded-index fibres: TE₀ₘ, TM₀ₘ or weak-guidance LP sectors | `prop_waveguide(..., provider=guide)` | [graded-index providers](graded-index.md#radial-graded-index-providers) |

The [capillary model](capillary-model.md) establishes conventions that the
Vincetti, tabulated-hollow, rectangular and slab models share. These models
share its gas fill, its loss policy (`loss=True`, `False`, a scale or a callable), its
way of forming $n_{\rm eff}-1$ and its analytic derivatives. Each model chapter explains its additional assumptions and differences. Every way solprop represents the transverse field, these
modes among them, is set side by side with its figure and its validity in
[spatial decompositions](../setup/spatial-decompositions.md). Where solprop's choices for these
models differ from Luna.jl's, the differences are listed in
[Noteworthy differences from Luna](../workflows/luna.md).

Polarised propagation uses either orthogonal mode rows or a two-component state with
a shared scalar spatial profile. See [polarisation and azimuthal modes](../setup/polarisation.md)
for a worked input example, component inference, normalisation and selection rules.

```text
Orthogonal mode rows                 Shared spatial profile
state: (2 modes, 1 pol, omega)        state: (1 mode, 2 pol, omega)
     | synthesis resolves x,y             | scalar synthesis on each pol
     +-------------------------+----------+
                               |
                     physical Ex, Ey at nodes
                               |
                  vector Kerr / magnitude-based plasma
                               |
                   projection back to each state layout
```
