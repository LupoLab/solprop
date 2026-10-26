# Choosing a spatial description

The spatial description determines which changes in a pulse's transverse field
the calculation can represent. Choose it from the symmetry and physics of the
experiment before deciding how fine the numerical grid should be.

## One mode, or a resolved mode set

A **mode-averaged** calculation follows one temporal amplitude and represents the
transverse profile through its effective area and propagation constant. It is
useful when a fixed guided mode is an adequate description. It cannot describe
energy transfer into another mode or a beam changing its transverse shape.

```{figure} ../guide/figures/mode_averaged.svg
:alt: A capillary mode profile beside a uniform disc with the same effective area.

Mode averaging retains an effective area rather than resolving the local field
across the mode. The distinction matters for a response such as ionisation, which
depends strongly on the local electric-field amplitude.
```

A **resolved modal** calculation follows one or more guided modes and evaluates
the nonlinear response across their combined transverse field. Even one resolved
mode differs from mode averaging: `prop_capillary(modes=1)` resolves its profile
on a transverse quadrature, and `mode_average=True` replaces it by the effective area. Adding modes allows their amplitudes and phases to
evolve and exchange energy.

```{figure} ../guide/figures/capillary_radial.svg
:alt: Three radial capillary mode profiles and the quadrature rings on which their combined field is evaluated.

A radial capillary mode set represents the transverse field as a sum of guided
profiles. The quadrature resolves the nonlinear response to their superposition.
```

Use the [GNLSE](../guide/waveguides/gnlse.md) for a single envelope described by
dispersion and nonlinearity, [capillary propagation](../guide/waveguides/capillaries.md)
for gas-filled hollow guides, and [general modal propagation](../guide/waveguides/general-modal.md)
when the full vector electric and magnetic mode fields are required.

## A beam in free space

For a cylindrically symmetric beam, a **radial grid** follows the field as a
function of radius and time. It can describe focusing and a single axial filament,
but cannot represent azimuthal breakup or two beams crossing away from a common
axis.

```{figure} ../guide/figures/radial_free.svg
:alt: A circular free-space domain represented by radial samples and an outer absorbing collar.

Radial symmetry reduces the transverse calculation to one coordinate. The outer
boundary must remain far enough from the beam to avoid influencing the result.
```

A **Cartesian grid** follows both x and y. Use it for asymmetric fields, angled
beams, or dynamics that break cylindrical symmetry. Its extra spatial freedom
requires more memory and a resolution check in both directions. A **line grid**
instead assumes uniformity in y; powers and energies are then per unit width.

```{figure} ../guide/figures/cartesian_free.svg
:alt: A two-dimensional Cartesian transverse grid with absorbing boundaries.

Cartesian samples resolve both transverse directions. Choose enough points to
resolve the beam shape and any interference fringes, and keep the useful field
away from the boundary collars.
```

See [free-space propagation](../guide/free-space/index.md) for setup and output.
Planar guides can similarly be represented as uniform in one transverse direction
or as slab modes with a resolved second direction; these are distinct physical
models, explained in [waveguide propagation](../guide/waveguides/index.md).

## General modal guides and planar descriptions

A general modal guide uses supplied or computed electric and magnetic mode
fields, including longitudinal components where the model requires them. This
is useful for solid fibres and other guides whose modes need more than the
scalar capillary approximation. Mode coefficients are coordinates of that field
expansion; use the guide's physical-field and Poynting-flux processing instead of
assuming that every coordinate has an independently additive power. See
[general modal guides](../guide/waveguides/general-modal.md) and
[solid fibres](../guide/waveguides/solid-fibres.md).

```{figure} ../guide/figures/general_modal.svg
:alt: Vector modal fields with electric and magnetic components forming a general guided field expansion.

A general modal description retains the mode fields needed to reconstruct
physical power and intensity. The guide model supplies the transverse structure.
```

A planar guide confines light across one coordinate. A per-unit-width slab model
assumes uniformity in the other coordinate, while the planar propagation route
combines slab modes with a resolved transverse line. A free-space line grid has
no slab confinement. Their energies and powers have different transverse
measures: always check whether a reported energy is in joules or joules per
metre. See [planar guides](../guide/waveguides/planar.md) and the
[free-space model](../guide/free-space/model.md).

```{figure} ../guide/figures/planar_hybrid.svg
:alt: Slab modes across the confined coordinate combined with samples along the other transverse coordinate.

Planar propagation combines guided structure in one direction with a resolved
transverse coordinate in the other. It is distinct from free-space line
propagation and from a slab assumed uniform along its width.
```

| Physical question | Start with | Then check |
|---|---|---|
| Is a fixed guided profile sufficient? | Mode averaging; [N = 3 lesson](lessons/plot_02_soliton.rst) | Effective area and validity of the chosen response |
| Does nonlinear mode coupling matter? | Resolved modes; [RDW lesson](lessons/plot_01_rdw.rst) | Modal basis size and transverse quadrature |
| Does the guide vary along its length? | Local modes; [varying profiles](../guide/waveguides/profiles.md) | Profile resolution and coupling assumptions |
| Is the beam cylindrically symmetric? | Radial grid; [filament lesson](lessons/plot_03_filament.rst) | Radial resolution and excluded azimuthal dynamics |
| Do beams cross or break cylindrical symmetry? | Cartesian grid; [crossing-beam lesson](lessons/plot_04_crossing_beams.rst) | Both transverse sample spacings and boundary distance |

## Check the approximation, then the resolution

Increasing the number of modes or grid samples improves resolution within the
chosen description. It does not recover physics excluded by a symmetry assumption.
Check both the model and its numerical resolution when interpreting a result.

The [detailed spatial comparison](../guide/setup/spatial-decompositions.md) sets out
the expansions, units, boundaries, and limitations of every representation.
Continue with [resonant dispersive-wave propagation](lessons/plot_01_rdw.rst), or choose a related calculation
from the [examples](../examples/index.md).
