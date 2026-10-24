# Enhancement proposals

A solprop Enhancement Proposal (SEP) is the design document for one major change — a new
model, route or numerical scheme, a changed default or convention, a redesigned
interface. It sets out the motivation, the theory with its references, the implementation
options and how the result will be verified; once accepted it is implemented in phases,
`numpy_ref` first, then `numba_cpu` and the Kokkos backend, and each phase appends its
report to the SEP. Finished SEPs are archived and linked from the manual pages they
produced. [SEP 0](sep-0000-process.md) describes the process and says when a SEP is
needed; new SEPs start from [the template](template.md). The [roadmap](../roadmap.md)
lists accepted and in-progress SEPs first, then drafts under discussion, followed by
informal work not fully covered by a proposal.

Commit hashes recorded before solprop's first public release (in proposals, reports
and benchmark entries) refer to the unpublished development history; the public history
starts at that release. Likewise, the campaign harnesses and data they cite under
`benchmarks/` (other than the standard suite's own modules) are kept with the project's
development records, not in this repository.

## Open

Proposals that are active, in draft, accepted, in progress or deferred.

| SEP | Title | Status | Type |
|---|---|---|---|
| [0](sep-0000-process.md) | Purpose and process | Active | Process |
| [1](sep-0001-raman-response.md) | Efficient causal Raman evaluation | Draft | Feature |
| [2](sep-0002-two-photon-absorption.md) | Two-photon absorption | Draft | Feature |
| [4](sep-0004-execution-autotuning.md) | Bounded execution autotuning | Draft | Feature |
| [5](sep-0005-solid-ionisation.md) | Solid ionisation and bulk filamentation | Draft | Feature |
| [7](sep-0007-laser-gain-dynamics.md) | Laser gain dynamics in solid-core fibres | Draft | Feature |
| [8](sep-0008-quadratic-crystal-optics.md) | Second-order nonlinearity and birefringent crystal optics | Draft | Feature |
| [9](sep-0009-propagation-statistics.md) | Broadband beam measurements and propagation statistics | Draft | Feature |
| [10](sep-0010-low-rank-raman.md) | Low-rank Raman kernels along density paths | Draft | Feature |
| [11](sep-0011-finite-density-gas-index.md) | Finite-density gas index and mixture optics | Draft | Change |
| [17](sep-0017-gas-two-photon-ionisation.md) | Broadband two-photon continuum response beside strong-field ionisation | Accepted | Feature |
| [21](sep-0021-absorbing-boundaries.md) | Temporal boundaries: wider windows, absorption and the escaped field | Draft | Change |
| [31](sep-0031-simulation-coupling.md) | Coupling simulations and transporting ultrafast fields | Accepted | Feature |
| [32](sep-0032-raman-maxwell-bloch.md) | Raman Maxwell–Bloch dynamics | Draft | Feature |
| [33](sep-0033-single-precision-mlx.md) | Single-precision propagation, and an MLX backend for Apple GPUs | Draft | Feature |

SEP 13, the standalone numerical prerequisite for SEP 12, SEP 14, which carried it to
the compiled backends, and SEP 16, which unified the multistep family and made it the
default, are final and archived; numbering records creation order, not implementation
order.

## Archived

Proposals that are final, rejected, withdrawn or superseded; the
[archive](archive/index.md) lists them.

| SEP | Title | Status | Type |
|---|---|---|---|
| [3](archive/sep-0003-optmat-materials.md) | Materials, Raman responses and ionisation rates from optmat | Final | Change |
| [6](archive/sep-0006-dispersive-nonlinearity.md) | Frequency-dependent electronic third-order response | Final | Feature |
| [12](archive/sep-0012-tdse-nonlinear-response.md) | TDSE-driven nonlinear response for gas-filled capillaries | Final | Feature |
| [13](archive/sep-0013-expensive-response-steppers.md) | Low-call integration for expensive nonlinear responses | Final | Feature |
| [14](archive/sep-0014-stepper-efficiency.md) | Step efficiency across routes: limits, collars, tables and dense output | Final | Change |
| [15](archive/sep-0015-exponential-adams.md) | Exponential Adams stepping for dispersive, broadband problems | Superseded | Feature |
| [16](archive/sep-0016-unified-multistep.md) | Unified multistep integration and work–precision qualification | Final | Feature |
| [18](archive/sep-0018-band-exact-plasma-integrals.md) | Band-exact running integrals for the field-grid plasma | Final | Change |
| [19](archive/sep-0019-field-scale-quadrature.md) | Field-scale radial quadrature for single-mode capillaries | Final | Feature |
| [20](archive/sep-0020-hacc-helium-tdse.md) | haCC helium TDSE target | Final | Feature |
| [22](archive/sep-0022-memory-lean-spatial-execution.md) | Memory-lean spatial execution: statistics-only planes, construct-once runs and operator settings | Final | Change |
| [23](archive/sep-0023-far-field-aperture-spectra.md) | Far-field aperture and angular-region spectra for free-space runs | Final | Feature |
| [24](archive/sep-0024-low-storage-interaction-picture-runge-kutta.md) | A low-storage interaction-picture Runge–Kutta march for memory-bound fixed-step runs | Final | Feature |
| [25](archive/sep-0025-rounding-level-reproducibility.md) | Rounding-level reproducibility and the performance it frees | Final | Change |
| [26](archive/sep-0026-edge-free-tdse-source.md) | An edge-free gated TDSE source: hold the current instead of switching it off | Final | Change |
| [27](archive/sep-0027-held-plasma-current.md) | An edge-free field-grid plasma source: hold the current at the temporal collar | Final | Change |
| [28](archive/sep-0028-free-space-evaluation.md) | Free-space evaluation at the memory bound | Final | Change |
| [29](archive/sep-0029-compiled-backend-distribution.md) | The native backend in solprop, distributed as binaries | Final | Change |
| [30](archive/sep-0030-standard-benchmark-suite.md) | Standard benchmark suite | Final | Process |
| [34](archive/sep-0034-medium-flux.md) | Medium flux for free-space propagation | Final | Change |

```{toctree}
:hidden:
:maxdepth: 1

sep-0000-process
sep-0001-raman-response
sep-0002-two-photon-absorption
sep-0004-execution-autotuning
sep-0005-solid-ionisation
sep-0007-laser-gain-dynamics
sep-0008-quadratic-crystal-optics
sep-0009-propagation-statistics
sep-0010-low-rank-raman
sep-0011-finite-density-gas-index
sep-0017-gas-two-photon-ionisation
sep-0021-absorbing-boundaries
sep-0031-simulation-coupling
sep-0032-raman-maxwell-bloch
sep-0033-single-precision-mlx
template
archive/index
```
