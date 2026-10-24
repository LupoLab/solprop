---
html_theme.sidebar_secondary.remove: true
---

# The structure of solprop, in figures

Six figures of the same package the other development documents describe in prose: the
layers and the modules in each, the data flow of one run, the kernel contract section by
section, one stage of the interaction-picture step, the seam and the backends behind it,
and the output side. Each one says what to look for and which section of the text it
draws; nothing here is new information, and where a figure and a page disagree the page
wins and the disagreement is a bug in this one.

**How they are made.** `docs/dev/figures/build.py` draws them and writes
`docs/dev/figures/*.svg`; `uv run poe figures` runs it; the SVGs are committed and this
page only includes them, so the documentation builds on an install with no drawing
library. The names in the figures are kept as data in that script, and
`tests/test_structure_figures.py` checks every one of them against the package, the
kernel contract and the backend seam, and redraws every figure to compare it with what
is committed. A box that names something the code no longer has fails the fast lane.

Each figure carries its own light and dark palette and follows the reader's colour-scheme
preference: an SVG included as an image is an isolated document and receives none of this
page's CSS, so it cannot follow the theme switcher on its own. The page is wider than the
rest of the manual: `docs/_static/structure.css` releases the theme's column widths on this
page alone, selected by its section id, so the six figures render at the window's width
rather than the prose column's.

## 1. The layers

One colour per layer, used in every figure on this page and stated in the legend here.
Read the bands top to bottom in the order imports flow, and the arrows in the other
direction: each points from the layer that imports to the layer it imports. Solid boxes
are modules that exist; dashed ones are named in [architecture.md §4](architecture.md)
as candidates to be written only for an agreed use case. The numbered arrows on the
right skip a band; one of them is the one import of P that [architecture.md
§3](architecture.md) allows a backend: the grid, the modes and the profiles its
statistics are handed at construction, types and constants and never a physics object.

![The five layers of solprop, with every module in its band and the import edges between them](figures/layers.svg)

Layer letters and rules: [architecture.md §3](architecture.md) and
[decisions.md](decisions.md) ADR-001. Package tree:
[architecture.md §4](architecture.md).

## 2. One run

The spine on the left is the call sequence, from the convenience function down to the
plots; the column on the right is what each stage hands to the next, in the colour of the
layer that owns it. The annotations between them are
[architecture.md §5](architecture.md)'s own. The thing to notice is where the boundary
falls: the outer driver visits save planes, while `StepDriver.propagate_to` runs the
shared Python step loop. Reference passes, compiled kernels and library transforms,
or one native trial do the numerical work behind that loop.

![One run of solprop, from prop_capillary through the kernel problem and the step driver to a saved plane](figures/one_run.svg)

Text: [architecture.md §5](architecture.md). The seam itself is figure 5.

## 3. The kernel contract

`KernelProblem` is the only object that crosses from the physics and translation layers
into a backend, and this is it section by section: on the left what builds each section,
in the middle the field it becomes, on the right what a backend does with it. The inset
at the bottom is the one thing about the contract that is not a field — Φ and the
prefactor each take one of four representations, and `device_ready` is a predicate over
the pair.

![The eight sections of the kernel contract, what feeds each of them and what a backend does with each](figures/kernel_contract.svg)

Text: [kernel-contract.md §1](kernel-contract.md) to §8, the split in
[kernel-contract.md §3](kernel-contract.md) and
[backends.md §1](backends.md).

## 4. One stage

The seven numbered passes of one stage of the interaction-picture DOPRI5 step, each one
pass over its data and each a named function in `backends/numpy_ref/passes.py`. The
column beside pass 4 is `response_order`, the fused pass that is the whole reason the
seam is drawn where it is; the panel on the right is the step the seven passes sit
inside, and what runs once per step rather than once per stage.

![The seven passes of one interaction-picture stage, the fused response column and the DOPRI5 step around them](figures/one_stage.svg)

Text: [architecture.md §8](architecture.md) for the passes,
[stepper.md §2](stepper.md) for the tableau and FSAL, and
[backends.md §2](backends.md) for the reference functions.

## 5. The backends

The seam at the top is the whole of what a backend is. Under it is the step loop no
backend reimplements — the march, the replay, the landing, the controller, the history,
the noise and the per-stage bundle all live in `backends/driver.py` — and under that the
three implemented backends: NumPy reference, Numba CPU and native (built from `native/`). Each
supplies trial/state operations and the relevant hooks; the loop's
statistics channel and progress reporter call back through the seam. Fixed fibre
batches extend this interface with independent member observations and metadata
([kernel contract](kernel-contract.md)).

![The backend seam, the shared step driver, and the implemented backends](figures/backends.svg)

Text: [backends.md §1](backends.md) for the seam and the three things a backend owes,
[backends.md §2–4](backends.md) for the implementations, and
[backends.md §9–11](backends.md) for what runs where and the current limits.

## 6. The output side

What a finished run holds and what reads it back. The datasets on the left are the ones
`HDF5Output` creates, in the order it creates them; the box top right is the resume key,
and what it deliberately leaves out is as much of the design as what it covers.

![A saved run's datasets and metadata, the resume digest, and the accessors that read them back](figures/output.svg)

Text: [architecture.md §10](architecture.md), and `output/schema.py` for the record
types.

The output layer also owns `output/snapshots.py`: coordinated suspension of both
writers, immutable publication and verified restoration. This sits above the backend
seam and retains live native resources; persistent writer sidecars preserve ownership
across temporary HDF5 closure. See [architecture §10](architecture.md#10-output-and-statistics).
