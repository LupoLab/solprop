# Recording and inspecting provenance

A saved result needs the grid, field convention, material models, response choices and software identities that produced it. Inspect these before comparing runs or reusing a field. A record identifies the declared model and code version; it does not embed all source code or establish physical accuracy.

## The metadata schema

{py:class}`~solprop.output.schema.RunMetadata` is a frozen record with these parts:

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - field
     - what it holds
   * - ``grid``
     - A :py:class:`~solprop.output.schema.GridRecord`: the scalars that rebuild the
       frequency axis — kind, sizes, spacings, the reference frequency, the band limits.
       Scalars only, so the record stays small and the digest cheap.
   * - ``modes``
     - One :py:class:`~solprop.output.schema.ModeRecord` per propagated mode: its class,
       its effective area, its serialised configuration and its ``label``, the physical
       channel that row of the mode axis is, as
       :py:meth:`~solprop.modes.protocol.ChannelLabel.as_mapping` writes it.
   * - ``simulation_kind``
     - Which transform the run used: ``"mode_averaged"``, ``"multimode"``, ``"radial"``,
       ``"free"`` or ``"slab"``.
   * - ``modal``
     - The identity of a :doc:`general modal guide <../waveguides/general-modal>`, and ``None`` for every
       other route: the ordered channel labels, the reference tables, the basis metric
       with its measured Gram defect, the compiled per-region responses, the guide
       itself (its provider, its sampling and its materials with their declared
       responses -- the instantaneous Kerr one with the wavelength a declared ``n2``
       was measured at, and the delayed one beside it), the shared co-moving frame, the
       run's active band, the basis the noise was drawn in, and the accessors this basis
       admits.
   * - ``spatial``
     - For a radial, free-space or planar run, the spatial geometry, the meaning of each
       axis, the integration measures and the field units; ``None`` for a modal run.
   * - ``quadrature``
     - The transverse rule of a resolved modal run: its family, counts, nodes and
       physical weights, included in the resume key. ``None``
       for mode-averaged and spatial runs. See :doc:`resuming` for compatibility checks.
   * - ``materials``
     - The run's **material manifest**, a
       :py:class:`~solprop.physdata.selection.MaterialManifest`: what the run took from
       optmat and under which policies (`the material manifest below <#the-material-manifest>`_). Empty for a
       run of supplied materials only.
   * - ``configuration``
     - Every physics object that built the run, serialised.
   * - ``initial_z``
     - Absolute launch position in metres, zero by default. Profiles and modal
       normalisation retain this absolute coordinate.
   * - ``tdse``
     - The resolved scientific TDSE descriptor, or ``None``. It records the atomic
       basis and potential, calibration identity and independent source scales,
       shadow/output contract, sample lattice, atomic step limit and any explicit
       native source filter with its temporal-collar geometry. No native runtime
       handle is stored.
   * - ``stepper``
     - A :py:class:`~solprop.output.schema.StepperRecord`: the tableau by name, the
       controller, the step limits, and the fixed schedule actually taken, so a run
       can be replayed step for step.
   * - ``noise``
     - A :py:class:`~solprop.output.schema.NoiseRecord`: the model, the seed and the
       realisation itself, so an ensemble member is reproducible.
   * - ``backend``
     - A :py:class:`~solprop.output.schema.BackendRecord`: what executed the run -- the
       backend's name, the propagator mode it resolved, its threading and planning
       settings, and the memory estimate and budget
       (:doc:`running <run-access>`).
   * - ``versions``, ``git``, ``created``
     - Provenance: package versions, the commit and dirty flag of the running solprop
       checkout when there is one, and when the run started.
   * - ``reference``
     - The **plane-reference record** (output schema 24,
       `SEP 31 <../../dev/seps/sep-0031-simulation-coupling.html>`_): the carrier, the
       frame integral of the terms the linear operator removed (exact Chebyshev
       antiderivatives of the frame's ``beta_0`` and ``beta_1`` tables, with the frequency
       origin of the linear term), the reference origin of the run's path, the parent
       plane of a run launched from one, and the basis identity at each planned plane.
       :func:`~solprop.coupling.plane.reference_at` rebuilds any saved plane's phase and time
       reference from it alone; :func:`~solprop.processing.planes.field_plane` reads it
       (:doc:`coupling simulations <../workflows/coupling>`).
   * - ``schema_version``, ``spectral_convention``, ``field_layout``
     - The version of this record layout (currently 24), the sign convention of the
       stored field (always ``"optics"``) and the axis order of the stored field.
```

`backend`, `versions`, `git`, `created` and `reference` are outside the
[resume key](resuming.md): the plane-reference record is derived from the hashed
problem, and a run launched from another plane carries its inherited origin and parent
in `configuration["coupling"]`, which is inside it. Every other part is inside it, the material manifest's identities and policies included.
`versions` names optmat's full version string beside solprop's, NumPy's, SciPy's,
h5py's and Python's.

## The material manifest

A run that takes a material from optmat -- a gas, an air composition or a solid,
selected with `materials=` ([materials](../media/material-selection.md#materials-from-optmat)) or built
by the caller -- records what it took. The manifest is collected once per run, from the
run's own objects, wherever an optmat-backed one sits: the mixture, a mode's fill and
cladding, a planar grid's slab modes, a step-index core, a general guide's material
regions, a bulk medium. The walk enters any object with fields or attributes --
solprop's, a caller's subclass, a mode of the caller's own --, a `functools.partial`
and a bound method, but not a function's closure, nor numpy's, SciPy's or optmat's own
objects (optmat's are recorded through the solprop adapters that hold them). It
holds two kinds of entry.

**Identities and policies**, which are inside the resume key:

- `optmat_version`, the installed optmat's full version string;
- `records`, every resolved model's optmat identity
  ({py:class}`~solprop.physdata.selection.MaterialRecord`: name, quantity, material,
  content digest, status, overrides and their base), each digest once, keyed
  `material:kind:name`;
- `species`, each optmat population's label (or a planar slab's single-species fill)
  with the digests of its models;
- `compositions`, the named optmat composition a mixture's fractions came from, such as
  `"dry air (N2, O2)"`, with its labels and fractions;
- `solids`, each optmat solid's material, axis, model and digest, and whether its index
  is taken as transparent, with the absorption bound of that approximation;
- `policies`, the approximations in force: linear density scaling of a gas
  susceptibility, with its warning threshold; the electronic policy and the carrier
  wavelength; the Raman line-shift policy and reference temperature; and the plasma
  energy-loss convention. A policy the run does not use (a Raman shift without a
  shifting response, a GNLSE fraction off the GNLSE) is recorded as `None`.

**Measurements**, which are recorded beside them and are not in the key:

- `selection`, the {py:class}`~solprop.physdata.selection.MaterialSelection` as it was
  given. What a name resolved to is in `records`, so naming a model `"recommended"` or
  by its own name keys alike;
- `dilute_scaling`, the error of linear density scaling for each optmat population (or
  a planar slab's single-species fill) at its largest density, measured once over the
  run's whole band, collars included, and recorded whatever its size;
- each solid's `transparency`, the largest extinction and absorption coefficient a
  transparent index dropped in this run (an adapter shared by several runs records
  each run's own);
- `ranges`, the parts of the evaluated bands that lay outside a model's qualified
  intervals, as the run's range warnings named them;
- `citations`, the citation of every other material object of the run, supplied or
  solprop's own.

A run that takes nothing from optmat has the empty manifest — a run of supplied
materials only: its key holds `materials: null`. Every run of the built-in materials is an optmat
run, with a manifest.

```{code-block} python
from solprop.driver import prop_capillary

run = prop_capillary(..., "Ar", 1.0, output="run.h5")  # materials=MaterialSelection()
manifest = run.metadata.materials
[record.name for record in manifest.records]
# ['eos/Ar', 'hyperpolarizability/Ar', 'fit/Ar', 'rate/default/Ar@800nm',
#  'transport/Ar', 'fit/SiO2']: the gas, its Kerr fit and rate at the carrier, the wall
manifest.dilute_scaling[0]["relative_error"]   # about -2e-4 for argon at 1 bar
```

## Post-run annotations

Some of what a run learns about its materials is known only once it has closed. The
record is written before the first plane, so it cannot hold it, and it must not: the
record is the run's identity, and the identity of a run cannot depend on how far it
got. Such results are **post-run annotations**: plain JSON, by name, written beside the
record when the run closes and **outside the configuration hash**, so the resume gate,
a scan's reuse of a completed point and the record's digest are exactly what they were.
Both handlers carry them (`handler.annotations`, a mapping by name;
`handler.annotate(name, value)` writes one, replacing one of the same name, and a
closed file is reopened briefly for it), `MemoryOutput.to_file` copies them, and
{py:func}`~solprop.output.hdf5.open_file` reads them back. A file without any reads with
an empty mapping.

One is defined, {py:data}`~solprop.output.schema.PEAK_FIELDS_ANNOTATION`
(`"material_peak_fields"`): the material manifest's post-run diagnostic of a run with
optmat ionisation rates ([ionisation](../media/ionisation.md#rates-from-optmat)), one mapping
per population — the largest field it reached against its model's maximum field, and
the rate-weighted energy per event there against the constant the run paid. The run
writes it when it closes; the record carries only its inputs. A run without an optmat
rate writes none.

```{code-block} python
from solprop.output import open_file
from solprop.output.schema import PEAK_FIELDS_ANNOTATION

with open_file("run.h5") as stored:
    (report,) = stored.annotations.get(PEAK_FIELDS_ANNOTATION, [])
    report["largest_field"], report["reached_maximum"], report["departure"]
```

## Serialising the physics, and admitting what will not serialise

{py:func}`~solprop.output.schema.serialise` is total and idempotent. Dataclasses become
tagged dictionaries of their constructor fields, enumerations carry their qualified class
name as well as their value, NumPy arrays become nested lists with their dtype (a complex
array splitting into real and imaginary parts, since JSON has no complex number), and
mappings and sequences recurse.

What cannot be reduced is not dropped. A function, a lambda, a spline closure — the
`beta(omega)` a `SpecifiedMode` was built from, say — is recorded as its `repr`
with `reproducible: False` beside it. A configuration that is silently incomplete is
worse than one that says where it stops.

Fields declared `init=False` are skipped: they are derived in `__post_init__` from the
ones that are not, so recording them would store the same information twice.

## Tabulated materials

Cubic splines in material definitions are recorded by their knots, polynomial
coefficients, interpolation axis and extrapolation setting. Independently created
copies therefore have the same configuration hash, and changing any of those values
changes the hash. This covers tabulated cladding glasses, among others.

## Provenance, not a source snapshot

A file records the resolved configuration, the package versions, and the commit with a
dirty flag — never the source text, which would make every file carry a copy of the
package. When there is no checkout, or git is absent, or the
command fails, {py:meth}`~solprop.output.schema.GitProvenance.detect` returns `None`
rather than a sentence dressed up as a hash.
