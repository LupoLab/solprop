# Resuming a calculation and checking compatibility

Continue a saved calculation only when the saved state and the requested problem are compatible. Solprop checks a resume identity and completion state before proceeding. A deliberately changed physical model should instead start a [new segment from a saved field](interior-launch.md).

## The resume key

{py:meth}`~solprop.output.schema.RunMetadata.configuration_hash` is a SHA-256 of the
canonical JSON of the schema version, the grid, the modes, the configuration, the noise,
the stepper settings, the material manifest's identities and policies, the stepping
method when the Adams integrator
([SEP 13](../../dev/seps/archive/sep-0013-expensive-response-steppers.md),
[the Adams integrator](../numerics/adams.md)) stepped the run instead of the tableau,
named with its formulation, and the region the error norm scores. A run whose field
plasma integrates by the band rule
([SEP 18](../../dev/seps/archive/sep-0018-band-exact-plasma-integrals.md)) records the
rule in its configuration as `plasma_integration`. The record holds the policy, `K`,
$\theta_b$, the taps at full precision, the certified error, and the boundary and
exposure policies. A resume that would integrate the plasma differently is therefore
refused. The file's root attributes repeat it as `plasma_integration`,
`plasma_integration_half_width`, `plasma_integration_theta_b`, `plasma_integration_taps`
and `plasma_integration_certified_error`. A field-plasma run also records its
held-current source equation under `configuration.plasma_source`, whose certificate
limit and taper counts are part of the identity; the measured certificate is execution
evidence under `backend.settings.plasma_source`, outside the hash. A resolved modal run
records its transverse rule in `quadrature` and in the key -- the family
(`radial_rule`), the requested and realised counts, the nodes and the physical weights
([SEP 19](../../dev/seps/archive/sep-0019-field-scale-quadrature.md)) -- so a resume on
another rule is refused. A run on the plain trapezoid records none of these, and a
response's `integration` is left out of the key when it is `"trapezoid"`.
The key also includes the absolute launch position, the scientific TDSE descriptor, the
interval-cap and start-up rules, interpolation of save planes (`interpolate_planes`) and
saving of accepted-step endpoints (`step_planes`)
([SEP 14](../../dev/seps/archive/sep-0014-stepper-efficiency.md)). It is *not* Python's
`hash`, which is salted per interpreter,
so the same run digests to the same string in every process — a test runs it in a
subprocess with a different `PYTHONHASHSEED` and compares.

For a TDSE response the identity includes the resolved atomic window, the source gate
and its breakpoints, so a run never resumes under another gate or window
([the edge-free gated source](../media/tdse.md#the-edge-free-gated-source)).

Four things are deliberately outside it: the wall-clock start time, the package
versions, the git provenance and the backend record. A NumPy upgrade, an unrelated commit
or a change of backend must not refuse to resume a run that is otherwise the same run.
So is the boundary check's budget and guard width: a run stopped at the time window's
edge may be resumed under a relaxed check, and its monitor's record continues across the
resume ([watching the window's edges](../numerics/boundaries.md#watching-the-windows-edges)).
Only the quiet-collar opt-in, which changes what the stages compute, is part of the key.

The schema version, on the other hand, is inside it, so only a file written at the
current output schema (22) can be resumed; an older file reads normally but is not
continued ([the schema version](output.md#the-schema-version)).

The material manifest is inside it too, deliberately conservatively: a resume is
refused when any optmat model's identity, any material policy or the optmat version
differs, even where the numbers the run uses would not have moved. The refusal names
the entries that changed, `materials.records.Ar:gas_refractive:fit/Ar.digest` or
`materials.optmat_version`, and says that the material manifest differs. For an
editable optmat installation the version string is fixed when it is installed, so it is
the digests, not the version, that tell two optmat trees apart.

{py:func}`~solprop.output.hdf5.check_cache` is the gate:

```{code-block} python
from solprop.output import check_cache, resume

if check_cache("run.h5", metadata):        # False when there is no file
    state = resume("run.h5")               # metadata, last plane, step history
    start_from(state.plane)                # already back in the FFT convention
```

A file written for a different run raises {py:class}`~solprop.exceptions.OutputError`
listing the fields that changed — `configuration.gamma`, `grid.n_omega`, `modes[0].kind`
— rather than saying only that a hash disagrees. Because the key covers the whole
configuration, a run can never continue from the state of a run with different physics
that happens to have the same shape.

To carry on in the same file, build the handler with `append=True`: it opens the file
read-write instead of replacing it, checks that the record offered describes the run
the file holds, keeps the file's own metadata and creation time, and numbers the next
plane and the next attempt where the file left off.

## Completion status and terminal checkpoints

Both standard handlers implement the optional
{py:class}`~solprop.output.schema.CheckpointOutput` capability. Their `outcome`
property is a {py:class}`~solprop.output.schema.RunOutcome`: `running`, `complete`,
`budget_exhausted` or `failed`, together with the actual position, a stop message,
cumulative logical and atomic call ledgers, and the latest execution facts. The
`previous_executions` records retain earlier runs of the same resumed file. These
records are outside the scientific configuration hash. A budget stop is unfinished;
it never becomes a successful result by changing the requested end position.

An ordinary exception or a call-budget stop also writes a
{py:class}`~solprop.output.schema.TerminalCheckpoint` at the last available finite
field. Normally this is an accepted step endpoint. A failure while processing a
dense observation can instead retain that interpolated field. It is separate from the regular observations: `n_saved`, `handler["z"]`,
`handler["field"]` and the per-plane statistics keep their original meanings.
Checkpoint creation uses available observations and histories and makes no new
material-statistics request. An interpolated checkpoint is labelled as such:
`resume(path)` reads it, while restarting requires the explicit
`Simulation.run(..., resume="interpolated")` policy. Ordinary `resume=True`
refuses it. TDSE uses landed save planes.

```{code-block} python
from solprop.output import open_file, resume

with open_file("run.h5") as stored:
    status = stored.outcome.status if stored.outcome is not None else "unrecorded"
    checkpoint = stored.terminal_checkpoint
    regular_planes = stored.n_saved

state = resume("run.h5")
restart_z = state.plane.z
```

Resume prefers the terminal checkpoint and computes the next scheduled plane from
its absolute position. Modal amplitudes are divided by
$\sqrt{\operatorname{Re}n_{\rm eff}(\omega,z)}$ at that same position. The Adams
march continues from the snapshot kept beside the checkpoint (below), and starts
afresh at order one where there is none; its logical call ledger and the atomic attempt
ledger remain cumulative either way, so restarting cannot replenish a budget. A checkpoint exactly on a
scheduled plane whose statistics were interrupted is saved there before stepping
on. Pending step history is attached once to the next regular save, or to the next
terminal checkpoint if execution stops again.

Scientific changes still require a new run. Device choice, batch width and execution
memory limits may change on resume and are recorded in `outcome.execution`; a new
atomic call limit cannot be smaller than the calls already attempted. The driver
validates the checkpoint and ledgers before preparing atomic resources. Resuming
an already completed file returns it without preparing another backend or atomic
runtime. A hard process kill cannot run exception recovery, so the file retains its
last flushed state and `running` status rather than a manufactured completion record.

**A run that keeps no field at its planes** (`fields="none"` or `"planes"`,
[output](output.md#statistics-only-planes-which-planes-keep-their-field)) continues only
from a plane that carries a field
([SEP 22](../../dev/seps/archive/sep-0022-memory-lean-spatial-execution.md)). The terminal
checkpoint always carries the field, whatever the run keeps at its planes: it costs one
observation when the run stops, and a run stopped by an error, a signal or a call budget
resumes from it. A file whose last regular plane kept no field and which holds
no checkpoint — what a run killed outright leaves — is refused by name:

```{code-block} text
the run's last saved plane, at z = 1.000000e-04 m, kept no field (it was saved with
fields='none' or 'planes') and the file holds no terminal checkpoint, which always
carries the field, so there is no state to continue from: a run stopped by an error or
a signal writes one, a run killed outright does not. Rerun it, or keep the field on its
last plane
```

A finished field-free file resumes as finished, running nothing. A run continuing a file
keeps its fields by the file's rule, and is refused if it asks for another.

Custom handlers implementing only {py:class}`~solprop.output.schema.Output` remain
usable for ordinary propagation, keeping every field; a field-free run needs a handler
that can hold field-free planes (`retain`), as both standard handlers do. TDSE requires the checkpoint capability and
refuses an incompatible handler before native preparation.

```{code-block} python
handler = HDF5Output("run.h5", append=True)
handler.open(metadata)                     # refused, by field, if it is another run
handler.n_saved                            # the planes already there
```

To read a finished file without writing to it, {py:func}`~solprop.output.hdf5.open_file`
opens it read-only through the same protocol: the accessors, `stat_names`, `n_saved` and
`metadata` answer from the file, `open` and `save` are refused, and a `with` block closes
it. That is the way {py:mod}`solprop.processing` reads a run after the driver has closed
its handler.

```{code-block} python
from solprop.output import open_file

with open_file("run.h5") as stored:
    energy = stored["stats/energy"]
```

## What a general modal run's key holds

A [general modal guide](../waveguides/general-modal.md) is propagated in a basis a file must describe
before a reader can say what its mode axis means, so that identity is inside the key.
Two runs on guides that differ in a core radius, a declared `chi3`, a node rule, a table
extent or the order of their channels have different digests, and a resume is refused by
the field that moved — `modal.guide.provider`, `modal.guide.regions`, `modal.tables`,
`modal.channels` — not by a bare hash mismatch.

A guide that declares a **change of channel basis** — circular partners of a degenerate
pair, see [general modal propagation](../waveguides/general-modal.md) — is in the key four times over:
`modal.metric.kind` moves from `identity` to `dense`, `modal.metric.gauge_id` and
`modal.tables.gauge_id` both gain the composed `+circular(i,j)`, `modal.channels` carries
handedness instead of parity, and `modal.guide.sampling.basis` carries the declaration
itself, matrix and all. A file written in one basis therefore cannot resume in another,
and the refusal names the field. The sampling's entry is present only where a basis is
declared, so a guide in its provider's own coordinates keeps the key it would have
without the feature; the two gauges are always compared, so the other direction is
refused too.

**The key also holds what the basis costs.** `modal.metric.occupied_departure` is how
far the *interpolated* basis is from orthonormal over the bins the run occupies —
measured at the run's own coarse bins, between the table's knots, and gated against the
same budget as the knot defect — and `modal.metric.active_departure` is the same
quantity over every bin the active band admits, weighted by nothing, which is warned on
rather than gated (see [general modal propagation](../waveguides/general-modal.md)).

**`modal.accessors` is in the record and deliberately *not* in the key.** It is the list
of statistic names the basis admits, which is a statement about what the package version
that wrote the file could compute — not about what the run computed, which is what every
other entry of `modal` says. Were it in the key, every accessor solprop gains would
invalidate every general modal file already written, for a capability those files *gain*
rather than lose. {py:func}`~solprop.output.hdf5.check_cache` compares the two lists as
a **subset** instead: a file whose list lacks a name this version admits resumes, and a
file naming an accessor this version does not admit is refused by that name.

Because the record holds the metric's **identity** and not its arrays,
{py:func}`solprop.processing.modal.physical_energy` refuses a file whose metric is not
the identity by name: the number it could form from the stored state alone is the
unweighted one, which is the answer for a different basis. The run's own recorded
`physical_energy` statistic carries the weighted value, and
{py:func}`solprop.processing.modal.reconstructed_flux` reconstructs the physical flux
from the guide's reference and needs no metric at all.

## Reading E and H back

A general modal file stores channel amplitudes, not fields. {py:mod}`solprop.processing.modal`
is the way back: given the run's own reference tables, rebuilt from the guide the record
names, it synthesises `E(x, y, t)` in V/m and `H(x, y, t)` in A/m at the guide's
quadrature nodes and forms what a detector would see.

```{code-block} python
from solprop.processing.modal import (
    channel_labels,
    electric_field,
    plane_nodes,
    reconstructed_flux,
)
from solprop.translate.general import general_modal_route

with open_file("run.h5") as stored:
    print([one.display_label for one in channel_labels(stored)])
    tables = general_modal_route(guide, grid, z_max=length, omega_reference=w0).tables
    field = electric_field(stored, tables, plane=-1, guide=guide)  # (n_nodes, 3, n_t)
    power = reconstructed_flux(stored, tables, plane=-1, guide=guide)  # (n_t,), W
    xy, weights = plane_nodes(stored, tables, plane=-1, guide=guide)  # m, m^2
```

{py:func}`~solprop.processing.modal.plane_nodes` says where those values are: the
nodes' positions `(x, y)` in the cross-section, in the order of the node axis every
field accessor returns, and the area weights the fluxes are integrated with, so that
`weights @ reconstructed_flux(..., density=True)` is `reconstructed_flux(...)`. On a
guide whose geometry moves along z the plane is reconstructed on **its own** modes: the
accessors solve the guide again at the plane's stored position, on the rule that has
moved there, so the nodes the fields are returned at, and the ones `plane_nodes`
returns, are that position's, and the flux is integrated with that position's area
weights ([a guide whose geometry moves](../waveguides/moving-modal.md#a-guide-whose-geometry-moves-along-z)).

The tables are **not** stored in the file: they are tens of megabytes per run and are
rebuilt from the guide in **tens of milliseconds** for a one-channel guide of 288 nodes
(under a second if the guide itself has still to be solved, which is most of that
time), and the record carries their identity instead. Every accessor
checks the tables it is given against that identity, path by path, and refuses the first
entry that disagrees, so a reconstruction is never made from the wrong guide.

**The guide is required.** A table identity records the run's own extents and the node
count and says nothing about the frequencies the provider was solved at or what each
region declared, so two guides sampled at different interior knots, or declaring
different `chi3`, build tables whose identity agrees entry for entry. `guide=` is
checked against `metadata.modal["guide"]` — the sampling with its knots, and the regions
— and the refusal names the path: `modal.guide.sampling.omega.4`, or
`modal.guide.regions.0.kerr.chi3`. The keyword has **no default** on any of the six
accessors that reconstruct from the tables (`electric_field`, `magnetic_field`,
`reconstructed_flux`, `instantaneous_flux`, `coordinate_intensity` and `plane_nodes`):
every caller has the guide, because that is what the tables were rebuilt from, and a check that
closes the hole only when it is asked for is not a gate. A call that omits it raises
`TypeError`. A record whose identity block is a **subset** of the rebuilt one is
refused as incomplete rather than passing on the entries it happens to carry: what was
not compared was not checked.

A file written in a **declared** channel basis needs the guide rebuilt with that same
declaration, `ModalGuide.from_provider(..., basis=circular_pair(...))`: the sampling's
`basis` entry is part of the identity, and a guide in the provider's own coordinates is
refused at `modal.guide.sampling.basis`.

{py:func}`~solprop.processing.modal.electric_field` and
{py:func}`~solprop.processing.modal.magnetic_field` take `convention=` like every other
processing accessor and with the same `"optics"` default; the real field is
convention-free, and `analytic=True` gives the complex signal in the convention asked
for.

Beside the two fluxes, {py:func}`~solprop.processing.modal.instantaneous_flux` keeps the
sum-frequency term of the real fields that the cycle-averaged
{py:func}`~solprop.processing.modal.reconstructed_flux` drops, and
{py:func}`~solprop.processing.modal.coordinate_intensity` gives the flux density each
channel would carry alone. That last one is **not additive**: the sum over channels is
the physical density only where the channels do not interfere, and nothing is clipped to
hide the difference. {py:func}`~solprop.processing.modal.channel_amplitudes` is the
inverse map, through the coefficient dual; reprojecting a field that was not synthesised
from this basis creates a new initial-value problem and is not a resume path.

An ordinary modal or spatial run is
refused every one of these names: it carries neither the certified metric nor the E/H
reference they promise. Which names a basis admits is {py:mod}`solprop.output.capability`'s
table, and it is checked at **four** doors: here, when the run's statistics are admitted,
in each backend's `stats`, and in {py:func}`~solprop.processing.time.energy` and
{py:func}`~solprop.processing.time.peak_power`
([statistics](statistics.md#which-names-a-basis-admits)).

## Restoring a live-run snapshot

[Periodic snapshots](output.md#periodic-snapshots-while-computing) preserve a running
accepted endpoint while the original process continues. Use `read_snapshot` to inspect
a complete generation and `restore_snapshot(generation, new_directory)` to verify and
copy it into a new writable directory. Restoration checks every file's size and
SHA-256, configuration and provider identities, checkpoint position, retained history,
archive UUID/session and recorded counters. Missing manifests, incomplete staging,
mismatched pairs and altered files are refused before working files are published.

The returned `SnapshotFiles.output` and optional `SnapshotFiles.tdse_archive` are the
arguments for the existing `resume=True` workflow. Reconstruct the same simulation;
the normal configuration and provider compatibility rules still apply. The original
archive identity, saved data, Adams history and cumulative accounting continue from
the checkpoint. A fresh process adds an ordinary resume session to the restored
archive. Taking a snapshot in the original process adds no session. Cumulative atomic
call limits remain cumulative: restoring a snapshot does not grant a new budget.

Snapshots never claim that propagation finished. Even one taken at the requested
end is a `running` endpoint; the normal resume workflow establishes completion.
Independent resumed computations meet the repository's accuracy-envelope policy;
stored copies, history arrays and structural identifiers remain exact.
