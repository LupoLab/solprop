# Resuming a calculation and checking compatibility

Continue a saved calculation only when the saved state and the requested problem are compatible. Solprop checks a resume identity and completion state before proceeding. A deliberately changed physical model should instead start a [new segment from a saved field](interior-launch.md).

## What a resume compares

A resume asks whether continuing the stored state with the requested problem gives the
run the user asked for. Solprop answers it by comparing the file's run record with the
record of the run about to start, path by path
({py:meth}`~solprop.output.schema.RunMetadata.compare`,
[SEP 42](../../dev/seps/archive/sep-0042-tolerant-resume.md)). Every recorded path belongs to
one of four classes:

| Class | What it holds | A difference |
|---|---|---|
| layout | what gives the stored arrays their meaning: the grid, the save planes, the launch position, the mode count and labels, the transverse axes, the quadrature counts, the noise model, the precision | refuses, with no override |
| compared | physics and launch parameters: everything no row classifies | refuses, unless the resume is forced |
| numerics | how accurately the run steps: the stepper record (`rtol`, the integrator, the step limits) and the propagator mode | continues while there is work left, with a warning, and is recorded |
| ignored | provenance -- the optmat version, model names, statuses, sources, citations -- and a launch realisation whose parameters are compared | nothing |

The table is {py:data}`~solprop.output.compatibility.CLASS_TABLE`; a path no row
classifies is compared, so a field nobody has classified is held to agreement.

**Numbers** agree element by element when
$|a_i - b_i| \le \varepsilon \max(|a_i|, |b_i|)$ with
$\varepsilon = 10^{-11}$ ({py:data}`~solprop.output.compatibility.RECORD_RTOL`), the
largest departure a change of libm, BLAS or compiler makes among a record's numbers. The
test is per element because one array may hold different quantities: the dispersion
coefficients `[β2, β3, β4]` span eighteen orders of magnitude, and a change of β4 is
seen. A few arrays with tails of rounding -- an array launch, a passed-in noise
realisation, the absorber's rates -- add a floor of $\varepsilon$ times their largest
element. The save planes, the length and the launch position are compared exactly,
since a resume does exact arithmetic on them. So a run resumes on another platform, or
after an upgrade that moved a derived table in its last bits.

**A path one record lacks** is a difference, unless a release declared its *neutral
value*, the value that reproduces the code before the path existed
({py:data}`~solprop.output.compatibility.NEUTRAL_VALUES`). A field's default is
deliberately not consulted: a new field that changes what runs compute would otherwise
agree with itself. A stored record is read as it was written, so a file from another
release of the same output schema is read, and resumes if the comparison allows it.

**Materials** are compared by the digest of each optmat model's numbers, keyed by
material, kind and digest, so a model renamed or re-graded with the same numbers is the
same model, and an optmat release that changed no number a run uses continues it. A model
whose numbers changed is refused by name, `optmat model Ar gas_refractive (digest
f0efa785e42d) only in the stored run`, and the material policies are compared as
recorded.

**The launch.** A launch built from a pulse records both the pulse and the realised
field; the field is ignored, since the stored state supersedes it, and the pulse is
compared. A changed pulse energy is therefore refused: a script that changed its pulse
and kept its output path is more likely a mistake than a wish to continue the old run.
An array launch has no other record, and its array is compared. Input noise drawn from
a recorded seed is compared by its model and seed; a realisation passed in, and modified
noise, which acts at every step, are compared by their spectra.

### The verdict

What a resume does depends on the differences and on whether the file has work left: a
save plane ahead of it, or a terminal checkpoint still owed its plane. That is decided by
position, not status, so a file killed after its last plane has none.

| The differences | `resume=True`, work left | `resume=True`, finished | `resume="force"` |
|---|---|---|---|
| none, or only rounding and neutral values | continues | returned, if every session ran under these numerics | as `resume=True`, and accepts an interpolated last plane |
| numerics | continues, warned and recorded | refused: nothing would be computed | as `resume=True` |
| compared | refused | refused | continues if work is left, marked, with the Adams history restarted |
| layout, or the statistics do not fit | refused | refused | refused |
| a session already forced the file | refused, naming the position | returned, with a warning | continues if work is left |

{py:func}`~solprop.output.hdf5.check_cache` is the comparison's gate; the decisions that
need the file's state are the driver's:

```{code-block} python
from solprop.output import check_cache, resume

if check_cache("run.h5", metadata):        # False when there is no file
    state = resume("run.h5")               # metadata, last plane, step history
    start_from(state.plane)                # already back in the FFT convention
```

A refusal raises {py:class}`~solprop.exceptions.OutputError` naming what differs, by
class and by part of the record, an array as one entry with its count and largest
difference, and says what to do instead:

```text
OutputError: run.h5 cannot be resumed; these compared values differ:
  configuration: medium.shared_pressure.values (257 elements, largest 0.2), ...
To change the physics, launch a new segment from a saved plane; to continue in this
file regardless, pass resume="force".
```

### Changed numerics

An unfinished run continues under a changed tolerance or integrator:

```{code-block} python
sim = Simulation.from_capillary(..., rtol=1e-6)
sim.run("run.h5")                       # stopped part-way

sim = Simulation.from_capillary(..., rtol=1e-5)
sim.run("run.h5", resume=True)
# SolpropWarning: resuming run.h5 at z = 0.0412 m with changed numerics;
#   the numerics changed: stepper: controller.rtol 1e-06 -> 1e-05
```

The continued run is two adaptive integrations from one state, each held to its own
tolerance, so it is no better than the looser of the two. The session records how it
continued in a `resume` entry of its execution record, written before any step:

```text
{"from_z": 0.0412, "from_plane": 82, "forced": false,
 "numerics": {"stepper.controller.rtol": [1e-06, 1e-05]},
 "compared": "", "neutral": []}
```

{py:attr}`~solprop.output.schema.RunOutcome.resumes` reads every session's entries back.
A finished file is returned only when every session ran under the requested numerics:
a tightened `rtol` never hands back the loose result, and a file finished under another
tolerance than it started with matches neither. Write a new file instead. A run moving
from the Adams integrator to Runge--Kutta keeps its charged call ledger in the earlier
execution record, and an Adams history continues across a changed call budget or safety
factor, since only the formulation, the carrier frame and the highest order define it
({py:attr}`~solprop.stepper.multistep.Multistep.history_key`).

### Forcing a resume

`resume="force"` continues in the same file across compared differences, whenever the
layout agrees, the statistics fit and there is work left. It is for a user who knows
why the records differ: a corrected optmat model on a long run, a release that moved a
derived table, a change of physics they mean.

```{code-block} python
sim = Simulation.from_capillary(..., pressure=1.2)
sim.run("run.h5", resume="force")
# SolpropWarning: forcing a resume of run.h5 at z = 0.0412 m; these compared values
#   differ: configuration: medium.shared_pressure.values (257 elements, largest 0.2), ...
#   The Adams history starts afresh. From here the file's record does not describe its
#   planes: stats/* hold, and anything rebuilt from the record does not.
```

- The state comes from the stored plane, and the Adams history starts afresh: the
  snapshot was stepped under the other physics.
- Before any step, the backend's statistics are checked against the file's `stats/*`,
  so force cannot change the number of species or turn plasma on or off, and a refused
  force writes nothing.
- The session's `resume` entry says `"forced": true` and holds the compared differences
  as text. It is written before any step, so a stopped or failed force leaves its mark.
- The file's record is never rewritten: it describes the run as first written. From the
  forced position on it does not describe the planes. `stats/*` hold, because the
  requested problem wrote them; anything rebuilt from the record does not.
- The mark is respected. `resume=True` refuses a forced file with work left, naming the
  position, and returns a finished one with a warning; only `resume="force"` continues
  it. {py:func}`~solprop.processing.planes.field_plane` refuses a plane beyond the position,
  `source=` included. The peak-field report reads only the forced session's planes.
  {py:func}`~solprop.output.hdf5.open_file` warns once.
- A mistyped value of `resume` is a `ValueError`; scans have no `resume` and never
  force.

**Force or a new segment.** {py:meth}`~solprop.driver.simulation.Simulation.with_input` from the
last plane records a changed physics properly, in its own file, and that file can be
coupled onward and processed from its record. Force keeps the same file, cumulative
call ledgers, modified noise and fixed schedules, at the cost of a record that stops
describing the later planes. Anything that will be coupled onward or processed from the
record takes {py:meth}`~solprop.driver.simulation.Simulation.with_input`
([starting from a saved interior field](interior-launch.md)).

### The resume key

The record a resume compares is the hashed part of the run record,
{py:meth}`~solprop.output.schema.RunMetadata.hashed`, whose digest
{py:meth}`~solprop.output.schema.RunMetadata.configuration_hash` is stamped in every
file; the stamp tells whether the file is intact, and the exact checks of a suspended
writer, a snapshot and a scan row compare it.

The record holds the schema version, the grid, the modes, the configuration, the noise,
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

The output schema is in the record too, but not in the comparison: a file of another
output schema is not read at all ([the schema version](output.md#the-schema-version)).

### Noise drawn without a seed

A call without `seed=` draws fresh operating-system entropy each time it is built
([reproducibility](../media/noise.md#reproducibility)). So that re-running the same call
with `resume=True` continues its own file, a realisation the constructor drew without a
seed is drawn again from the seed the file recorded before the records are compared,
and the run logs that it did. This covers the default noise of `prop_gnlse`,
`prop_capillary`, `prop_waveguide` and `prop_radial` and an explicit `noise="input"` or
`"modified"` given without a seed. A seed the call gives, or a realisation it passes in,
is never replaced: a resume that differs in it is refused, and the refusal names the
seed the file was drawn from (`pass seed=17 to draw it again`).

To carry on in the same file by hand, build the handler with `append=True`: it opens the
file read-write instead of replacing it, compares the record offered with the file's and
refuses a layout or compared difference, keeps the file's own record and creation time as
its own, and numbers the next plane and the next attempt where the file left off.

## Resuming from the file alone

A run built from a [recipe](../setup/recipes.md) records it in its output, and the file
then resumes without the script that made it:

```text
solprop resume run.h5 [--interpolated | --force] [--memory-budget "16 GiB"]
```

or `solprop.recipe.resume_file(path, mode=True)` from Python. The simulation is rebuilt
from the file's resolved recipe and continued with `resume=mode`, so the comparison above
decides, and records any difference, as for any resume. A file written from a script
records no recipe, and its script resumes it.

**Resuming a file runs the setup it holds.** A file from someone else is a program to
run: its recipe passes the same checks as a recipe file (registered types only, no keys
that choose paths), its embedded data files are read into memory and checked against
their hashes, and it does not choose its own limits: its `memory_budget` and
`node_budget` are ignored, and yours apply, refusing by default rather than warning
(four fifths of the machine's memory unless you give `--memory-budget`). Opening an
HDF5 file at all exposes the HDF5 library to it.

### What a file allows

{py:func}`solprop.output.resumable.resumability` answers three questions about a file without
resuming it:

```{code-block} python
from solprop.output import resumability

answer = resumability("run.h5")
# Can it continue, from where, and with which resume=?
print(answer.resumable, answer.z, answer.mode)
# Why not: complete, no state, another schema, a live writer.
print(answer.reason)
# "stored", "none" (a file from a script) or "newer".
print(answer.recipe)
# What a resume would compare, at the cost of a rebuild.
print(resumability("run.h5", gate=True).comparison)
```

`mode` is `"force"` for a file a forced session wrote and `"interpolated"` when its last
plane lies inside a step. `gate=True` rebuilds the run from its recipe and compares the
records, so a moved optmat digest shows before a resume is tried.

### Stopping a run

`Simulation.run(..., stop=flag)` takes any object with an `is_set()` method, such as a
{py:class}`threading.Event`. The run polls it before its first step and before every
trial step, so it answers within one step. When it is set, the run writes its state as
SIGTERM would, tells the reporter through `interrupted`, and **returns** the handler: a
requested stop is not an error, and the outcome's message says the caller's flag stopped
it. `resume=True` continues the file. A front end sets the flag from any thread, on any
platform, with no signals; each run polls its own. `solprop run` and `solprop resume` set
it on Ctrl-C, SIGTERM and, on Windows, Ctrl-Break, and exit with the usual status; a
second Ctrl-C interrupts at once.

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

The design is [SEP 42](../../dev/seps/archive/sep-0042-tolerant-resume.md).
