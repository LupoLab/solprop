# SEP 42 — Resuming by what matters to the run

| | |
|---|---|
| **SEP** | 42 |
| **Title** | Resuming by what matters to the run |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | Opus 5.5 (AI) |
| **Type** | Change |
| **Created** | 2026-10-31 |
| **Resolution** | — |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Not started |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Today a file resumes only when the rebuilt run's configuration hash equals the stored one,
exactly. That is stricter than continuing the run needs. A release that adds a field, a
newer optmat with the same numbers, a looser tolerance, or a platform whose mathematics
library rounds differently in the last bit each refuses a resume of the same physics. This
SEP replaces hash equality as the resume gate with a comparison of the two records path by
path. A short table sorts every recorded path into one of four classes:

- **layout:** the stored state and the file's arrays depend on it; it must agree, with no
  override;
- **compared (the default):** physics and launch parameters; it must agree, unless the user
  forces the resume;
- **numerics:** how accurately the run steps; it may change while there is something left to
  compute, and the change is recorded;
- **ignored:** provenance, and launch realisations whose parameters are compared.

Numbers agree per element within a rounding-level tolerance, and a path one record lacks is
a difference unless a table declares its neutral value. optmat models are compared by the
digest of their numbers. `resume="force"` continues in the same file whenever the layout
agrees, and says so in the file. The output schema is bumped only for a change an older
reader would misread.

Success means:

- a run resumes after an upgrade, an optmat release with the same numbers or a move to
  another platform, unless the physics changed;
- an unfinished run resumes with a changed `rtol` or integrator, and the file says which
  session stepped how;
- every refusal names what changed, and a change of physics is never continued unasked;
- the compatibility code is one table of classes and one table of neutral values and
  renames, and the rest is small edits to existing code.

### 2. Motivation and scope

A resume should succeed whenever continuing the stored state with the requested problem
gives the run the user asked for, and be refused, with a reason, when it would not. The
present gate departs from that in four ways, each of which a user meets in ordinary work.

- **A field added in a release breaks every earlier file.** A run record holds every field
  of every dataclass it serialises, at its default too (ADR-051). A release that adds a
  field to `PIController`, a pulse, a response or a mode moves every configuration hash and
  bumps the output schema, though the new field's default may reproduce the old behaviour
  exactly. The roadmap's *A resume-compatibility policy before 1.0* asked what to do about
  it.
- **The material manifest is compared by version.** The resume key covers the optmat
  version, and each model's name, status, status reason and sources. An optmat release that
  changes no number a run uses still refuses its resume.
- **Numbers are compared bit for bit.** The record holds numbers computed with the
  platform's mathematics library and FFTs: the launch field, the noise spectrum, the mixture
  tables, the absorber rates, an ionisation threshold found by bisection. SEP 25 gave up
  bitwise agreement across platforms, and `tests/record_numbers.py` already compares these
  numbers to a bound, but the resume key did not follow. So nothing guarantees that a run
  started on one machine resumes on another.
- **Accuracy settings count as physics.** A run stopped part-way under `rtol=1e-6` cannot be
  finished under `rtol=1e-5`, nor with another integrator, though either continuation is
  well defined. The boundary check's budget left the key for the same reason (SEP 21).

And there is no way to insist. A long run stranded by a corrected optmat model, or by a
release that moved a derived table, can only continue in a new file through `with_input`,
which restarts the Adams history and the call ledgers and refuses modified noise and fixed
schedules.

In scope:

- the classification of recorded paths and the comparison rules;
- reading a stored record that an earlier or later release wrote;
- what a session holds after continuing, and the code a continuation meets behind the gate;
- `resume="force"`;
- scans' reuse of finished points;
- the rule for when the output schema is bumped.

Out of scope:

- **Continuing a finished run to a longer length.** The longer run differs from the stored
  one in hundreds of length-dependent paths, its absorber depends on the length, and the
  file's record cannot describe planes beyond its own. It needs its own proposal, after the
  recipes of SEP 41 (in draft). `with_input` from the last plane already continues a
  finished run into a new file. The roadmap keeps the item.
- **Reading files of another output schema.** ADR-058 stands: a file is read only at its own
  schema. This SEP makes bumps rare; it adds no readers for old layouts.
- **Bitwise reproduction of a resumed run.** A resume agrees with an uninterrupted run to
  the bounds SEP 25 set, as today.
- **The size of the run record.** The roadmap's *Large arrays in the run record* is
  unchanged.

### 3. Context

The resume key is `RunMetadata.configuration_hash()` (`solprop/output/schema.py`), a SHA-256
of the canonical JSON of `RunMetadata.hashed()`. That holds the schema version and
simulation kind; the grid, modes, spatial and modal records, quadrature and power
convention; the configuration (medium, guide, pulse, launch field, responses, profiles,
absorber, save planes, statistics, precision and derived tables); the noise; the stepper
record; the material manifest's identities and policies; the launch position and the TDSE
descriptor. Kept out are the wall-clock time, package versions, git provenance, backend
record and boundary check ([resuming](../../guide/results/resuming.md#the-resume-key)).

`check_cache` (`solprop/output/hdf5.py`) is the gate. These places test the hash too, and §5
says what happens to each:

| Where | What it asks | Under this SEP |
|---|---|---|
| `check_cache` | may this file be continued? | the comparison |
| `HDF5Output` opening in append mode | does the file hold this run? | the comparison, then adopts the file's record |
| `_stored_metadata`, stamp against record | is the file intact? | exact, against the stored text |
| `_validate_metadata` (`output/scan.py`) | is a scan row intact? | exact, against the stored text |
| `resolve_metadata_write` | is a second write the same run? | exact |
| `HDF5Output.reopen`, snapshots, the TDSE archive context | has the session's file changed? | exact, against the adopted record |
| the scan drivers' finished-point check | is this point done? | the comparison, numerics held to agreement |
| `Queue` and `Slurm` at collection, and the queue manifest | is this the declared scan? | exact, unchanged |
| processing's `source=` check | did this simulation write this output? | the comparison |

Every route to a stored record goes through `RunMetadata.from_json`, which rebuilds
`GridRecord`, `ModeRecord`, `StepperRecord`, `NoiseRecord` and others through `_rebuild`. A
field the class no longer has is an unexpected keyword, a required field that is absent is
missing, and a defaulted field that is absent is filled in, after which the digest no longer
matches the file's stamp. Any of these refuses before a hash is compared.

Two earlier rulings relaxed the gate for particular cases: a realisation drawn without a
seed is drawn again from the file's seed (`Simulation._adopt_recorded_seed`), and the
boundary check's budget is outside the key. A session's backend and provider facts already
go into `RunOutcome.execution`, and earlier sessions' into `previous_executions`.

`tests/output/test_record_stability.py` and `tests/record_numbers.py` already pin run
records: the structure by an exact hash, and the numbers to `RECORD_RTOL = 1e-11`, a bound
chosen from the largest departure a change of libm, BLAS or compiler can make.

The relevant decisions:

- **ADR-051:** no compatibility with earlier configurations; every field is recorded at its
  default; a file of another schema is never resumed.
- **ADR-058:** the 0.x releases may break the resume key if the changelog says so; a file of
  another schema is refused everywhere; a resume-compatibility policy is to be decided
  before 1.0.
- **SEP 25:** no bitwise reproducibility anywhere; bounds instead.
- **SEP 14:** an interpolated plane resumes only with `resume="interpolated"`.

This SEP answers the roadmap item *A resume-compatibility policy before 1.0*. It chooses
neither option the item named; §10 says why.

### 4. Theory

**Records as trees.** A record is the parsed JSON of `hashed()`. A path is a tuple of
components: mapping keys and list indices, never a dotted string, because a key may contain
a dot. A leaf is any value that is not a plain mapping or list, and these tagged mappings
are leaves too:

| Tag | Meaning | Agreement |
|---|---|---|
| `__ndarray__`, with `__imaginary__` when complex | an array | numeric, below |
| `__complex__` | a complex scalar | numeric, below |
| `__enum__` | an enumeration member | exact |
| `ArrayDigest` | the SHA-256 of a large array | exact |
| `__callable__`, `__repr__`, marked `reproducible: false` | a repr, unstable across processes | exact, with the reason in the refusal |

A list of numbers is compared number by number, like an array. Differences in a list or an
array are reported as one entry with a count.

**Normalisation.** Two steps run before the walk, because neither is table data: model
records and solids are re-keyed from `material:kind:name` to `(material, kind, digest)`, so
that a renamed model is the same model; and a subtree whose `__class__` begins with
`optmat.` is dropped, because the manifest records its digest.

**Classes.** A table $T$ maps path patterns to classes. A pattern is a tuple of components,
with `*` matching one component (a list index included), `**` any number, and braces listing
alternatives. The first row that matches a leaf's path decides, and an unmatched path is
compared, so a field nobody has classified is held to agreement: the safe error is a
refusal.

$$
c(p) \in \{\text{layout},\ \text{compared},\ \text{numerics},\ \text{ignored}\}.
$$

**Agreement of leaves.** Strings, booleans, integers, `None`, enumerations and digests agree
when equal. An integer and a float, such as `pressure=1` and `1.0`, compare as numbers.
Floating-point numbers and arrays of the same shape agree when, for every element,

$$
|a_i - b_i| \le \varepsilon \max(|a_i|, |b_i|), \qquad \varepsilon = 10^{-11},
$$

with complex values measured by modulus, NaN agreeing only with NaN in the same place and an
infinity only with the same infinity. The test is per element because the elements of an
array may be different quantities: the dispersion coefficients `[β2, β3, β4]` span 18 orders
of magnitude, and a scale taken from the array's largest element would let β3 double and β4
change sign. A few arrays have tails of rounding, which a per-element test would judge
against themselves. Only the paths $T$ marks **tail** add a floor, taken from the array:
$|a_i - b_i| \le \varepsilon \max_j(|a_j|, |b_j|)$.

$\varepsilon$ is `RECORD_RTOL`, one constant shared with the tests. It rests on a
measurement: the largest departure among the record's numbers is the ionisation threshold,
found by bisection to a relative bracket of $10^{-12}$. It is far below any stepper
tolerance in use and any change a user makes on purpose. Continuing with numbers that agree
to $\varepsilon$ perturbs the solution at about the order of the rounding the march makes
itself, which is the kind of difference SEP 25 bounds rather than forbids.

**Exact paths.** A few numbers cannot take a tolerance, because the driver does exact
arithmetic on them. A resume finds its plane by `searchsorted` on the save planes and lands
on them, so a save plane moved by $10^{-13}$ fails later with an unrelated error. $T$ marks
them **exact**: `configuration.save_planes`, `initial_z` and `grid.length`. Plain arithmetic
on a user's number makes them bit-identical across platforms, so nothing is lost.

**Absent paths.** A path present in one record and absent from the other is a difference,
unless a table $N$ gives its neutral value: the value that reproduces the behaviour of the
code before the path existed. With an entry, the absent side reads as that value and the
comparison goes on. The table holds one line per field that a release declared neutral.
Every new recorded field therefore needs a decision, and the frozen records of §7 fail until
someone makes it. The field's default is deliberately not consulted: a new field that
changes what runs compute would agree with itself through its own default, and resolving a
stored `__class__` would import a module a file names.

A default changed deliberately needs no entry. The earlier record holds the old value, which
the comparison sees as a change of physics, and so it should. $N$ also has a short section
of class renames, for a class moved between modules.

**Materials.** After normalisation a model is identified by `(material, kind, digest)`.
optmat's digest is the SHA-256 of the model's canonical record, which identifies its
numbers, and it leaves out status, status reason and the override fields, so `modified` and
`base` stay compared. Two optmat versions that give a model the same digest give the run the
same numbers.

**Verdict.** For stored record $S$ and requested record $R$, with equal digests a fast path
meaning no differences:

| Differences | `resume=True`, unfinished file | `resume=True`, complete file | `resume="force"` |
|---|---|---|---|
| none, or only rounding and neutral | continues | nothing to do, as today | as `resume=True` |
| numerics | continues, recorded | refused: nothing would be computed | as `resume=True` |
| compared | refused | refused | continues, recorded, flagged; refused on a complete file |
| layout | refused | refused | refused |

A complete file with changed numerics is refused rather than returned, because the user
asked for something the file does not hold: a tightened `rtol` must not hand back the loose
result. The remedy is a new file.

**Numerical changes on continuing.** With the controller held to $\text{rtol}_1$ on $[0,
z_r]$ and $\text{rtol}_2$ on $[z_r, L]$, the continued run is a concatenation of two
adaptive integrations from one state, and its local error is bounded on each segment by that
segment's tolerance. So the whole run is no better than the looser of the two. That is a
statement about accuracy, not identity, and it is recorded, not refused.

### 5. Proposed interface

#### 5.1 The comparison

A new module `solprop.output.compatibility` holds $T$, $N$ and the comparison. `RunMetadata`
gains one method:

```text
comparison = stored.compare(requested)     # RecordComparison
comparison.continues                       # bool, for resume=True
comparison.layout                          # tuple of PathChange
comparison.refused                         # tuple of PathChange (compared paths)
comparison.numerics                        # tuple of PathChange (recorded)
comparison.neutral                         # tuple of paths read through N
comparison.summary()                       # the refusal or report text
```

`PathChange` holds the path and both values, with arrays and lists summarised by shape and
largest difference. `summary()` groups elements as `summarise_differences` does, because a
pressure change moves about 500 paths: it names every part of the record that changed, with
its first entries and a count of the rest. `differences` and `summarise_differences` stay,
for displays that want every difference.

`check_cache(path, metadata)` keeps its signature and returns `True` or raises `OutputError`
with `comparison.summary()`. It wraps an internal function that returns the comparison,
which the driver uses for the record of §5.3. The refusal names the models whose digests
changed, and for a callable says that its repr changes between processes and to pass an
object that carries its data.

#### 5.2 The class table

In order; the first row that matches decides, and an unmatched path is compared.

| Pattern | Class | How, and why |
|---|---|---|
| `simulation_kind`, `power_convention` | layout | exact |
| `initial_z`, `grid.length`, `configuration.save_planes` | layout | exact, §4 |
| `grid.**` | layout | the stored arrays' axes |
| `modes`' length, `modes[*].kind` | layout | the state's basis; a mode's data is compared |
| `spatial.**`, `quadrature.**`, `modal.**` | layout | the state's basis and transforms |
| `configuration.precision`, `configuration.statistics.**` | layout | the error floor and the per-plane datasets |
| `tdse.**` | layout | the archive and call ledgers bind to it; compared as text, so numbers inside `configuration_json` are exact |
| `stepper.**`, `configuration.propagator_mode` | numerics | how accurately, not what |
| `configuration.initial`, where `configuration.pulse` is present | ignored | realised from a compared pulse on a compared grid; the stored field supersedes it |
| `configuration.initial` otherwise | compared, tail | an array launch: this is its only record |
| `noise.spectrum`, where `noise.seed` is set and `noise.model` is `input` | ignored | drawn from a compared model and seed, used only at the launch |
| `noise.spectrum` otherwise | compared, tail | a realisation passed in has no other record, and modified noise is injected at every step |
| `configuration.absorber.{spectral_rate, temporal_rate, temporal_rate_fine}` | compared, tail | rates with rounding tails |
| `materials.optmat_version` | ignored | provenance |
| `materials.records.*.{name, status, status_reason, sources, optmat_version}` | ignored | provenance; the digest identifies the numbers |
| `materials.policies.ionisation[*].{model, status}`, `**.cladding.source` | ignored | a model's name and status, outside the manifest's records |
| `**.citation` | ignored | a citation string, which may carry a file's hash and the optmat release |

`with_input` drops `configuration.pulse` for an array launch, so that condition is what
separates a launch built from parameters from one that is only an array. The unseeded-noise
ruling is kept, and a passed-in realisation is still never replaced. A launch that differs
in its parameters is refused: the stored field does supersede it, but a script that changed
its pulse and kept its output path is more likely a mistake than a wish to continue the old
run.

The table is data, reviewed like code. A test asserts that every row matches a path in at
least one route's record, so a stale row is noticed.

#### 5.3 What a session holds after continuing

After a continuation **the session's identity is the file's record**. Append mode adopts it;
the driver uses it for the writer, for the TDSE archive context and for `output.metadata`.
So the exact checks of §3 compare a file with its own record and hold, and `suspend`,
`reopen` and snapshots work after a tolerant resume. The requested problem is what steps.
The file's stored record is never rewritten: it describes the run as first written, and the
executions say how each session continued it.

A continuation with differences gains a `resume` entry in that session's
`RunOutcome.execution`:

```text
{"from_z": 0.0412, "from_plane": 82, "forced": false,
 "numerics": {"stepper.controller.rtol": [1e-06, 1e-05]},
 "compared": {},
 "neutral": ["stepper.controller.new_field"]}
```

`previous_executions` keeps it with the session. A non-finite value, such as the default
`dz_max`, is written as text, because `RunOutcome` refuses non-finite numbers.

#### 5.4 Reading a stored record

The stored record reaches the comparison as it was written.

- `from_json` keeps the raw hashed part. A record read from a file digests to its own stamp
  and compares by its own text, whatever today's code would write, so the stamp check and
  `_validate_metadata` still mean "the file is intact".
- Rebuilding a dataclass for reading drops fields the class no longer has, and fills an
  absent field from $N$, or from the field's default where it has one. The rebuilt object is
  for reading only and the comparison never sees it. An absent required field with no $N$
  entry refuses, by path.

A file from a newer release of the same schema can therefore be read. It resumes only if the
comparison allows it.

#### 5.5 Forcing a resume

`resume="force"` on `Simulation.run` and the `prop_*` routes continues in the same file
whenever the layout agrees and there is something left to compute, whatever the compared
paths say. It is for a user who knows why the records differ: a corrected optmat model on a
long run, a release that moved a derived table, a change of physics they mean.

- **What it refuses:** any layout difference, because the stored state cannot be loaded and
  the file's arrays would not fit; and a complete file, because nothing would be computed.
- **What it also accepts:** an interpolated last plane, recorded as `resume="interpolated"`
  records it.
- **What it records:** every difference, in the §5.3 entry with `"forced": true`, and the
  position in a new annotation `forced_resumes` (`FORCED_RESUMES_ANNOTATION`).
- **What it warns about:** the file's stored record no longer describes the planes after
  that position, and processing reads that record. `open_file` warns once for a file with
  the annotation, naming the position, and `source=` is refused by the comparison unless the
  simulation matches.

A scan never forces. `with_input` stays the way to record a changed physics properly in its
own file, and the manual says which to use when.

#### 5.6 The numerics behind the gate

The continuation meets these, which phase 1 changes:

- **Call ledgers.** A run that goes from `Multistep` to `RungeKutta` is refused today by
  `_validate_resume` and by `restore_call_ledgers`, because a logical call ledger requires
  Adams integration. The charged ledger stays on as provenance in `previous_executions` and
  is not installed in a Runge–Kutta march. The reverse direction already works. A TDSE run
  admits only `Multistep`, so it never meets this.
- **Adams history.** `AdamsMarch._refusal` compares the snapshot's `repr` of the method, so
  raising `max_calls` or changing `safety` restarts at order one although the history is
  still valid. The check compares only the fields that define the history. A change of those
  restarts the march, with the warning it gives today.
- **Interpolated planes.** `_open_output` refuses an interpolated last plane only when the
  requested run is dense. With `interpolate_planes` now free to change, the test drops
  `dense and`, because `_interpolated` reads the file. Otherwise a run stored densely and
  resumed sparsely would continue silently from a state the march never passed through.

#### 5.7 Scans

A finished point is reused when its stored record and the point's new record differ only in
rounding, neutral values or ignored paths. Numerics are held to agreement here: tightening
`rtol` is the commonest reason to rerun a scan, and a done point has nothing left to
continue. A scan with changed numerics writes to a new file.

The worker keeps its digest check. A mismatch already returns a `conflict` result carrying
the new record, and the parent, which holds the scan file, compares it with the stored one
and treats a pass as `skipped`. So `_Job` is unchanged, and the common case costs nothing.

`Queue` and `Slurm` keep exact identity: their manifest hashes the builder's code and
refuses a change, and their per-point checks stay exact. The manual says so.

#### 5.8 The output schema

`schema_version` stays in the record and in the configuration hash, but leaves the
comparison. A resume needs the stored file's schema to equal the current one, as
`_refuse_schema` already requires, checked before the comparison.

The schema is bumped only for a change an older reader would misread or that cannot be
appended to:

- a dataset renamed, re-laid or given new units, or an attribute changing meaning;
- anything added inside `stats`, `step_history` or `step_stats`, or anything grown per
  plane.

It is not bumped for a new field in the JSON record, a root attribute, or a top-level group
written once. Those are the additions readers already tolerate (an unknown top-level group
and root attribute were measured to read and resume), and a new top-level group is what SEP
41's `recipe` group needs.

The rule goes into `docs/guide/results/output.md`, which today says a new field "bumps the
output schema", and a new ADR. The ADR narrows ADR-051's "a file of another output schema is
never resumed, through its resume key" to the schema precondition, and replaces its "no
field is left out to keep an earlier record's hash" with the neutral-value rule. Fields are
still recorded in full.

#### 5.9 Example

```text
sim = Simulation.from_capillary(..., rtol=1e-6)
sim.run("run.h5")                   # stopped at z = 41 mm

# after upgrading solprop and optmat, with no physics change:
sim = Simulation.from_capillary(..., rtol=1e-5)
sim.run("run.h5", resume=True)
# WARNING resuming run.h5 at z = 0.0412 m with changed numerics:
#   stepper.controller.rtol 1e-06 -> 1e-05

sim = Simulation.from_capillary(..., pressure=1.2)
sim.run("run.h5", resume=True)
# OutputError: run.h5 cannot be resumed; these compared values differ:
#   configuration: medium.shared_pressure.values (201 elements, largest 0.2),
#   profiles.pressure.value 1.0 -> 1.2, ... and 518 more
#   To change the physics, launch a new segment from a saved plane; to
#   continue in this file regardless, pass resume="force".
```

### 6. Implementation strategies

**Where the classification lives.** One table of path patterns is recommended over
annotating fields with `dataclasses.field(metadata=...)`. It covers configuration keys,
which are not dataclass fields, and optmat's classes, which solprop does not own, and it
keeps every compatibility decision in one reviewable place.

**Layer.** The work belongs to the output layer, with small edits in the driver and
processing, and one in each of `backends/driver.py` and `backends/multistep.py` (§5.6). The
kernel contract and `KERNEL_SCHEMA_VERSION` do not change.

**Cost.** The walk is linear in the record's size: a prototype took 0.04 s on a 28.6 MB
record, against 0.28 s for today's canonical JSON and SHA-256. The digest fast path keeps
processing's per-plane `source=` check as cheap as it is.

**Retention.** Each $N$ entry names the release that added it, and is deleted with that
release's frozen records (Q3).

**Earlier workarounds.** The conditional records in `metadata_for` (`transverse_measure`,
`plasma_integration`, `quiet_collar`, `evanescent_rate`) stay as they are. Both records lack
them in the ordinary case, so they cost nothing. Writing them unconditionally later would
need $N$ entries, and is not proposed.

**New dependencies.** None.

### 7. Verification and validation

The tests are P (structural) and R (rounding) class (`docs/dev/testing.md` §3), and T for
the resumed-run envelope. Nothing here changes a computed field. Unless it says otherwise, a
test goes through a real file and `check_cache`, not a hand-built record.

- **Frozen records.** `tests/output/test_record_stability.py` already pins each case's
  structure and numbers against today's build. It gains, per case and for a capillary and a
  GNLSE route, the whole record of the latest release when phase 1 lands, built by checking
  out that release's tag and never regenerated.
  - Each must compare as continuing with the current build, on every CI platform. A fixture
    captured once and compared on all platforms is the real cross-platform check.
  - A new recorded field fails until it has an $N$ entry, which is what enforces the rule.
  - An optmat data revision moves a digest and is a correct refusal, not a solprop
    regression, so this test tolerates differences under `materials` alone.
    `test_record_stability`'s docstring changes to match.
- **Another release's file.** Files with the stamp an earlier build would write, each of
  them: lacking a defaulted `StepLimits` field, lacking a required `GridRecord` field,
  lacking a top-level entry, and holding an unknown field. With an $N$ entry the first three
  continue, and without one they refuse by path.
- **Tolerance.** The mixed-scale case: `betas` with β3 doubled and β4's sign flipped
  refuses. A 4-ulp change of `lambda0`, `trange`, `energy` or `fwhm` continues on the
  capillary with one and two modes and on the radial route. A named tail agrees where its
  per-element test would not. NaN, infinity, complex and integer-against-float cases.
- **Exact paths.** A length moved by $10^{-13}$ refuses by name, rather than failing later.
- **Materials.** On a real route's record: a changed optmat version, model name or status
  with the same digests continues, and a changed model number moves a manifest digest and
  refuses, naming the model. So dropping the `optmat.` subtrees hides nothing.
- **Launch.** A changed pulse energy refuses. `with_input(spectrum*1.5)` and a different
  passed-in realisation refuse. An unseeded rerun continues. A different modified-noise
  realisation refuses.
- **Numerics.** A capillary run stopped part-way resumes with a changed `rtol`, and with
  `RungeKutta()` after a ledger was written. The `resume` entry records both, a non-finite
  value included, and the resumed run agrees with an uninterrupted run at the looser
  tolerance, to the T-class envelope `testing.md` uses for resumed runs. Raising `max_calls`
  keeps the Adams history.
- **Complete files.** A complete file with a tightened `rtol` refuses, and one with only
  neutral differences has nothing to do.
- **Session identity.** `suspend` and `reopen` after a tolerant resume work, and
  `output.metadata` is the file's. A snapshot on a TDSE run after a tolerant resume, where
  the provider is available.
- **Interpolated planes.** A file stored with `interpolate_planes=True` that ends on an
  interpolated plane refuses under `False`. An ordinary non-dense run never trips the test.
- **Force.** A changed pressure under `resume="force"` continues, records every difference,
  writes the annotation, and warns on `open_file`. A layout change and a complete file
  refuse under force.
- **Scans.** A rerun whose points differ only in rounding or a neutral value skips them, and
  one with changed numerics or physics is a conflict.
- **Schema rule.** A file with an unknown top-level group and root attribute is read and
  resumed. Adding a dataset in `stats` is covered by the rule, not by tolerance.
- **The table.** Every row matches a path in at least one route's record.

Not exercised by CI: a real pair of files from two platforms beyond the frozen records, and
the queue and Slurm executors, whose identity does not change.

### 8. Documentation

- **`docs/guide/results/resuming.md`:** "The resume key" becomes "What a resume compares":
  the classes and table, the tolerance, neutral values, materials by digest, the numerics
  record, forcing, and which of `force` and `with_input` to use when, with an example
  refusal and an example continuation. "Noise drawn without a seed" stays.
- **`docs/guide/results/run-access.md`:** the paragraph on `check_cache`.
- **`docs/guide/results/output.md`:** when the schema is bumped; "every field is recorded"
  keeps its first half.
- **`docs/dev/contributing.md`:** a new recorded field needs a decision, and how to add an
  $N$ entry and refresh frozen records at a release.
- **Docstrings:** `RunMetadata.compare`, `RecordComparison`, `PathChange`, `check_cache`,
  `Simulation.run`'s `resume`, and the resume-key paragraphs of `output/schema.py` and
  `output/hdf5.py`.
- **ADR:** a new ADR for the policy, narrowing ADR-051 and answering ADR-058's open policy.
- **Roadmap:** *A resume-compatibility policy before 1.0* goes; *Continuing a finished run
  to a longer length* comes in.
- **SEP 41:** its §3, §5.2 and §9 describe the resume gate and the schema as they stand
  today, and follow this SEP when it is accepted.

### 9. Compatibility and impact

- **Behaviour:** resumes that were refused now continue where the run is unchanged, or with
  a warning where numerics changed. Nothing that continued before is refused, because equal
  hashes mean no differences. The `dense and` edit affects only a non-dense file ending on
  an interpolated plane, which an ordinary run cannot produce; §7 tests it.
- **Breaking changes:** none to the API; `resume` gains a value, and `check_cache`'s refusal
  text changes.
- **Output schema:** unchanged by this SEP. The `resume` entry goes into the free-form
  `RunOutcome.execution`, and `forced_resumes` is an annotation.
- **Files already written:** the `DataPulse` phase fix takes the output schema to 2 before
  this SEP lands, so 0.1.0 files stay refused by their schema. The rules here apply to files
  of the schema current when phase 1 lands, and later ones, and to earlier files of that
  schema through their full records and digests.
- **`KERNEL_SCHEMA_VERSION`:** unchanged.
- **The configuration hash:** unchanged. It is still stamped in every file, and it is the
  exact identity §3 lists.
- **Validated results:** unaffected, since no computed field changes.
- **Luna.jl:** no difference to record.

### 10. Alternatives considered

- **Hash with the stored file's own rules.** One hashing function per released record
  layout. This is a growing set of versioned hashers, the compatibility code the maintainer
  wants to avoid, and it still compares bit for bit and still refuses numerics changes.
- **Reintroduce omit-at-default.** Hashes stay stable when fields are added, but not when
  they are removed or a default changes, and the record stops stating every value used,
  which ADR-051 chose for good reason.
- **Read an absent field as its current default.** Less to write per field, but a new field
  that changes the physics agrees with itself and continues silently, and a test built on
  the same rule cannot catch it. An explicit table is less code and keeps the refusal as the
  safe error.
- **Scale the tolerance by the array's largest element.** Simpler, but it lets real changes
  in dispersion coefficients through (§4).
- **Ignore the launch altogether.** The stored field supersedes it, but a mistaken resume
  would continue the wrong run silently. Only realisations whose parameters are compared are
  ignored.
- **No override.** `with_input` covers a deliberate change of physics, but it splits the run
  across files and restarts the history and ledgers. For a long run stranded by a change
  that does not matter to the user, the same file is what is wanted, and the cost is one
  class and one annotation.
- **Ask for numerics changes to be opted into.** A value for a choice users make on purpose.
  A warning and a record serve as well.
- **Continue a finished run to a longer length here.** Moved to its own proposal (§2).

### 11. Phases

Both phases go on one branch, `feature/sep-0042-tolerant-resume`, with one pull request
opened at Final (SEP 0).

1. **Comparison and resume.**
   - **Deliverables:** `solprop.output.compatibility` with $T$, $N$, normalisation and
     `RunMetadata.compare`; `from_json` keeping the raw record and tolerating absent and
     unknown fields; `check_cache`, append mode and `source=` on the comparison; the session
     adopting the file's record; the `resume` entry; the ledger, history and
     interpolated-plane edits; `resume="force"` with its annotation and warning; the frozen
     records; the manual and the ADR.
   - **Verifies:** §7 except scans and the schema rule.
2. **Scans and the schema rule.**
   - **Deliverables:** the parent's comparison on a digest conflict in `Local`, `Pool` and
     `Batch`; the bump rule in `output.md` and `contributing.md`.
   - **Verifies:** the scan and schema tests of §7.

### 12. Open questions

- **Q1. The tolerance.** Per element at $10^{-11}$, with an array-scale floor only on named
  tails (recommended); or the array-scale test at $10^{-12}$ the first draft proposed, which
  the review showed passes real physics changes. **The maintainer agreed the first draft's
  recommendation on 2026-11-01, before the review reopened it; to confirm.**
- **Q2. The launch realisations.** Ignore `configuration.initial` and `noise.spectrum` only
  where compared parameters made them (recommended); or always, as the first draft proposed,
  which hides array launches. **As Q1: to confirm.**
- **Q3. How long neutral and rename entries live.** For the release that adds them and the
  next minor release, then deleted with the frozen records they serve (recommended); or
  until 1.0. **Decision (John Travers, 2026-11-01):** as recommended.
- **Q4. Scans and numerics.** A finished point whose numerics differ is a conflict
  (recommended, §5.7); or it is reused and counted, as the first draft proposed, which hands
  back loose results after a tightened `rtol`. **As Q1: to confirm.**
- **Q5. A force override.** Add `resume="force"`, limited by the layout class and recorded
  in the file (recommended, §5.5); or none, with `with_input` the only route. The maintainer
  asked what leaving it out would exclude, noting that forcing is sometimes useful. **To
  confirm.**
- **Q6. Opting into numerics changes.** Allow them on `resume=True` with a warning and the
  execution record (recommended), or require a new value of `resume`. **Decision (John
  Travers, 2026-11-01):** as recommended.
- **Q7. Precision.** Keep `configuration.precision` from changing (recommended), or make it
  numerics. **Decision (John Travers, 2026-11-01):** as recommended; it is layout, so
  `force` does not change it either.

### 13. References

1. solprop, [Resuming a calculation and checking
   compatibility](../../guide/results/resuming.md).
2. solprop, [Output: the schema version](../../guide/results/output.md#the-schema-version).
3. solprop, [Starting from a saved interior field](../../guide/results/interior-launch.md).
4. solprop, [Design decisions](../decisions.md): ADR-051 and ADR-058.
5. solprop, [SEP 25 — Rounding-level
   reproducibility](archive/sep-0025-rounding-level-reproducibility.md).

## Part II — Implementation record

## Part III — History

- 2026-10-31: Drafted.
- 2026-11-01: AI review (Opus 5.5).
- 2026-11-01: Opus 5.5 (drafting agent) answered R1–R24 and revised Part I (response below):
  a layout class, a per-element tolerance with named tails, an explicit table of neutral
  values, stored records read as written, the session adopting the file's record,
  `resume="force"`, and no phase 3. Maintainer decisions on Q3, Q6 and Q7 recorded; Q1, Q2,
  Q4 and Q5 ask again, because the review reopened them or the maintainer asked.

### AI review — 2026-11-01 (Opus 5.5)

The whole of Part I was read against SEP 0's acceptance criteria and the code on this
branch. Part I is not edited. Most of the points below rest on measurements, made on
macOS arm64 with `numba_cpu` on the capillary, GNLSE and radial routes, using very short
runs. They used a prototype of §4's comparison over `hashed()` trees. To exercise what
happens behind the gate, the prototype patched `RunMetadata.configuration_hash` to leave
out `stepper`, or to be constant. A point says when it rests on reading alone. No real
cross-platform pair of records was compared.

**Summary.** The direction is right and serves the maintainer's aim, and the numerics
class held up well: nothing in `stepper.**` turned out to be physics. But the draft
fails both ways in places. It would continue some runs it must refuse (R1, R2, R4), and
it would refuse the runs that are its main purpose, before the comparison is ever
reached (R3). Most of the other points are code a continuation meets behind the gate,
which the draft does not mention. Several fixes make the design smaller (R23).

#### Blockers

**R1 (blocker; §4, §7, Q1). The array-maximum scale lets real changes in physics
through.** The scale of §4 is the array's largest element, on the stated ground that a
spectrum's tail should be judged against the array. That is wrong for an array whose
entries are different quantities.

- **Measured:** `Simulation.from_gnlse` with
  `betas = [0, 0, -1.183e-26, 8.1e-41, -9.5e-44]`, against the same run with β3 doubled
  and β4's sign flipped. The largest relative difference by §4's rule is $6.8\times
  10^{-15}$, so the comparison **continues**, with no compared path refused. The
  coefficients are an array leaf whether the caller passes a list or an array:
  `configuration.gnlse.mode.dispersion.betas`, `modes[0].configuration.dispersion.betas`
  and their `dispersion_derivative` twins.
- **Why it matters:** this is the GNLSE route's ordinary input. A resume after an edit to
  the dispersion would continue silently under different physics, which today's hash
  refuses and which §2 says must be refused.
- **Inconsistent treatment:** the same kind of quantity is judged two ways according to
  how it happens to be recorded. `configuration.medium.shared_pressure.values` and
  `...density.values` are JSON lists, tested scalar by scalar.
  `configuration.mixture_processing.density_m-3` is an array, tested against its maximum.

*Recommendation:* make the default per element, $|a_i-b_i| \le \varepsilon \max(|a_i|,
|b_i|)$. Allow an array-scale floor only on paths the table names as having tails. The
measurement supports this: under a 4-ulp change of `lambda0`, `trange`, `energy` or
`fwhm`, the only arrays over $10^{-12}$ per element were `configuration.initial` (ignored
anyway) and `configuration.absorber.temporal_rate_fine` ($1.35\times10^{-11}$, a tail).
Do not copy the floor in `tests/record_numbers.py` to every array. At order $10^{-16}$ of
the peak it still hides β4 above. §7 needs a mixed-scale test, for which this `betas`
case is ready-made.

**R2 (blocker; §4, §6, §7). Reading an absent field as its current default assumes every
new field is neutral, and the frozen-record test cannot catch one that is not.** §4 says
that the safe error is a refusal, but for absent paths the rule does the opposite: $\nu(p)$
falls back on the field's default.

Suppose a developer adds `some_term: bool = True` to a response, changing what new runs
compute. The stored record lacks the field, $\nu$ reads `True`, the new build records
`True`, and the two agree. The run continues under new physics with only an INFO line.
§6's guard does not help: "A change that adds a field with a non-neutral default fails
the test" is false, because the frozen fixture is compared through the same rule. It
fails only if the developer has already declared a different `neutral`, and that is the
very mistake it was meant to catch. This was established by reading the rule, since no
comparer exists yet.

The generic rule has further problems:

- **Trust:** resolving a `__class__` string from a stored file to read a default means
  importing a module named by file content. SEP 41 treats received files as an exposure.
- **Fields that do not fit:** required fields have no default (`GridRecord.dt_fine`, for
  example). A field that `__post_init__` normalises records something other than its
  default. `init=False` and `record=False` fields need handling too.
- **Another package:** optmat dataclasses are in the record (R11), and the convention
  cannot bind them.

*Recommendation:* an absent path is a difference unless the table $N$ lists it, with its
neutral value. Drop `metadata["neutral"]` and the default lookup. The frozen fixtures
then enforce the convention for real: every new recorded field fails the test until a
developer decides, with one line, that it is neutral. That is less machinery, needs no
import and keeps the refusal as the safe error.

**R3 (blocker; §1, §6, §9). The stored record never reaches the comparison.** §6 calls
the comparer "a recursive walk over two parsed records", but every route to a stored
record goes through `RunMetadata.from_json` and `_stored_metadata`, and the draft mentions
neither. Files were edited to look as an earlier or later release would have written
them, with the stamp the earlier code would have computed:

| Stored record | Result of `check_cache` today |
|---|---|
| lacks `stepper.limits.step_planes` (a defaulted field) | `is inconsistent`: `_rebuild` fills the default, the digest moves and the stamp check fails (`output/hdf5.py:1733`) |
| lacks `noise.spectral_convention` | the same |
| lacks `grid.dt_fine` | `missing or malformed`: required field (`output/schema.py:590`) |
| lacks top-level `quadrature` | `missing or malformed`: `document["quadrature"]` (`output/schema.py:2979`) |
| has an extra `stepper.limits.old_knob` | `missing or malformed`: unexpected keyword |
| lacks `stepper.controller.region`, or `configuration.shock` | reaches the hash comparison, since these are kept as plain mappings |

So the draft's first motivation, "a field added in a release", dies before `compare` for
`StepLimits`, `GridRecord`, `ModeRecord` and `NoiseRecord` and for any new top-level
entry. "Files written by 0.1.0 benefit at once" (§9) is false until this is fixed. The
same applies wherever `from_json` is used: `_adopt_state`, `_write_metadata`, `reopen`,
`_adopt_recorded_seed`, `open_file`, and `_validate_metadata` in `output/scan.py:144`.
§7's neutral-value tests would pass on hand-built records and miss all of this.

*Recommendation:* add to Part I and to phase 1:
- `compare` works on the raw parsed JSON, and `RunMetadata` is not rebuilt for the gate;
- `from_json` becomes tolerant, through $N$, of absent and unknown fields, and keeps the
  raw `hashed` part so that a stored record still digests to its own stamp, rather than
  to what today's code would write;
- §7 gets a test through a real file with a stamp from "earlier" code.

**R4 (blocker; §5.2, §10, Q2). Ignoring the realised launch hides a changed launch whenever
no parameters describe it, and the stored field does not supersede modified noise.** §5.2
says "A resume whose launch differs is refused", and §10 rejects ignoring the launch
because it "lets a mistaken resume continue the wrong run silently". That holds only for a
launch built from pulse parameters.

- **Array launches, measured:** `sim.with_input(spectrum, initial_z=0.05, noise=None)`
  against the same call with `spectrum * 1.5`. No path differs other than the ignored
  `configuration.initial`, so the comparison continues. The same applies to
  `from_radial(initial=<array>)` and to any launch above `INLINE_ARRAY_LIMIT`, by reading:
  the launch array is the only record of the launch.
- **Passed-in noise, by reading:** `noise.spectrum` is ignored and `noise.seed` is `None`
  for a realisation passed in without a seed, so a different realisation continues.
  `docs/guide/results/resuming.md:90` promises the opposite, and §5.2 says that passage
  stays.
- **Modified noise, by reading:** under `model="modified"` the realisation is a source
  injected at every step (`solprop/noise.py:9`), and a resume uses the *requested* run's
  (`backends/driver.py:1208`). So "the stored field supersedes both" (Q2) is false there,
  and a NumPy generator change alters the rest of the run. That may be acceptable as a
  fresh realisation, but it is a noise change and has to be said and recorded.

*Recommendation:* `configuration.initial` and `noise.spectrum` are ignored only where the
parameters that made them are recorded and compared (a pulse in `configuration.pulse`, a
non-null seed). Otherwise they are compared, within the tolerance, and bitwise for an
`ArrayDigest`. For modified noise, record the path as a numerics change, or compare it
under tolerance and say what a generator change does.

#### Significant

**R5 (significant; §4, Q1). $\varepsilon = 10^{-12}$ is tighter than the repository's own
measured bound on the same numbers, and the argument for it is not a measurement.**
`tests/record_numbers.py:23` sets `RECORD_RTOL = 1e-11` for "each number in a run record".
Its stated reason is that the largest departure a change of libm, BLAS or compiler can
make is the threshold field's, "found by bisection to a relative bracket of 1e-12". That
field is in compared paths (`materials.policies.ionisation[].threshold.field`). §4's "short
chains of operations" misses this, and $\varepsilon$ sits on the bracket. So the tolerance
could still refuse a run moved between platforms, which is one of the SEP's four aims.

Numbers inside strings escape the tolerance altogether: strings must be equal, and
`tdse.configuration_json` and its `identity` (`translate/tdse.py:797-801`) carry floats
as text. Whether any of those floats varies with the platform was not checked. Nor was the
material digest fully checked: it changed for a 4-ulp `lambda0`, apparently because the rate
model binds its wavelength, and a literal input is the same on every platform, but whether
any digest takes in a libm-computed number is open.

*Recommendation:* use `1e-11` and share one constant with the tests, stating why. Add a
real cross-platform check to §7. The frozen fixtures, captured once and compared on every
CI platform, would be that check if §7 says so. State the TDSE descriptor's position
explicitly.

**R6 (significant; §4, §5.2). `configuration.save_planes` cannot be tolerant, because the
driver does exact arithmetic on it.** `_validate_resume` runs `np.searchsorted(save_planes,
state.plane.z, side="right")` (`driver/simulation.py:7189`), and the run lands on the
requested planes. A requested length moved by a relative $10^{-13}$ passes §4, then
measured:

| Stored file | Planes | Result |
|---|---|---|
| stopped | $+10^{-13}$ | `DriverError`: checkpoint next_save_index disagrees with its absolute position |
| stopped | $-10^{-13}$ | continues correctly |
| complete | $+10^{-13}$ | `ConvergenceError`: step control asked for $2\times10^{-15}$ m, below `dz_min` |
| complete | $-10^{-13}$ | `DriverError`: resume position lies outside the declared interval |

Planes built by plain arithmetic should be bit-identical across platforms, so little is
lost by exactness.

*Recommendation:* compare `configuration.save_planes`, `initial_z` and `grid.length`
exactly, with a table column or class for it. Alternatively, resolve the requested planes
to the stored ones before the driver sees them, which is more code. `grid.length` moved
in the same probe and also feeds `dz_*` and the absorber.

**R7 (significant; §4, §7). A change from Multistep to Runge–Kutta is refused behind the
gate.** With the gate relaxed, measured: `DriverError: a logical call ledger requires Adams
integration` (`driver/simulation.py:7221`). `restore_call_ledgers` also refuses it
(`backends/driver.py:974`). §7's numerics test and §1's "changed `rtol` or integrator"
would both fail, and §6 says no backend code changes. With the ledger cleared by
hand, the run completes (measured). The reverse direction, Runge–Kutta to Multistep,
already works (measured).

*Recommendation:* Part I must say what happens to the charged-call ledger across a method
change. It should be kept as provenance beside the session's own and not installed in a
Runge–Kutta march. List both call sites in phase 1.

**R8 (significant; §3, §5.2). Making `interpolate_planes` numerics opens a hole in SEP
14's rule.** `_open_output` refuses an interpolated last plane only when the *requested*
run is dense: `if dense and resume != "interpolated" and _interpolated(state)`
(`driver/simulation.py:7781`). Today's hash keeps the two the same. Measured with a
relaxed gate: stored `interpolate_planes=True`, stopped on an interpolated plane, and
resumed with `False` continues silently from a state the march never passed through. The
`True` to `True` control refuses as it should. §3 cites the rule as standing.

*Recommendation:* drop `dense and` from that test, since `_interpolated` already reads the
file; add a test.

**R9 (significant; §2, §5.4, Q4). A finished run or point with changed numerics is not a
continuation.** §2's test is whether continuing "gives the run the user asked for". With
a relaxed gate, measured: a complete file at `rtol=1e-5`, run with `rtol=1e-9` and
`resume=True`, does no work (219 attempts before and after; `driver/simulation.py:2965`)
and returns the old result. The only sign would be a warning saying "resuming ... with
changed numerics". Before, this was a refusal naming the field.

Q4's scan rule does the same for a whole dataset. Tightening `rtol` is the commonest
reason to rerun a scan, and it would now return the loose results and count them in a
closing line. This is partly a judgement.

*Recommendation:* numerics may differ only where there is something left to compute. A
complete file or a done point with numerics changes should refuse, naming the paths and
the remedy (a new file), or at the least say plainly that nothing was computed. Neutral
values and rounding differences may still reuse.

**R10 (significant; §2, §5.4). The session holds the requested record, so the "exact"
integrity checks the draft keeps will fail on any tolerant resume.** §2 keeps snapshot and
suspend checks at exact hash equality, on the ground that they compare a file "with the
session that is writing it". But `PlaneStore.open` sets `self._metadata = metadata` after
appending (`output/schema.py:3852`), so that session holds the requested record, and
`reopen` compares the file's hash with it (`output/hdf5.py:1030`).

- **Measured:** with the append check relaxed, a handler with `rtol=1e-5` over a file at
  `1e-7`, then `suspend()` and `reopen()`, raises `suspended optical checkpoint changed`.
  Snapshots go through that pair (`output/snapshots.py:431`).
- **By reading, on a TDSE run:** the archive context stores the requested hash as
  `optical_configuration_hash` (`driver/simulation.py:7458`), and a snapshot checks it
  against the file's (`output/snapshots.py:816`). So `rtol` changed on a TDSE run makes
  its snapshots fail. Not run: it needs the TDSE provider.
- `run_problem` and the writer also receive the requested record, so `output.metadata`
  would differ from the file's.

*Recommendation:* state once that after a continuation the session's identity is the
file's record: append mode adopts it, and the driver uses it for `_require_metadata`,
for the archive context and for `metadata=`. The numerics stay in the execution record.
This also gives §5.4's "keep the file's record" a defined meaning.

**R11 (significant; §1, §2, §4, §5.2). Materials by digest does not reach the copies of
the models elsewhere in the record.** The table handles `materials.*`, but the record also
holds the optmat objects themselves under compared paths. Measured on a capillary: 72
`optmat.*` dataclass nodes outside `materials`, such as `FittedModel`,
`UniversalDispersionModel` and `Term`.

- **Compared leaves:** `modes[0].configuration.cladding.model.status`, `.status_reason`,
  `.name`, `.covariance`, `.reduced_chi2` and `.record`, plus the solprop wrapper's
  `cladding.source` (`fit/SiO2`) and `Species.name`. 19 `status`, `status_reason`, `source`
  or `record` leaves alone.
- **In the manifest:** `materials.policies.ionisation[].model` and `.status`, and the
  `materials.solids` key, which embeds the model name.
- **Consequence:** an optmat release that revises a status, a status reason or a model
  name still refuses, against §2's second motivation. §7's materials test would pass only
  on a synthetic manifest. A field optmat adds to a class, or a class optmat moves, refuses
  too, or needs $N$ entries for another package.
- **What is right:** the version alone is handled, because `optmat_version` appears only
  in the manifest and in citations, the same set that `without_optmat_release` in
  `tests/test_display.py:615` neutralises (measured on the capillary).
- **Not checked:** §5.2's list ignores `reference_temperature`; whether it is inside the
  digest was not established.

*Recommendation:* add a rule on the class: a subtree whose `__class__` begins with
`optmat.` is ignored, because `Simulation.material_manifest` records its digest. Add a test
that a model change moves a manifest digest. Add rows for the wrapper's `name` and `source`
leaves and the policies' `model` and `status`. Test against a real route's record.

**R12 (significant; §3, §5.4). The scan rule does not fit the drivers, and §5.4 misses the
queue.**

- `_Job` carries only the digest (`driver/_scan_run.py:65`, `:708`), and the check runs in
  the worker. Comparing records means reading `staged/<i>` and sending or re-reading a
  record of up to tens of MB per point. Doing the check in the parent is simpler, after a
  fast path on equal digests.
- `Queue` and `Slurm` never take this path: their jobs have no cached digest
  (`:779`). A changed builder is refused outright at `prepare`
  (`driver/executors/queue.py:416-423`), and its key hashes the builder's code (`:129`).
  So a numerics edit reuses points on `Local`/`Pool`/`Batch` and is refused on `Queue`.
  Per-point hash checks at collection (`:679`, `:696`) stay exact.
- `commit` returns early on `"skipped"` (`output/scan.py:375`), so the row keeps its
  stored hash, which §5.4 wants, but the summary count needs a new status or a return
  value.

*Recommendation:* name the queue's behaviour in §5.4, and decide whether it changes. If it
does not, say that a queue scan keeps exact identity.

**R13 (significant; §5.5, §8). "Readers ignore groups and attributes they do not know" is
true only at the top level.** Measured: an unknown top-level group and root attribute
read and resume correctly, which supports SEP 41's `recipe` group. An unknown dataset in
`stats/` refuses (`plane 3 reports statistics ...`). One in `step_history/` fails with a
raw `TypeError: Only chunked datasets can be resized`. An older build would also append
to a newer file and leave any newer per-plane dataset short.

The rule also contradicts text the draft leaves standing: `docs/guide/results/output.md:337`
("moves the configuration hash and bumps the output schema"), the docstring and purpose of
`tests/output/test_record_stability.py`, and SEP 41's §3 and §9.

*Recommendation:* limit "no bump" to root attributes and top-level groups, written once
and not extended per plane. Anything inside `stats`, `step_history` or `step_stats`, or
grown per plane, bumps. Add the three places above to §8.

**R14 (significant; §5.6, §7, §11). Phase 3 is not coherent as written, and I recommend
moving it to its own SEP.**

- **The gate is empty.** "The stored record compares as continuing against the requested
  run rebuilt at the stored length" compares the stored run with its own recipe, so it
  always passes. Nothing compares the *longer* run with the stored one on the stored
  interval, and the longer run is what continues. A gradient `pressure=(0.8, 0.1)` with
  `length` replaced is a different profile on the stored interval (measured:
  `...density.breakpoints_m[1]` moves from 0.1 to 0.2).
- **The table entry cannot do it.** A length change from 0.1 m to 0.2 m at constant
  pressure moves 524 compared paths (measured): the density and pressure tables' `z_m`, the
  mixture tables and the absorber. `**.span` matches `configuration.profiles.core_radius.span`,
  but also `materials.policies.ionisation[].energy_loss.span`, which is a field range.
- **The absorber depends on the length.** `configuration.absorber.reference_length` and its
  rates move with the length (0.005 to 0.01, measured), so §7's "continued from 5 mm
  agrees with a 10 mm run" compares runs with different absorbers on the first 5 mm. The
  agreement is plausible but unmeasured.
- **The file's record cannot describe the extension.** §5.3 says it is never rewritten,
  but `configuration.save_planes` and `reference.basis` hold the planned planes.
  `reference_at` raises outside the frame record's span (`coupling/plane.py:605`), and
  a plane beyond the record gets an identity no port matches
  (`processing/planes.py:226`). Extended planes could not be exchanged.
- **Dependencies are misstated.** `solprop resume` and `Recipe.from_output` are SEP 41's
  phase 2, not its "recipe core". The header says **Requires** —. SEP 41 §5.2 has
  `solprop resume` check that "its hash equals the stored one", which conflicts with this
  SEP's gate. SEP 41 Q4 recommends "no in this SEP; extending a finished run is its own
  question", which does not rule this out, but neither SEP has been changed to match.
  Only recipe-written files could be extended, not any script-written or 0.1.0 file.
- **An alternative already exists.** `with_input` from the last plane continues a finished
  run into a new file today, with no machinery.

*Recommendation:* close this SEP at phase 2, and leave extension as a roadmap item on
SEP 41. If it stays, Part I must define the on-interval comparison and the record of an
extended file, and **Requires** must name SEP 41.

**R15 (significant; §6, §7). The verification would pass over most of the above, and the
fixtures need more thought.**

- §7's tolerance test perturbs "a compared array" uniformly (R1), the neutral tests would
  use hand-edited records (R3), and the materials test a synthetic manifest (R11). None
  goes through a file of "another release". §7 has no test of an interpolated last plane,
  of a complete file with changed numerics, of suspend and reopen, or of an integrator
  change after the ledger is written.
- `tests/output/test_record_stability.py` and `tests/record_numbers.py` already pin records,
  with their structure exact and their numbers to a bound. §6 proposes a second mechanism
  without saying how they relate.
- Frozen records "compared with a fresh build" go red on any optmat revision of a model the
  routes use, since the digest moves. Say whether that is intended, as `without_optmat_release`
  says it is not for the existing pins, or what refreshes them.
- "Captured from the 0.1.0 release": the `v0.1.0` tag exists but is not an ancestor of this
  draft's branch, which predates the route keyword rename. The fixtures must be built by
  checking out the tag, not from the branch. The measurements in this review were made on
  the branch.

*Recommendation:* extend the existing pins rather than add a second mechanism. Add the
tests named here.

#### Minor

**R16 (minor; §5.3). The numerics record can crash the run.** `RunOutcome` refuses
non-finite values (`output/schema.py:1861`), and `StepLimits.dz_max` defaults to `inf`
(`:1212`). Measured: `RunOutcome(..., execution={"resume": {"numerics":
{"stepper.limits.dz_max": [inf, 0.01]}}})` raises `OutputError`. A change from the default
would fail at the first outcome written. Encode non-finite values as text.

**R17 (minor; §5.1, §5.3). `check_cache` keeps its signature and returns a `bool`**, so the
comparison §5.3 needs is lost and the driver must compare again or receive it another way.
A small internal function returning the comparison, wrapped by `check_cache`, does it.
`compare` should also take the fast path on equal digests, particularly for processing's
`source=` check, which runs per plane where today's digest is cached.

**R18 (minor; §4, §5.2). The tree and pattern definitions are under-specified.**
- "The first matching pattern wins" and "the deepest node that matches decides" are two
  different rules.
- The `except` of the `materials.records` row is not in §4's syntax.
- Whether `modes[0]` is one component is not stated, yet `**.citation` has to cross it.
- Tags beyond `__ndarray__` are not covered: `__imaginary__`, `__enum__`, `__complex__`,
  `__callable__`, `reproducible`.
- A mixed integer and float leaf (`pressure=1` against `1.0`) is not covered.
- Dotted strings are ambiguous: a record key can hold a dot (measured:
  `He:ionisation_rate:rate/default/He@1030.5nm`). Match on tuples of components.
- `materials.records` is keyed by `material:kind:name` (`physdata/selection.py:876`), so
  "compared as a set keyed by digest" is a re-keying the walk must special-case, as must
  the species' digest lists, and neither is table data.

**R19 (minor; §6). "Their absence is now neutral by the generic rule" is wrong for most of
the conditional records.** `transverse_measure`, `plasma_integration`, `quiet_collar` and
`evanescent_rate` are keys of the route's `configuration` mapping, not dataclass fields,
so they would need $N$ entries. No harm follows if they stay conditional, as §6 proposes,
because both records lack them. But the sentence is misleading about what a later clean-up
costs.

**R20 (minor; §4, §11; partly a judgement). The history rule largely exists.**
`AdamsMarch._refusal` already refuses a snapshot whose method representation differs
(`backends/multistep.py:3046`). Measured: order and formulation changes restart with
`the Adams history starts afresh`, so phase 1's "history dropped on a method change" is
almost no work. It is a warning, though, not "exactly as `resume_history=False`", which
only logs. The representation also includes `max_calls`, `safety` and `max_growth`, so
raising a call budget, a use this SEP newly permits, restarts the history at order one
although the history is still valid. Consider comparing only the fields that affect
history.

**R21 (minor; §3, §8). The inventories are incomplete.** §3's list of hash tests omits the
stamp check (`output/hdf5.py:1733`), `_validate_metadata` (`output/scan.py:144`), the queue's
checks, `reopen`, and the snapshot and TDSE archive checks. §8 omits
`docs/guide/results/run-access.md:113`, the "Cache and resume" paragraph of
`output/hdf5.py`, the resume-key paragraph of `output/schema.py`, and SEP 41's §3 text.

**R22 (minor; §2; a judgement). Callables.** A callable is recorded by `repr`, which by
`metadata_for`'s own docstring makes the digest unstable across processes. So a script
with a callable profile can never resume, then or now, and the refusal gives no reason.
The draft says nothing about this. Given "trying hard", at least a refusal that says so
would help. Not exercised on a route.

**R23 (minor; §1, §6; a judgement). A simpler shape for the same result.** "Compatibility
code is confined to one table" no longer holds once R3, R6, R7, R8 and R10 are counted.
Those are mostly small edits to existing code rather than compatibility layers. The draft
can be smaller where it matters to the maintainer: one explicit $N$ and no default lookup,
field metadata or imports (R2); one class-prefix rule for optmat (R11); no phase 3 (R14).
That reaches the first half of the aim, resuming wherever the physics is unchanged, with
less of what the second half asks to avoid.

**R24 (minor; §5.7). The example does not match the record.** A pressure change gives about
520 refused paths on a capillary (measured), `configuration.medium.shared_pressure.values[i]`
being a list and not an array of "201 elements". So the summary must group list elements
too, as `summarise_differences` does, and "names every such path" (§4) needs softening.

#### Open questions

My view of each, beyond the points above:

- **Q1:** per element at $10^{-11}$ (R1, R5).
- **Q2:** conditional on a recorded parameter (R4).
- **Q3:** no objection.
- **Q4:** no, not by default (R9).
- **Q5, Q6, Q7:** agree with the recommendations.

#### Checked and found correct

- §9's "Nothing that continued before is refused now" holds at the gate: equal hashes
  mean bit-equal leaves, and the prototype refuses nothing for an identical record. The
  risk lies in the other direction (R1, R2, R4).
- The numerics class holds. `rtol`, `RungeKutta()`, formulation, order, `interpolate_planes`
  and `step_planes` each moved only `stepper.*` on the capillary record. `auto` to
  `recompute` resumed on GNLSE with the history kept.
- Step plane changes in both directions appended correctly. `fields="planes"` beside
  `step_planes` is refused by name. A schedule added to an adaptive file landed and ran.
- Rounding at the inputs: 4-ulp changes refused nothing by §4's rule, on the capillary with
  one and two modes and on radial, except the digest noted in R5.
- `modal`'s measured quantities are already quantised (`translate/general.py:1784-1794`) and
  the plasma certificate is already outside the record (`driver/simulation.py:1346`), so no
  rounding-level residual was found in a compared path. The general modal route was read,
  not run.
- optmat's digest excludes `status`, `status_reason`, `modified` and `base`, so keeping
  `modified` and `base` compared is right.
- A file of another schema is refused first, by `_refuse_schema`, as §5.5 requires.
- `ArrayDigest` appears only under `configuration.initial` (`driver/simulation.py:2617`).
- $10^{-12}$ is about 4500 ulp. §6's cost holds: on a 28.6 MB record, the prototype's walk
  took 0.04 s against 0.28 s for canonical JSON and SHA-256.

**Not checked:** a real pair of records from two platforms; the TDSE, general modal,
Cartesian and planar routes' records, or any run of them; a queue or Slurm scan, which
were read only; and anything of SEP 41 beyond its text, since it has no code.

### Response to review — 2026-11-01 (Opus 5.5, drafting agent)

R1, R3, R7 and R8 were re-checked against the code before this response: R1 by the
arithmetic ($6.8\times10^{-15}$ for the review's `betas`), R3 by reading `_rebuild` and the
stamp check, R7 and R8 by reading `_validate_resume` and `_open_output`. Part I is revised
as follows. All 24 points are taken, except as stated in points 5, 9 and 12.

1. **R1 taken.** The test is per element (§4). Only rows $T$ marks **tail** add an
   array-scale floor, namely the array launch, a passed-in noise spectrum and the absorber's
   rates. §7 has the `betas` case. The FFT-bound floor of `tests/record_numbers.py` is not
   copied, since it would hide β4.
2. **R2 taken.** An absent path is a difference unless $N$ lists it. The default lookup,
   field metadata and imports are gone (§4, §10).
3. **R3 taken.** `from_json` keeps the raw hashed part and tolerates absent and unknown
   fields, and the comparison works on the raw record (§5.4). §7 goes through real files
   with the stamp an earlier build would write. §9's claim about 0.1.0 files is withdrawn,
   for the further reason in point 15.
4. **R4 taken.** `configuration.initial` is ignored only where `configuration.pulse` is
   present, and `noise.spectrum` only for seeded input noise (§5.2). `with_input` removes
   the pulse record for an array launch (`driver/simulation.py`, `with_input`), which is
   what makes the condition safe. Modified noise is compared, not recorded as numerics,
   because it acts at every step.
5. **R5 taken, with one limit.** $\varepsilon = 10^{-11}$ is `RECORD_RTOL`, shared with the
   tests. The TDSE descriptor is stated as compared text (§5.2), and frozen records on every
   CI platform are the cross-platform check. Whether a digest ever takes in a libm-computed
   number remains open, and those frozen records would show it. They do not cover a TDSE
   record, because CI has no provider.
6. **R6 taken.** `save_planes`, `initial_z` and `grid.length` are exact (§4), so a moved
   length refuses by name rather than failing later.
7. **R7 taken.** The ledger is kept as provenance across a change to Runge–Kutta and is not
   installed (§5.6). Phase 1 edits both call sites, and §6 now names the backend edits.
8. **R8 taken.** `dense and` goes (§5.6), with a test both ways.
9. **R9 taken for files and for scans, as a refusal.** A complete file with changed numerics
   is refused with the remedy (§4). Neutral and rounding differences still reuse. For a
   scan, a done point with changed numerics is a conflict (§5.7). This reverses the first
   draft's Q4, which the maintainer had agreed, so Q4 asks again.
10. **R10 taken.** After a continuation the session's identity is the file's record (§5.3),
    which defines "keep the file's record" and leaves the exact checks valid.
11. **R11 taken.** Subtrees whose `__class__` begins with `optmat.` are dropped in
    normalisation, with rows for the wrapper's `source` and the policies' `model` and
    `status`, and a test on a real route that a changed number moves a digest (§4, §5.2,
    §7). `reference_temperature` stays compared until phase 1 establishes whether the digest
    covers it.
12. **R12 taken, by a smaller route than the one suggested.** The worker's check stays as it
    is, because a `conflict` result already carries the new record, and the parent compares
    only then (§5.7). `_Job` does not change, and no record travels in the common case.
    Queue and Slurm keep exact identity, stated in §3 and §5.7. No summary count is needed
    now that numerics are not reused.
13. **R13 taken.** "No bump" is limited to record fields, root attributes and top-level
    groups written once (§5.8), and §8 lists the places that contradict the rule.
14. **R14 taken.** Phase 3 is out; §2 says why, and the roadmap keeps the item. The SEP
    closes at phase 2 and needs nothing from SEP 41. SEP 41's Q4 decision text moves with
    it.
15. **R15 taken.** The frozen records extend `test_record_stability` rather than adding a
    second mechanism, an optmat data revision is a tolerated difference there, and §7 gains
    the missing tests. Fixtures come from a tag, checked out. Since the review, the
    `DataPulse` phase fix has taken the output schema to 2 on its way to main, so 0.1.0
    records would be refused by schema: the fixtures come from the latest release when phase
    1 lands.
16. **R16 taken.** Non-finite values are written as text (§5.3).
17. **R17 taken.** `check_cache` wraps an internal function that returns the comparison, and
    `compare` takes the digest fast path (§4, §5.1).
18. **R18 taken.** Paths are tuples, the first matching row decides, the tags are listed,
    integers against floats are stated, and re-keying is a normalisation step and not table
    data (§4).
19. **R19 taken.** §6 no longer says the conditional records are neutral by a generic rule.
20. **R20 taken.** §5.6 says the history check compares only the fields that define the
    history, so a raised call budget keeps it.
21. **R21 taken.** §3 is now a full table of what tests the hash and what becomes of it, and
    §8 lists the further pages and docstrings.
22. **R22 taken.** A callable stays compared, because its repr is all the record holds, and
    the refusal now gives the reason and the remedy (§4, §5.1). `resume="force"` also gets
    past it.
23. **R23 taken.** §1 now claims less about where the compatibility code sits and the design
    is smaller, as the point suggests: one table $N$, one class-prefix rule, no phase 3.
    `resume="force"` adds one class and one annotation, for the reason in the next
    paragraph.
24. **R24 taken.** Reports group elements, and "names every path" is softened to what
    `summary()` does (§5.1, §5.9).

**Where this departs from the review.** The review agreed with the first draft's Q5, no
override. The maintainer asked what that would exclude, noting that forcing is sometimes
useful, and the answer was: every compared difference, with `with_input` the only way on, at
the cost of a split file and restarted ledgers. So §5.5 adds `resume="force"`, limited by a
new layout class, and Q5 asks for confirmation. The layout class is new since the review and
has not itself been reviewed.

**Still unmeasured,** as the review says: a real pair of files from two platforms; the TDSE,
general modal and planar records; and queue and Slurm scans.
