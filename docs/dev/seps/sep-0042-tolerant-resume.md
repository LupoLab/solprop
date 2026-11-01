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
exactly. That is stricter than continuing the run needs. A later release that adds a
field, a newer optmat with the same numbers, a looser tolerance, or a platform whose
mathematics library rounds differently in the last bit each refuses a resume of the same
physics. This SEP replaces hash equality as the resume gate with a comparison of the two
records path by path. The comparison needs four rules:

- **classes:** a short table sorts every recorded path into one of three classes:
  - *compared*, which must agree (the default);
  - *numerics*, which may change and is recorded;
  - *ignored*, which is provenance, or a launch realisation that the stored field
    supersedes;
- **tolerance:** numbers agree to within a rounding-level tolerance rather than bit for
  bit;
- **neutral values:** a path one record lacks reads as its neutral value;
- **materials:** optmat models are compared by the digest of their numbers, not by
  version or name.

The output schema version leaves the comparison: a resume needs the same schema, and the
schema is bumped only for a layout an older reader would misread.

A final phase continues a finished run to a longer length. It depends on the recipes of
SEP 41, which is still in draft.

Success means:

- every 0.1.0 route's record resumes under later releases unless the physics changed;
- a resume with a changed `rtol` or integrator continues and says so;
- every refusal still names the paths that changed;
- compatibility code is confined to one table of classes and one table of neutral
  values and renames.

### 2. Motivation and scope

A resume should succeed whenever continuing the stored state with the requested problem
gives the run the user asked for, and be refused, with a reason, when it would not. The
present gate departs from that in four ways, each of which a user meets in ordinary
work.

- **A field added in a release breaks every earlier file.** A run record holds every
  field of every dataclass it serialises, at its default too (ADR-051). A release that
  adds a field to `PIController`, a pulse, a response or a mode moves every configuration
  hash, so nothing written by the previous release resumes, though the new field's
  default reproduces the old behaviour. Several conditional records in
  `metadata_for` (`transverse_measure`, `plasma_integration`, the trapezoid rule's
  absent `integration`) exist only to keep earlier hashes, and the roadmap's
  *A resume-compatibility policy before 1.0* asks what to do about it.
- **The material manifest is compared by version.** The resume key covers the optmat
  version, and each model's name, status, status reason and sources. An optmat release
  that changes no number a run uses still refuses its resume. So does a renamed model
  with the same digest.
- **Numbers are compared bit for bit.** The record holds derived arrays: the launch
  field `configuration.initial`, the noise spectrum, the mixture-processing tables, the
  absorber rates, the band rule's taps. They are built with the platform's `exp`, `sin`
  and FFTs. SEP 25 gave up bitwise agreement across platforms and builds, but the resume
  key did not, so nothing guarantees that a run started on one machine resumes on
  another.
- **Accuracy settings count as physics.** A run stopped part-way under `rtol=1e-6`
  cannot be finished under `rtol=1e-5`, nor with another integrator, though either
  continuation is well defined. The boundary check's budget was taken out of the key for
  the same reason (SEP 21).

In scope:

- the classification of recorded paths and the comparison rules;
- their use wherever the configuration hash is used as a compatibility test: the resume
  gate `check_cache`, an appending `HDF5Output`, a scan's finished points, and
  processing's `source=` checks;
- the record of a resumed session's numerical changes;
- the rule for when the output schema is bumped;
- continuing a finished run to a longer length (phase 3).

Out of scope:

- **Reading files of another output schema.** ADR-058 stands: a file is read only at its
  own schema. This SEP makes bumps rare; it does not add readers for old layouts.
- **Resuming with changed physics.** A deliberately changed model starts a new segment
  from a saved field ([interior launch](../../guide/results/interior-launch.md)). No
  override continues a run under different physics in the same file.
- **Bitwise reproduction of a resumed run.** A resume agrees with an uninterrupted run
  to the bounds SEP 25 set, as today.
- **The size of the run record.** The roadmap's *Large arrays in the run record* item is
  unchanged. The comparison treats an array the same whether it is held in the JSON or
  as a dataset.
- **Snapshot and suspend integrity checks.** They compare a file with the session that
  is writing it, and keep exact hash equality.

### 3. Context

The resume key is `RunMetadata.configuration_hash()` (`solprop/output/schema.py`). It is
a SHA-256 of the canonical JSON of `RunMetadata.hashed()`, which holds:

- the schema version and the simulation kind;
- the grid, the modes, the spatial and modal records, the quadrature and the power
  convention;
- the configuration: medium, guide, pulse, launch field, responses, profiles, absorber,
  save planes, statistics, precision and the derived processing tables;
- the noise model, seed and spectrum;
- the stepper record: tableau, controller, limits, schedule and integrator;
- the material manifest's identities and policies;
- the launch position and the TDSE descriptor.

Kept out are the wall-clock time, the package versions, the git provenance, the backend
record and the boundary check
([resuming](../../guide/results/resuming.md#the-resume-key)).

`check_cache` (`solprop/output/hdf5.py`) is the gate. On a mismatch,
`RunMetadata.differences` lists the dotted paths that differ and `summarise_differences`
words the refusal. Four other places test compatibility by hash equality:

- `HDF5Output.open` in append mode;
- `resolve_metadata_write`;
- the scan drivers' finished-point check (`solprop/driver/_scan_run.py` and
  `_scan_batch.py`);
- processing's check that `source=` wrote the output (`solprop/processing/planes.py`).

Two earlier rulings relaxed the gate for particular cases.

- **Unseeded noise.** A realisation drawn without a seed is drawn again from the file's
  seed before the comparison (`Simulation._adopt_recorded_seed`).
- **The boundary check.** Its budget and guard width are outside the key, so a run
  stopped at the window's edge resumes under a relaxed check.

A resumed session's backend and provider facts already go into `RunOutcome.execution`.
Earlier sessions' facts are kept in `previous_executions`, outside the scientific
identity.

The relevant decisions:

- **ADR-051:** no compatibility with earlier configurations; every field is recorded at
  its default; a file of another schema is never resumed.
- **ADR-058:**
  - the 0.x releases may break the resume key if the changelog says so;
  - a file of another schema is refused everywhere;
  - a resume-compatibility policy is to be decided before 1.0.
- **SEP 25:** no bitwise reproducibility anywhere; bounds instead.
- **SEP 14:** interpolated planes resume only with `resume="interpolated"`.

This SEP answers the roadmap item *A resume-compatibility policy before 1.0*
(*Release preparation*). It chooses neither option the item named; §10 says why.

### 4. Theory

**Records as trees.** A record is the JSON tree of `hashed()`:
- an object is a mapping;
- a serialised dataclass is a mapping with a `__class__` entry;
- an array is a mapping with `__ndarray__`, `dtype` and `shape`, treated as one leaf;
- a list is a sequence;
- every other value is a leaf.

A path $p$ is the dotted route to a node, with list indices as `[i]` and keyed tables
under their keys, as `differences` writes them today.

**Classes.** A table $T$ of path patterns (dotted names with `*` matching one component
and `**` any number) assigns each path a class. The first matching pattern wins, and the
deepest node that matches decides for its subtree:

$$
c(p) \in \{\text{compared},\ \text{numerics},\ \text{ignored}\},
\qquad c(p) = \text{compared} \text{ if no pattern matches.}
$$

Compared is the default, so a field nobody has classified is held to agreement: the safe
error is a refusal.

**Agreement of leaves.** Two leaves $a$ and $b$ at a compared path agree when:

- strings, booleans, integers and `None` are equal;
- floating-point scalars and arrays have equal shapes and dtype kinds, and
  $$
  \max_i |a_i - b_i| \le \varepsilon \max\!\Big(\max_i |a_i|,\ \max_i |b_i|\Big),
  \qquad \varepsilon = 10^{-12},
  $$
  with complex values measured by modulus, NaN agreeing only with NaN in the same place
  and an infinity only with the same infinity.

The scale is the larger array's maximum, not each element, so that the near-zero tail
of a spectrum is judged against the array's scale rather than its own.

$\varepsilon = 10^{-12}$ is about $4500$ units in the last place of a double. That is
well above the spread of a few libm ulps carried through the short chains of operations
that build these arrays. It is orders of magnitude below the stepper tolerances runs use
(the routes default to $10^{-6}$ or $10^{-5}$ relative) and below any parameter change a
user makes on purpose. Continuing with physics arrays that agree to $\varepsilon$
perturbs the solution at the same order as the rounding the march already makes at
every step. That is the kind of difference SEP 25 bounds rather than forbids.

**Absent paths.** A path present in one record and absent from the other is resolved
before comparison by a neutral value $\nu(p)$, the value that reproduces the behaviour
of the code before the path existed:

- If the parent is a serialised dataclass that still has the field, $\nu(p)$ is the
  field's `metadata["neutral"]` where declared, otherwise its default.
- Otherwise $\nu(p)$ comes from a table $N$ of neutral values. A path in neither is a
  difference.

The convention this rests on:

> A field added to a recorded dataclass takes, as its default, the value that reproduces
> the earlier behaviour, or declares that value as `neutral`.

A default changed deliberately needs nothing. The earlier record holds the old value,
which the comparison sees as a change in physics, and so it should.

**Materials.** The manifest's model records are compared as a set keyed by
`(material, kind, digest)`. A model's name, status, status reason, sources and optmat
version are *ignored*, and so is the manifest's optmat version. Species, compositions,
solids and policies are *compared*, with each species' record list read as digests. The
digest is optmat's SHA-256 of the model's canonical record, which identifies its
numbers. Two optmat versions that give a model the same digest give the run the same
numbers.

**Save planes.** `configuration.save_planes` is compared, with one exception that phase
3 uses. When the stored run is complete and the new planes extend the stored ones, it
counts as an extension rather than a difference. The new planes $z'_k$ extend the
stored planes $z_k$ when $z'_k$ agrees with $z_k$ for every stored $k$ and there are
more of them.

**Verdict.** For stored record $S$ and requested record $R$, after resolving absent
paths:

- the resume is **refused** if a compared path disagrees, and the refusal names every
  such path;
- otherwise it **continues**, and the numerics paths that differ are reported and
  recorded.

Ignored paths are never looked at.

**Numerical changes on continuing.** With the controller held to $\text{rtol}_1$ on
$[0, z_r]$ and $\text{rtol}_2$ on $[z_r, L]$, the continued run is a concatenation of
two adaptive integrations from one state. Its local error is bounded on each segment by
that segment's tolerance. So the whole run is no better than the looser of the two.
That is a statement about accuracy, not identity, and it is recorded, not refused.

Some integrator changes leave the stored history unusable:
- a change of method (Multistep versus Runge–Kutta, or another Adams formulation);
- a change of order or of the history's parameters.

After such a change the Adams history is not reused, and the march starts at order one
from the stored field, exactly as `resume_history=False` does today.

### 5. Proposed interface

#### 5.1 The comparison

A new module `solprop.output.compatibility` holds the class table $T$, the neutral table
$N$ (with its class-rename entries) and the comparison. `RunMetadata` gains one method:

```text
comparison = stored.compare(requested)      # RecordComparison
comparison.continues                        # bool
comparison.refused                          # tuple of PathChange (compared paths)
comparison.numerics                         # tuple of PathChange (allowed, recorded)
comparison.summary()                        # the refusal or report text
```

`PathChange` holds the path, the stored value and the requested value, with arrays
summarised by shape and largest difference. The values of an absent path appear as
`absent → neutral`. `RunMetadata.differences` and `summarise_differences` stay, for
displays that want every difference, numerics included.

`check_cache(path, metadata)` keeps its signature:

- it returns `True` when the comparison continues;
- it raises `OutputError` with `comparison.summary()` when the comparison refuses;
- it logs the numerics changes at warning level when it continues.

The `materials` note in the refusal changes accordingly. It names the models whose
digests changed, not the optmat version.

#### 5.2 The class table

The initial table, in order. A path not matched is compared.

| Pattern | Class | Why |
|---|---|---|
| `stepper.**` | numerics | tableau, controller, limits, schedule, integrator: how accurately, not what |
| `configuration.propagator_mode` | numerics | which equivalent linear propagator form steps the run |
| `configuration.initial` | ignored | the launch field realised on the grid; the stored field supersedes it |
| `noise.spectrum` | ignored | the realisation drawn from `noise.model` and `noise.seed`, both compared |
| `materials.optmat_version` | ignored | provenance |
| `materials.records.**` except `material`, `kind`, `digest`, `modified`, `base` | ignored | provenance; the digest identifies the numbers |
| `**.citation` | ignored | a citation string, which may carry a source file's hash |

The pulse parameters, noise model and seed, launch position, precision, statistics, save
planes, grid, modes and every physics record stay compared. A resume whose launch
differs is refused. The stored field does supersede the launch, but a script that
changed its pulse and kept its output path is more likely a mistake than a wish to
continue the old run. The unseeded-noise ruling is kept.

The table is data, reviewed like code. A test asserts that every pattern matches a path
in at least one route's record, so that a stale row is noticed.

#### 5.3 The record of a resumed session

When a resume continues with numerics changes, the session's
`RunOutcome.execution` gains a `resume` entry:

```text
{"from_z": 0.0412, "from_plane": 82,
 "numerics": {"stepper.controller.rtol": [1e-06, 1e-05], ...},
 "neutral": ["stepper.controller.new_field"]}
```

It lists:

- the position the session started from;
- each numerics path that changed, with its stored and requested values;
- the paths that were read as neutral.

`previous_executions` keeps it with the session, so a reader can see which segment was
stepped how. The file's stored `RunMetadata` is never rewritten. It describes the run as
first written, and the executions describe how each session continued it.

#### 5.4 Scans, appends and processing

- **Appending.** `HDF5Output` in append mode, and `resolve_metadata_write`, use
  `compare`. They keep the file's record when it continues and raise as today when it
  refuses.
- **Scans.** A scan's finished point is reused when its stored record and the point's
  new record continue, so the scan reruns only points whose physics changed. A
  finished point whose numerics differ is reused, not rerun, and the scan's closing
  summary counts such points. A point that refuses is a `conflict`, as today. The
  per-point `configuration_hash` stays in the scan file as the point's exact identity.
- **Processing.** `source=` is accepted when the source simulation's record and the
  output's continue.

#### 5.5 The output schema

`schema_version` stays in the record and in the configuration hash, but leaves the
comparison:

- **Resume:** a resume needs the stored file's schema to equal the current one, and
  says so if it does not. This is a precondition, checked before the comparison.
- **When to bump:** the schema is bumped only for a change that an older reader would
  misread or that an older file cannot be appended to:
  - a dataset renamed, re-laid or given new units;
  - an attribute changing meaning;
  - a required group added.
- **No bump:** adding an optional group, attribute, annotation or record field does
  not bump the schema. Readers ignore groups and attributes they do not know, and read
  a missing optional one as absent. Phase 2 checks that every reader does, and states
  the rule.

The rule goes into `docs/guide/results/output.md` and a new ADR, which narrows ADR-051's
"a file of another output schema is never resumed, through its resume key" to "through
the schema precondition". The ADR also replaces ADR-051's "no field is left out to keep
an earlier record's hash" by the neutral-value rule. Fields are still recorded in full.

#### 5.6 Continuing a finished run (phase 3)

A complete run is continued to a longer length when:
- the stored record compares as continuing against the requested run rebuilt at the
  stored length;
- the requested save planes extend the stored ones.

Rebuilding at the stored length needs the builder's arguments, which a `Simulation`
does not keep. SEP 41's resolved recipe carries them, so this phase needs SEP 41's
recipe core. Its interface is SEP 41's `solprop resume FILE` with a new `--length L`
option, and the recipe with its length replaced in Python. This answers SEP 41's Q4 for
the length. A `Simulation.run(..., resume=True)` whose length exceeds a
complete file's says how to continue through the recipe, rather than refusing without a
reason.

Extending is refused for:
- a run whose length-dependent configuration is not a function of $z$ restricted to the
  stored interval, such as a profile sampled on the run's own planes, with the paths
  named;
- a budget-exhausted run, which is unfinished and resumes as today.

Phase 3 adds an entry to the class table: `**.span` and the configuration's `length`
entries are compared on the stored interval.

#### 5.7 Example

```text
sim = Simulation.from_capillary(..., rtol=1e-6)
sim.run("run.h5")                   # stopped at z = 41 mm

# after upgrading solprop and optmat, with no physics change:
sim = Simulation.from_capillary(..., rtol=1e-5)
sim.run("run.h5", resume=True)
# WARNING resuming run.h5 at z = 0.0412 m with changed numerics:
#   stepper.controller.rtol 1e-06 -> 1e-05
# INFO 2 paths read as neutral: stepper.controller.new_field, ...

sim = Simulation.from_capillary(..., pressure=1.2)
sim.run("run.h5", resume=True)
# OutputError: run.h5 cannot be resumed: these compared paths differ:
#   configuration.profiles.pressure.value 1.0 -> 1.2,
#   configuration.medium.shared_pressure (201 elements, max |Δ| 0.2), ...
#   To continue with different physics, launch a new segment from a saved
#   plane (interior launch).
```

### 6. Implementation strategies

**Where the classification lives.** There are two choices.

- **One table of path patterns (recommended).** It covers configuration keys, which
  are not dataclass fields, and keeps every compatibility decision in one reviewable
  place.
- **Annotate fields with `dataclasses.field(metadata=...)`.** This puts the class
  beside the field. But it spreads the decisions over the package, and it cannot
  annotate the route-built configuration mapping.

Field metadata is used only for the `neutral` value, which belongs with the field's
default.

**Layer.** The work belongs to the output layer (`solprop/output`), with call sites in
the driver and processing. No physics, translation or backend code changes. The kernel
contract and `KERNEL_SCHEMA_VERSION` do not change. No backend is touched.

**The comparer.** It is a recursive walk over two parsed records, like
`_differences`, but with arrays as leaves and the class decided on entry to each node.
An ignored subtree is skipped. The stored record is parsed once. The cost is linear in
the record's size, so tens of milliseconds for the largest spatial records the roadmap
mentions (about 20 MB). Today's check serialises and hashes that record, which costs
about the same.

**Class renames.** A recorded `__class__` names the class's module path. A class moved
between modules adds one line to $N$'s rename section, mapping the old path to the new
one. A class renamed for a new meaning adds nothing, and the resume is refused.

**Retention.** Neutral and rename entries record the release that added them (Q3).

**What the comparison guards.** One test per route records a run at a fixed small
configuration. The records captured from the 0.1.0 release are frozen as JSON fixtures
under `tests/data/records/` and compared with a fresh build of the same configuration.
A change that adds a field with a non-neutral default fails the test. The developer then
either declares the neutral value or accepts the break, lists it under "Changed
(breaking)" in the changelog and refreshes the fixture. The test is what enforces the
convention. Without it, the neutral rule would be a promise.

**Retiring earlier workarounds.** The conditional records in `metadata_for` that exist
only to keep earlier hashes stay as they are. Their absence is now neutral by the
generic rule, so they cost nothing. A later clean-up may write them unconditionally.

**New dependencies.** None.

### 7. Verification and validation

The tests are P (structural) and R (rounding) class (`docs/dev/testing.md` §3). Nothing
here changes a computed field.

- **Classification:**
  - every pattern in $T$ matches some path in at least one route's record;
  - every top-level key of `hashed()` is reached by the walk.
- **Tolerance:**
  - perturbing a compared array by $10^{-13}$ relative continues;
  - perturbing it by $10^{-10}$ refuses and names the path;
  - NaN, infinity and complex-modulus cases;
  - a zero array against a zero array.
- **Neutral values:**
  - a stored record with a field removed resumes when the field's default is neutral,
    and refuses when it is not;
  - a path in neither record nor table refuses;
  - a renamed class resolves through $N$.
- **Materials:**
  - a manifest with a changed optmat version, model name or status, and unchanged
    digests, continues;
  - a changed digest refuses and names the model.
- **Numerics:** a capillary run stopped part-way resumes with a changed `rtol` and with
  `RungeKutta()` instead of the default `Multistep`.
  - The `resume` execution entry records both.
  - The Adams history is not reused after the method change.
  - The resumed run agrees with an uninterrupted run at the looser tolerance, to the
    T-class envelope `testing.md` already uses for resumed runs.
- **Launch:**
  - a changed pulse energy refuses;
  - an unseeded rerun continues;
  - a perturbed `configuration.initial` alone continues.
- **Frozen records:** the 0.1.0 fixtures, per route, compare as continuing with the
  current build.
- **Scans:**
  - a two-point scan whose stored points differ only in numerics is reused, and its
    summary counts both points;
  - a point whose physics changed is a conflict.
- **Extension (phase 3):**
  - a complete capillary run continued from 5 mm to 10 mm agrees with a 10 mm run to the
    resumed-run envelope;
  - non-extending planes refuse with the remedy.

No benchmark is needed. The comparison's time on the largest test record is logged once
to confirm that it is no slower than the hash it replaces.

### 8. Documentation

- **`docs/guide/results/resuming.md`:** "The resume key" becomes "What a resume
  compares". It covers:
  - the three classes and the table;
  - the tolerance;
  - neutral values;
  - materials by digest;
  - the numerics record, and an example refusal and an example continuation.

  "Noise drawn without a seed" stays.
- **`docs/guide/results/output.md`:** the schema-version paragraph says when the schema
  is bumped and that a resume needs the same schema.
- **`docs/dev/contributing.md` or `structure.md`:** the convention for a new recorded
  field (a neutral default or a declared `neutral`), and the frozen-record test.
- **Docstrings:** `RunMetadata.compare`, `RecordComparison`, `PathChange` and
  `check_cache`.
- **ADR:** a new ADR for the policy, narrowing ADR-051 and answering ADR-058's open
  policy.
- **Roadmap:**
  - the *Release preparation* item is removed, since this SEP covers it;
  - phase 3 is linked from SEP 41's Q4.

### 9. Compatibility and impact

- **Behaviour:** resumes that were refused now continue where the run is unchanged. A
  resume with changed numerics continues with a warning where it used to be refused.
  Nothing that continued before is refused now, because every path the old key compared
  bit for bit is compared at least as loosely, or not at all.
- **Breaking changes:** none to the API. `check_cache`'s refusal text changes.
- **Output schema:** unchanged. The `resume` entry goes into the free-form
  `RunOutcome.execution` mapping.
- **`KERNEL_SCHEMA_VERSION`:** unchanged.
- **The configuration hash:** unchanged. It is still stamped in every file and used for
  exact identity.
- **Validated results:** unaffected, since no computed field changes.
- **Luna.jl:** no difference to record.

Files written by 0.1.0 benefit at once, because they already hold full records and
material digests.

### 10. Alternatives considered

- **Hash with the stored file's own rules.** Keep one hashing function per released
  record layout, and hash the requested run as the stored file's release would have.
  This is a growing set of versioned hashers, which is the compatibility code the
  maintainers want to avoid. It also still compares bit for bit and still refuses
  numerics changes.
- **Reintroduce omit-at-default.** Record a field only when it is not at its default.
  This keeps hashes stable when fields are added, but not when they are removed or
  when a default changes. It also gives up the record's statement of every value used,
  which ADR-051 chose for good reason. The neutral-value rule gets the same tolerance of
  added fields while keeping full records.
- **Ignore the launch.** The stored field supersedes the launch, so the pulse and noise
  could be ignored on resume. This is rejected because it lets a mistaken resume
  continue the wrong run silently (§5.2). Only the realised arrays are ignored.
- **An override such as `resume="force"`.** Continue in the same file whatever the
  compared paths say, and record the differences. This is rejected for now: interior
  launch already continues from a saved field with different physics, in a file that
  describes it. See Q5.
- **Ask for numerics changes to be opted into** (`resume="numerics"`). This adds a
  value for a choice users make on purpose when they change `rtol`. A warning and a
  record serve as well. See Q6.

### 11. Phases

All phases go on one branch, `feature/sep-0042-tolerant-resume`, with one pull request
opened at Final (SEP 0).

1. **Comparison and resume gate.**
   - **Deliverables:**
     - `solprop.output.compatibility` with $T$, $N$, the tolerance and
       `RunMetadata.compare`;
     - `check_cache`, append mode, `resolve_metadata_write` and processing's `source=`
       check on the comparison;
     - the `resume` execution entry;
     - the Adams history dropped on a method change;
     - the frozen 0.1.0 record fixtures;
     - the manual sections of §8 and the ADR.
   - **Verifies:** §7 up to the frozen records.
2. **Scans and the schema rule.**
   - **Deliverables:**
     - finished-point reuse by comparison in both scan drivers, with the summary count;
     - the schema precondition, and the bump rule in `output.md` and
       `contributing.md`.
   - **Verifies:** the scan tests of §7, and a file with an unknown optional group read
     and resumed.
3. **Continuing a finished run.** This needs SEP 41's recipe core.
   - **Deliverables:** the extension rule for save planes and spans, and
     `solprop resume --length`.
   - **Verifies:** the extension tests of §7.

### 12. Open questions

- **Q1. The tolerance.** $\varepsilon = 10^{-12}$ relative to the array's largest
  magnitude, or bit equality as today.
  - *Recommendation:* $10^{-12}$. Bit equality is what keeps a run from resuming on
    another machine.
- **Q2. The launch realisations.** Ignore `configuration.initial` and `noise.spectrum`
  (the parameters that made them stay compared), or compare them within tolerance.
  - *Recommendation:* ignore them. A change in how a pulse is realised on the grid, or
    in NumPy's generator, says nothing about the continuation, and the stored field
    supersedes both.
- **Q3. How long neutral and rename entries live.** Keep each entry for the release
  that adds it and the next minor release, then delete it with the frozen fixture it
  serves. Or keep them until 1.0.
  - *Recommendation:* the current and previous minor release. The table stays short,
    and a user two releases behind finishes the run with the older install.
- **Q4. Scans and numerics.** Reuse a finished point whose numerics differ, or rerun
  it.
  - *Recommendation:* reuse it and count it in the summary. A user who changed `rtol`
    to recompute writes to a new scan file.
- **Q5. A force override.** Offer `resume="force"`, recording every difference.
  - *Recommendation:* no, because interior launch covers changed physics. Revisit if
    users ask for it.
- **Q6. Opting into numerics changes.** Allow them on `resume=True` with a warning, or
  require `resume="numerics"`.
  - *Recommendation:* allow them with a warning and the execution record.
- **Q7. Precision.** Keep `configuration.precision` compared, or make it numerics.
  - *Recommendation:* compared. A single-precision continuation of a double run changes
    the error floor of the continuation, and the output guide promises that a run is
    never resumed in the other precision.

### 13. References

1. solprop, [Resuming a calculation and checking compatibility](../../guide/results/resuming.md).
2. solprop, [Output: the schema version](../../guide/results/output.md#the-schema-version).
3. solprop, [Design decisions](../decisions.md): ADR-051 and ADR-058.
4. solprop, [SEP 25 — Rounding-level reproducibility](archive/sep-0025-rounding-level-reproducibility.md).

## Part II — Implementation record

## Part III — History

- 2026-10-31: Drafted.
- 2026-11-01: AI review (Opus 5.5).

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
