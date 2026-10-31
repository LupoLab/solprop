# SEP 42 — Resuming by what matters to the run

| | |
|---|---|
| **SEP** | 42 |
| **Title** | Resuming by what matters to the run |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
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
