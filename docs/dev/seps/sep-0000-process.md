# SEP 0 — Purpose and process

| | |
|---|---|
| **SEP** | 0 |
| **Title** | Purpose and process |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 Astra |
| **Status** | Active |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Process |
| **Created** | 2026-09-26 |
| **Resolution** | Adopted with ADR-035, 2026-09-26 |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## 1. What a SEP is

A solprop Enhancement Proposal (SEP) is a design document for one major change to solprop:
a new physical model, route or numerical scheme, a changed default or convention, a
redesigned public interface. It states what is proposed and why, sets out the theory with
its references, discusses how the change could be implemented, and says how it will be
shown to work. Once accepted it is implemented on branches, in phases, and each phase
appends a report to the same document. When the work is complete the SEP is archived and
the manual pages it produced link to it, so that every major feature carries the record of
how it was specified, built and verified.

The process follows established practice in the scientific Python community: Python's
PEPs ([PEP 1](https://peps.python.org/pep-0001/), the process, and
[PEP 12](https://peps.python.org/pep-0012/), the template) and NumPy's NEPs
([NEP 0](https://numpy.org/neps/nep-0000.html) and the
[NEP template](https://numpy.org/neps/nep-template.html)). It differs from them in one
respect: a SEP is also the record of its own implementation, because in a numerical code
the decisions made while building and qualifying a feature — a tolerance, a refused case,
a measured cost — matter as much as the proposal.

solprop was developed from the start from written specifications, each implemented,
verified against independent references and reviewed before it merged (ADR-024). SEPs
formalise that practice for an open project. They apply to new work from the adoption of
this document; no SEP is written retrospectively for functionality that already exists.

## 2. When a SEP is needed

A SEP is required for:

- a new physical model or response (a nonlinearity, an ionisation rate, a material model);
- a new propagation route, geometry, spatial decomposition or mode family;
- a new numerical scheme: a stepper, a transform, a quadrature, a coupled flow;
- a change to a physics default or to a convention of [conventions.md](../conventions.md);
- a redesign of a public interface, or a change that breaks many users' scripts;
- a substantive change to the kernel contract ([kernel-contract.md](../kernel-contract.md))
  or a new backend;
- a change to how solprop is developed (a *Process* SEP, like this one).

A SEP is not needed for bug fixes, documentation, tests, refactoring, tooling and
continuous integration, dependency updates, small additions to the public interface (a
keyword, a helper, a statistic), or optimisations that leave results unchanged (they
still need a benchmark ledger entry, [benchmarks.md](../benchmarks.md)). Those follow the
working agreement of `AGENTS.md`: an issue or a short agreed plan, then a pull request.

When it is unclear whether a change needs a SEP, the maintainers decide; they may ask for
one on an issue or a pull request that turns out to be larger than it looked.

## 3. Types

| Type | For |
|---|---|
| Feature | New functionality: a model, a route, a scheme, a tool |
| Change | A change to existing behaviour: a default, a convention, an interface, the seam |
| Process | How solprop is developed, reviewed or released |

## 4. Statuses

```text
Draft ──► Accepted ──► In progress ──► Final
  │           │             │
  ├──► Deferred ◄───────────┤
  ├──► Withdrawn            └──► (back to Draft for a substantial revision)
  └──► Rejected
Any status ──► Superseded (by a later SEP)
```

| Status | Meaning | Lives in | On the roadmap |
|---|---|---|---|
| Draft | Written and under discussion; may change freely | `seps/` | yes, under drafts |
| Accepted | Agreed by the maintainers; waiting to be picked up | `seps/` | yes |
| In progress | Being implemented; at least one phase has started | `seps/` | yes |
| Final | Implemented on every backend, tested and documented | `seps/archive/` | no |
| Deferred | Not pursued for now; may return to Draft | `seps/` | no |
| Rejected | Decided against; the reasons are recorded in the SEP | `seps/archive/` | no |
| Withdrawn | Abandoned by its author | `seps/archive/` | no |
| Superseded | Replaced by a later SEP, named in the header | `seps/archive/` | no |
| Active | A Process SEP in force | `seps/` | no |

The folder, the [index](index.md) and the roadmap are kept consistent with the status in
the header; `tests/test_seps.py` checks it.

## 5. Roles

- **The author** is the person among the maintainers who initiated the SEP and manages it
  through drafting and acceptance, named in the **Author** field. The author is always a
  person.
- **Drafting agents** are the AI agents, if any, that helped draft or revise the proposal.
  They are named with their model and version in the **Drafting agent** field. An agent
  that joins at a later revision is added with the date of that revision. A drafting
  agent is never the author.
- **Implementers** carry out its phases. They may be the author or not, and may be people
  or coding agents; a phase can be picked up by a different implementer from the one
  before, from the SEP and its reports alone.
- **Reviewers** review the proposal before acceptance and each phase's pull request. Each
  review of the proposal is recorded in the **Reviewers** field as `Name (human)` or
  `Model version (AI)`, and the **Review status** field summarises them (§6, *Review*).
  The author may be one of the human reviewers.
- **The maintainers** accept, reject or defer a SEP, and approve the Final status.

## 6. The lifecycle

### Drafting

1. Copy [the template](template.md) to `docs/dev/seps/sep-NNNN-short-slug.md`, taking the
   next free number from the [index](index.md). Numbers are never reused, even for a
   withdrawn or rejected SEP.
2. Fill in the header and Part I: the **Author**, any **Drafting agent**, status `Draft`,
   review status `Not reviewed` and **Reviewers** `—`. Leave Part II empty and start
   Part III with the date drafted.
3. Add the SEP to the index table, the index's toctree and the *Draft enhancement
   proposals* section of the [roadmap](../roadmap.md), after the accepted proposals.
   Review the informal roadmap items in the same change: remove an item only when the
   SEP covers it in full; for partial coverage, link the SEP, state what it covers and
   retain the remaining work. Discussion of a possible future extension does not count
   as coverage by the proposal's implementation scope.
4. Open a pull request on a branch named `docs/sep-NNNN-short-slug`. A draft merges
   early — once it is coherent, not once it is agreed — so it appears in the developer
   documentation and can be discussed.

Revising a draft is free: edit it in place in further pull requests and add a line to
Part III for each substantial revision. Update its roadmap summary and the informal
items whenever its scope changes; restore work that the revised scope no longer covers.

### Review

A proposal is reviewed while it is a draft. A review is a reading of the whole of Part I
against the acceptance criteria below. When a review is complete, the reviewer is added to **Reviewers** as
`Name (human)` or `Model version (AI)`, and a line goes into Part III. The AI entry names
the model and version, for example `Opus 5.5 (AI)`. **Review status** is one of:

| Review status | Meaning |
|---|---|
| Not reviewed | No review is recorded |
| In review | At least one review is recorded, but not yet one human and one AI review |
| Reviewed | At least one human and at least one frontier AI model have reviewed it |

**Review in place, revise separately.** A reviewer, human or AI, records findings in the
SEP without rewriting Part I. The findings go in a dated subsection under Part III
(for example `### AI review — 2026-10-18`), as numbered points (R1, R2, …). Each point
names the sections it concerns, the problem, its priority and a recommended change,
with any evidence, probe or counterexample behind it. A maintainer's decision on a
point is recorded beside it. The drafting agent then answers every point in a
`Response to review` subsection, saying what changed and where, or why a
recommendation was not taken, and makes the revisions to Part I itself. This keeps
the author's text and the reviewer's reasoning separately readable, and makes each
change traceable to the point that caused it. A reviewer edits Part I directly only
for trivial corrections (typos, broken links), and says so in the review.

**Every SEP must be reviewed by at least one human and one frontier AI model before it is
accepted.** `tests/test_seps.py` checks that an accepted, in-progress or final SEP has
review status `Reviewed`, and that the status agrees with the recorded reviewers.

### Acceptance

A SEP is ready to accept when:

- the motivation names a concrete use case;
- the theory is complete: every equation derived or cited, its approximations and range of
  validity stated, in the conventions of [conventions.md](../conventions.md);
- the implementation options have been compared and one is recommended, with its place in
  the layers ([architecture.md](../architecture.md)) and its effect on the kernel contract;
- the verification plan names analytic solutions, limits, invariants or independent
  references, and says what will set each tolerance;
- the documentation plan names the manual page, and a tutorial or worked example where
  appropriate;
- the phases are defined, each independently mergeable;
- the open questions are resolved, or explicitly deferred to a named phase;
- its review status is `Reviewed`: at least one human and one frontier AI model have
  reviewed it (*Review*, above).

The maintainers accept a SEP in a pull request that sets the status to `Accepted`, fills
in **Resolution** (the date and the pull request or discussion where it was decided), adds
a line to Part III and moves the SEP from *Draft enhancement proposals* to *Accepted
enhancement proposals* on the [roadmap](../roadmap.md). A rejection or deferral records
the reasons in the SEP, removes its proposal listing from the roadmap and restores any
still-relevant informal work that it had replaced. Apply the same roadmap update on
withdrawal; for supersession, reconcile the work with the replacement SEP's scope.

From acceptance, Part I is frozen. A deviation found during implementation is recorded in
the phase report (§8), not edited into the proposal. A change large enough to invalidate
the proposal sends the SEP back to `Draft` for re-acceptance, or is made in a new SEP that
supersedes it. A return to `Draft` moves its roadmap listing back to the draft section.

### Implementation

An implementer picks up an accepted SEP in a pull request that sets the status to
`In progress` and records, in the **Implementation** field, who is implementing it and the
branch of the first phase. Each phase is then:

1. implemented on its own branch, `feature/sep-NNNN-short-slug` (with a suffix naming the
   phase when there are several), under the rules of §7;
2. qualified with the full gate and lanes of `AGENTS.md`;
3. reported: the phase report (§8) is appended to Part II in the same pull request, and
   the **Implementation** field and Part III updated;
4. reviewed and merged.

Phases may be taken by different implementers, and independent phases may run in
parallel. Work in `solprop-kokkos` is made in that repository on a branch of the same
name; its pull request cites the SEP, and its report goes into the SEP here.

### Completion

A SEP becomes `Final` when:

- every phase is merged and reported;
- the functionality runs on every backend (`numpy_ref`, `numba_cpu` and the Kokkos CPU
  build), with parity tests — or the accepted SEP records a justified exception, which
  then has a roadmap item;
- the manual documents it, and each manual page it produced ends with a link to the SEP;
- the open questions left are moved to the roadmap or to issues, and named in the last
  report.

In the pull request that sets `Final`: fill in **Manual pages**, move the file with
`git mv` to `docs/dev/seps/archive/`, move its index row to the archived table and its
toctree entry to the archive index, fix the relative links the move breaks, remove its
roadmap link, and cite the SEP in the `CHANGELOG.md` line.

A Final SEP is a historical record. It is not updated to follow later changes to the code;
the manual and the design pages describe current behaviour. Only errata and a
"Superseded-by" note are added afterwards.

## 7. Writing and implementing a SEP

### A SEP stands alone

A SEP is written so that a contributor, or a coding agent, who has never seen the
discussion behind it could implement it from the SEP and solprop's own documentation and
code. It may refer to:

- solprop and `solprop-kokkos`: their documentation pages, modules, tests and ADRs;
- optmat, solprop's companion optical-materials package: its documentation, modules and
  data files, cited by module or page name;
- the scientific literature, with DOIs where they exist;
- public reports, standards and data sources.

Apart from optmat, it does not refer to any other code's source; nor does it refer to
local files, machines or paths, or to private notes. A proposal may well be informed by another implementation; the SEP then re-derives
what it takes, chooses deliberately and cites the literature, as `AGENTS.md` asks of all
code ("re-derive, do not transcribe"). A deliberate difference from Luna.jl's behaviour is
still recorded on the [differences from Luna](../../guide/workflows/luna.md) page when it is implemented.

Theory is written in SI units and in the conventions of
[conventions.md](../conventions.md), with maths in MyST dollar notation; a departure from
a convention is stated where it occurs. The implementation section proposes strategies
and compares options; it is not a line-by-line plan, which belongs to the implementer.

### Implementation rules

- **Backends in order.** A feature is implemented first on `numpy_ref`, the oracle, with
  its tests and its manual page; then on `numba_cpu`; then on the Kokkos CPU build in
  `solprop-kokkos`. Device (GPU) qualification follows the standing policy of
  [backends.md](../backends.md), later and in one pass.
- **Parity, eventually.** Until a backend implements the feature it refuses the problem by
  name, with an error saying which backend does support it; it never falls back silently.
  A SEP is Final only when all backends agree within tolerances derived from summation
  bounds, as for every other feature ([testing.md](../testing.md)).
- **The seam.** A change to the kernel problem, the stage bundles or the protected driver
  surface bumps `SCHEMA_VERSION`, as `AGENTS.md` requires; the SEP says in which phase.
- **Tests.** Unit, invariant, convergence and parity tests as [testing.md](../testing.md)
  sets out, against the analytic and independent references the SEP names; every
  tolerance justified in a comment.
- **Documentation.** Comments on the physics and numerics, docstrings on every public
  name, a manual page with the background and a worked example, and, where it helps, a
  tutorial or a worked example in the galleries.
- **Qualification scope.** The limits of `AGENTS.md` hold: ten steps or a very short fibre
  in tests and timings; full-length problems only in the galleries.

## 8. The implementation record

Part II of a SEP holds one report per phase, appended when the phase merges. The reports
are how a SEP can be continued by someone else, and how the reasoning behind the finished
code is kept. Each report states:

- **Scope:** the phase, the date, the implementer, the branch and the merge commits (in
  both repositories when `solprop-kokkos` is changed).
- **Delivered:** the modules, tests and pages added or changed, as paths.
- **Deviations from the proposal**, and why each was necessary.
- **Decisions:** choices made during implementation that the proposal left open. A
  decision of lasting architectural weight is also recorded as an ADR in
  [decisions.md](../decisions.md), and the report cites its number.
- **Verification:** the gate and lanes run, on which tree; the references, measured
  errors, convergence orders and benchmarks, with the windows measured; what was not run.
- **Sources checked:** which constants, coefficients and citations were checked against
  their primary source, and which could not be.
- **Limitations:** cases refused, ranges not qualified, backends still missing.
- **Open questions**, numbered Q1, Q2, … through the whole SEP, each with its owner or
  destination (the next phase, the roadmap, an issue).
- **Handoff:** what the next phase needs to know.

A report is written honestly: a check that failed, a test left weaker than planned, an
assumption that had to be invented are all said plainly.

## 9. Relation to the other records

| Record | Holds |
|---|---|
| SEP | The proposal, and the history of how it was implemented and verified |
| ADR ([decisions.md](../decisions.md)) | A durable decision and its reasons; code comments cite ADRs |
| Manual (`docs/`) | How the functionality works and how to use it, as it is now |
| Design pages (`docs/dev/`) | How the code is built, as it is now |
| [Roadmap](../roadmap.md) | Open work: accepted and draft SEPs, then ideas and known issues not fully covered by one |
| `CHANGELOG.md` | What changed in each release, citing the SEP |

The theory of a SEP moves into the manual when it is implemented: the manual page carries
the derivation in its current form, and the SEP remains the record of the proposal.

## 10. Changing this process

This process is changed by a Process SEP, or for small clarifications by a pull request
that edits this document and adds a line to its history.

## History

- 2026-09-26: Drafted and adopted (ADR-035).
- 2026-09-27: List draft SEPs on the roadmap after accepted proposals and reconcile
  informal items against each proposal's full or partial scope throughout its lifecycle.
- 2026-09-29: The header's **Authors** became **Author**, always the person among the
  maintainers who initiated and manages the SEP. Added **Drafting agent**, **Review status** and
  **Reviewers**, and the rule that every SEP is reviewed by at least one human and one
  frontier AI model before acceptance. Existing SEPs were given the new fields.
- 2026-10-18: Reviews are recorded in place as numbered points under Part III, without
  rewriting Part I; the drafting agent responds point by point and revises the
  proposal (maintainer's preferred practice, first used for SEP 26).
