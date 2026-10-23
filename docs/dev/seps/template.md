# SEP NNNN — Title

<!--
Copy this file to docs/dev/seps/sep-NNNN-short-slug.md, with the next free number from
the index, and replace every placeholder. Keep the headings: the parts and their order
are the same in every SEP. Delete the italic guidance as you write each section. The
process is SEP 0 (sep-0000-process.md); read it first.
Add the draft to the SEP index and its toctree, and to the roadmap's Draft enhancement
proposals section after the accepted proposals. Remove informal roadmap items only when
fully covered by this proposal; for partial coverage, link the SEP and retain the
uncovered work explicitly. Keep these entries current when the scope or status changes.
-->

| | |
|---|---|
| **SEP** | NNNN |
| **Title** | A short title, in sentence case |
| **Author** | The person among the maintainers who initiated the SEP and manages it |
| **Drafting agent** | — (AI agents that helped draft it, by model and version; add later ones with the date) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — (each completed review of Part I: `Name (human)` or `Model version (AI)`) |
| **Type** | Feature, Change or Process |
| **Created** | YYYY-MM-DD |
| **Resolution** | — (at acceptance: the date and the pull request or discussion) |
| **Requires** | — (SEPs this one builds on) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — (once picked up: the implementer, and each phase's branch and state) |
| **Manual pages** | — (at Final: the manual pages that document the functionality) |

## Part I — Proposal

*Part I is written and revised while the SEP is a Draft, and frozen when it is accepted.
It stands alone: cite solprop's own documentation and code, optmat's, the scientific
literature and public reports, never any other code's source, local files or private notes
(SEP 0 §7).*

### 1. Abstract

*One paragraph: what is proposed, for what use, and how success will be judged.*

### 2. Motivation and scope

*The concrete use case or problem, and who needs it. What is in scope and, explicitly,
what is not, including tempting adjacent work.*

### 3. Context

*What solprop does today in this area, with links to the manual and design pages and to
the modules involved; the roadmap item this answers, if any; the prior SEPs and ADRs it
builds on.*

### 4. Theory

*The physics and mathematics, derived or cited to primary sources: the equations, the
approximations and their range of validity, the reference states, the units (SI) and the
conventions of conventions.md, with any departure stated where it occurs. Every constant
or coefficient to be transcribed is cited here. Write maths with dollar notation, for
example $\partial_z E = \dots$.*

### 5. Proposed interface

*How a user would call it: the public names, their parameters with units and ranges, and
a short example sketch (fence it as `text` until it runs). How it composes with existing
routes, backends, scans and outputs, and what is refused.*

### 6. Implementation strategies

*Options, with their trade-offs, and the recommended one; this is not a line-by-line
plan. Cover as applicable:*

- *the layer the work belongs to (physics, translation, backends; architecture.md §3);*
- *the effect on the kernel contract and whether SCHEMA_VERSION changes;*
- *each backend in turn — numpy_ref, numba_cpu, then Kokkos CPU — and anything that makes
  one of them hard; GPU considerations for later;*
- *the expected cost in time and memory, and how it scales;*
- *new dependencies (to be agreed before acceptance).*

### 7. Verification and validation

*How it will be shown to be right: analytic solutions and limits, conservation laws and
symmetries, convergence orders, independent references (published results, a different
method, a different solver), field/envelope and backend parity. For each, the quantity
compared and what will set the tolerance. The benchmarks to record, on short windows.*

### 8. Documentation

*The manual page(s) to add or extend, the docstrings, and whether a quick tutorial or a
worked example belongs in a gallery.*

### 9. Compatibility and impact

*Changed defaults, breaking changes, the output schema, SCHEMA_VERSION, the effect on
existing validated results, and any deliberate difference from Luna.jl to record on the
differences page.*

### 10. Alternatives considered

*Other designs or models, and why they were not chosen.*

### 11. Phases

*The proposed phases, each independently mergeable with a green gate. The usual order is
numpy_ref with tests and the manual page first, then numba_cpu, then Kokkos CPU. For each
phase: its deliverables and what it must verify.*

### 12. Open questions

*Questions to settle before acceptance, or deferred to a named phase.*

### 13. References

*Numbered references, with DOIs where they exist.*

1. A. Author and B. Author, "Title," *Journal* **volume**, page (year).
   doi:10.xxxx/xxxxx

## Part II — Implementation record

*One report per phase, appended in the pull request that merges the phase (SEP 0 §8).
Copy the skeleton below for each phase.*

### Phase 1 report — short title

- **Scope:** phase, date, implementer, branch, merge commits (solprop, and solprop-kokkos
  when changed).
- **Delivered:** modules, tests and pages added or changed, as paths.
- **Deviations from the proposal:** each, with why it was necessary.
- **Decisions:** choices the proposal left open, with ADR numbers where one was written.
- **Verification:** gate and lanes run and on which tree; references, measured errors,
  orders and benchmarks with their windows; what was not run.
- **Sources checked:** constants and citations checked against their primary source, and
  those that could not be.
- **Limitations:** cases refused, ranges not qualified, backends still missing.
- **Open questions:** Q1, Q2, … (numbered through the whole SEP), each with its
  destination.
- **Handoff:** what the next phase needs to know.

## Part III — History

- YYYY-MM-DD: Drafted.
