# Node workspace planning

The Julia node planner budgets retained numerical array data whose capacity grows
with `node_chunk` or the attached transverse `trace_chunk`. The budget is an
execution choice, outside scientific and restart identities. An explicit chunk
wins over any budget and is capped to the physical node count. Otherwise the
shared route default or explicit byte budget admits a limit, then the same number
of passes is balanced with `ceil(nodes / ceil(nodes / limit))`. A budget smaller
than one node's complete listed storage refuses before Julia starts.

This is not an RSS, total-session or FFTW-native-allocation bound. Fixed and
full-volume numerical work are reported separately. Immutable prepared tables,
method/state/history/input-noise/export storage, object and view headers, chunk
descriptor containers, allocator overhead, FFTW plan internals and temporary
planning copies are outside the node budget. FFTW plans remain named opaque
exclusions; neither `Base.summarysize(plan)` nor a column-equivalent multiplier
proves their native allocation size. No empirical multiplier is promoted to a bound.

The fixed/full-volume inventory covers source and observation numerical scratch,
the packed plane values and quadrature diagnostic. Bounded transport rows and
scalar notice/continuation records are excluded: their capacity is a separate
session transport choice. The inventory does not claim to enumerate a complete
session. An attached monitor is charged regardless of its current quiet state,
and its optional fresh-noise/resume readings receive their maximum retained size.

## Counting the source

Let `T` be fine time samples, `N` coarse time samples, `F` coarse spectral bins,
and `Q` fine spectral bins
(`T/2+1` for fields, `T` for envelopes), `M` retained spectral columns, `X` physical
nodes and `K` the selected chunk. Let `r`, `c`, `i` and `b` be the real, complex,
index and status element widths. Binary64 uses 8, 16, 8 and 1 bytes; binary32
uses 4, 8, 8 and 1. Host physical readings and coefficient work retain width `h=8`
(and `2*h` for complex coefficients). Let `s=r` for fields and `s=c` for envelopes.
Counts below are array payload
sizes; a repeated reference or view does not create another data array.

- Modal/radial source nodes and polarisation cost `2*T*s*K`. The failure vector
  costs `i*max(K,M)`, bounded by a separate `i*M` full-volume contribution plus
  `i*K`. Coarse projection adds only full-volume work.
- Cartesian/line source nodes and polarisation cost `2*T*s*K`; its fine spectrum
  and coarse gathered spectrum add `(Q+F)*c*K`. Its failure vector is `i*X`.
- A field source using analytic Kerr or any non-THG Raman component adds `T*c*K`
  for modal/radial and `3*T*c*K` for Cartesian/line. The modal analytic synthesis
  and spectrum are full-volume arrays, whereas their Cartesian counterparts are
  chunk arrays.
- Any number of Raman components share one doubled real drive and one spectral
  work array: `(2*T*r+(T+1)*c)*K`. Each component separately retains its doubled
  time kernel, spectrum and two per-resonance vectors; those are fixed arrays.
- Field plasma adds `(7*r+b)*T*K` for rates, exposure, fraction, survival,
  integrand, current, FIR work and status. Envelope plasma adds
  `(4*r+c+b)*T*K`. Both add two source columns `2*T*s*K`. Cartesian/line also
  holds two projected columns of that same chunk size; modal/radial projected
  and optional coarse-projected columns are full-volume. Channels reuse these
  workspaces; only the small channel-parameter array grows with channel count.

These counts follow `ModalSource`, `CartesianSource`, `RadialSource`,
`_source_analytic`, `_source_raman` and `_source_plasma`. Temporal FFT wrappers
retain plans and scalar sizes, not a second owned numerical buffer pair.

Let `C` be one when the source has a distinct embedded coarse projection and zero
otherwise. The modal base has full-volume payload
`(3+C)*T*M*s + Q*M*c + (1+2*C)*F*M*c + i*M`: modal samples and projections,
fine spectrum, prefactor/optional quadrature spectra, and the conservative fixed
portion of the failure vector. A radial source additionally holds `F*M*c+i*M`
for independent loss work. A Cartesian source has `F*X*c+i*X` for its band and
failures. Every source holds two padding maps, `i*(F+Q)` bytes. An expanded modal
prefactor adds one host binary64 vector of the represented expansion's term count.

Analytic modal work adds `2*T*M*c` outside the chunk. Each Raman component retains
its host kernel, host spectrum and profile work, `(2*T+2*n_terms)*h+(T+1)*2*h`.
Binary64 aliases the convolution spectrum to its host spectrum; binary32 adds
an independent `(T+1)*c` narrowed spectrum. Plasma adds two complex spectral
factor vectors in state precision and two host binary64 parameters per channel;
modal/radial plasma also adds
`(2+2*C)*T*M*s` in projected columns. These allocations occur once per source,
not once per pass or physical node.

## Observations and transverse services

Population statistics recompute populations in independently owned scratch. Modal
and radial observations add `T*s*K`; Cartesian/line adds
`(T*s+(Q+F)*c)*K`. If a population channel exists, add `(5*r+b)*T*K` for the
population-only recurrence. Species densities and complete node-by-species
fraction/peak arrays are fixed/full-volume; their size does not decrease with
chunking. Analytic observations, spatial width/axis reconstruction and temporal
monitor/split buffers are also fixed or full-volume.

All observation workspaces retain a coarse spectrum, `F*M*c`, a statistics power
matrix, `N*M*r`, two per-column reduction vectors, `2*M*r`, and a total-power
vector, `N*r`. Three complete per-node real vectors are present for spatial
observations and for modal population statistics. Selected packed statistic
values and the plane copy cost `2*width*h`; the quadrature alarm adds `5*M*r`.
Binary32 additionally retains `i*M` column exponents and `i*X` node exponents
where node statistics are present. These are independent owned arrays, not scalars.

Modal observations always retain `F*M*r` power weights and, for an expanded
weight, one host binary64 term vector. Analytic samples plus spectra, both ordinary
and
power weighted, cost `4*N*M*c`; each requested window repeats that pair. The
coarse on-axis peak uses two complex matrices with at most two physical components,
bounded by `4*N*c` even though the propagated modal polarisation is scalar.
Population species densities cost `n_species*h`, and fractions/peaks cost
`2*X*n_species*r`.
Their fine modal reconstruction costs `(Q*c+T*s)*M`; the Cartesian version uses
the already-counted chunk spectra and a full `F*X*c` spatial band. Each owns its
two padding maps independently of the source.

The four resolved-modal on-axis population rows own another independent fine
reconstruction: `(Q*c+T*s)*M`, at most `2*T*s` component samples, one `T*s` scalar
column, five `T*r` recurrence columns, `T` status bytes, one `n_species*h` density
vector and two `n_species*r` result vectors,
and `i*(F+Q)` padding indices. This storage is independent of the node chunk.
The original axis map and represented species profiles are immutable preparation;
FFT plans retain the stated opaque-storage exclusion. Envelope collinearity can
reduce the component allocation, but the planner conservatively charges both rows.

Spatial observations retain `F*M*r` weights only for spectral statistics and
`F*M*c` closure values only when physical or axis reconstruction is requested.
A material-index vector costs `F*2*h` when present; it is host coefficient work.
Physical spectra/closures cost
`2*F*X*c`, plus `F*X*c` for the Cartesian band; analytic intensity adds
`4*N*X*c`. The axis pair costs `2*F*c`, plus `4*N*c` for axis intensity. The
radial-average scratch is bounded by `X*r`: its half-pixel lattice covers the
smaller physical axis, so it cannot exceed that axis's sample count or the total
number of physical nodes. Prepared width-grid angles and weights are immutable
preparation, outside this workspace inventory.

A temporal monitor holds `F*M*c+N*M*s+N*r` plus two host binary64 occupancy
vectors of the complete shared split-rate bucket count, including empty buckets.
An attached split collar holds `N*h+N*M*s+F*M*c`; binary32 adds its `N*r` damping
corrections, evaluated from the wide rate/step with `expm1` before one narrowing.
The Cartesian split adds its independent `F*M*c` band.

Selected field-plasma diagnostics retain `6*K*r+6*h`. Binary32 adds `3*K*i`
exponents; its physical aggregate maxima remain binary64.

Every transverse line owns one complex twice-padded strip spectrum and, when an
onset stencil exists, three complex current rows. Its contribution is
`c*(2*strip_length+3*has_onset)*min(K,n_traces)`, conservatively bounded by the
same coefficient times `K`. Sum this over every line, since all workspaces coexist.
The spatial band, previous/noise column readings and complete node vector remain
full-volume. Optional resumed or noise data must not reduce the prepared bound.

The transverse fixed/full-volume bound is `F*X*c+X*r+5*M*h+7*n_lines*h`.
The five per-column vectors conservatively cover previous/noise state, the live
reading, an optional noise reading and optional clean columns. Each global
reading has two strip columns per line; each line also owns a one-column,
one-line reading, adding three real elements. Plans and their work are not
multiplied by a guessed FFT allocation factor.

## Qualification

The focused planner suite has 23 pure tests and 38 managed-runtime cases. Each
managed case constructs actual independent Julia sessions at chunks one and
three, reads the public retained-owner ledger and closes them without advancing.
It checks both the positive payload increment against `2*per_node_bytes` and
each complete listed workspace payload against the fixed/full-volume sum plus
the selected chunk contribution. Chunk-descriptor arrays are explicitly removed
from the numerical payload: their shrinking count must not hide a missing growing
numerical buffer. FFT plan estimates are excluded independently by ledger name.

The 24 material cases span field/envelope, modal/radial/Cartesian/line, and Kerr,
two separately labelled Raman components, or Kerr/Raman/plasma together. The
mixed cases attach independent population statistics; coarse analytic or spatial
intensity/width/axis statistics and the quadrature alarm are also present. The
12 transverse cases cover all three spatial geometries and both formulations,
with or without a collar. Resolved collars have nonempty padded FFT strips and
onset current work, plus optional input-noise and clean readings. Two additional
modal cases combine window and on-axis statistics, a distinct
embedded coarse projection, temporal monitoring and split damping. The window
views retain their shared data owner, so the public ledger counts each physical
array once rather than counting an aliased reshape again.

The existing managed checks qualify retained CPU binary64 array payloads. Pure
binary32 checks distinguish narrow arrays from host coefficient/readout storage and
integer scaling metadata. Complete binary32 retained-owner comparisons await the
observation reconstruction attachment; the planner alone does not admit public
binary32 execution. These checks neither measure RSS nor qualify vendor plans.
