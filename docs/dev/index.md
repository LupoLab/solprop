# Developer’s guide

This guide is for contributors working on solprop's code, numerical methods,
backends, tests, and documentation. For using solprop in a simulation, start with
the [introduction](../introduction/index.md) and [user guide](../guide/index.md).

Read [contributing](contributing.md), then [architecture](architecture.md) and
[conventions](conventions.md). The architecture describes the current code;
[enhancement proposals](seps/index.md) and [decisions](decisions.md) record how
substantial changes were proposed and assessed.

```{toctree}
:maxdepth: 1
:caption: Contributing and design

contributing
architecture
structure
conventions
dependencies
tooling
releasing
```

```{toctree}
:maxdepth: 1
:caption: Implementation and contracts

stepper
algorithms
backends
scans
kernel-problem
kernel-contract
```

```{toctree}
:maxdepth: 1
:caption: Testing and qualification

testing
accuracy
benchmarks
benchmarking
cpu-performance
```

```{toctree}
:maxdepth: 1
:caption: Decisions, proposals, and ongoing work

decisions
seps/index
roadmap
open-questions
```
