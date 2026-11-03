# solprop-gui

`solprop-gui` is a desktop front end for [solprop](https://solprop.readthedocs.io/en/latest/),
a Python package for ultrafast nonlinear pulse propagation. It is meant to let someone set
up, run and understand a solprop simulation without writing Python, and an expert do the
same faster than a script would. It never becomes a dead end: every setup made in it
leaves as a solprop recipe file or a plain Python script, and every results file solprop
writes can be opened in it.

**It is in development**, and not yet released. It covers solid-core fibres (GNLSE),
gas-filled capillaries (mode-averaged and multimode) and antiresonant fibres, and beams in
free space (Cartesian, line and radial). For each it can set up and check a run, show it
live, browse its results (the Overview along z, a plane with its spectrogram, the
statistics), and compress the pulse after it (Post-process). It also has the waveguide
explorer, scans, and a service that lets runs outlive the window
(`SOLPROP_GUI_SERVICE=1`).

The distribution is `solprop-gui` and the import package `solprop_gui`. It is built from
the `gui/` directory of the [solprop repository](https://github.com/LupoLab/solprop) and
released at solprop's own version, which it requires exactly.

## Installing

Once released, it will install with solprop:

```sh
pip install "solprop[gui]"
solprop-gui
```

Until then, run it from a checkout of the solprop repository:

```sh
uv sync --locked --no-install-package solprop-native --extra gui
uv run --no-sync poe fetch-wheel      # CI's native wheel: needs `gh`, no compiler
uv run --no-sync solprop-gui
```

`uv sync --extra gui` instead builds the native backend from source, which needs a C++20
compiler (`native/README.md`).

It uses Qt for Python ([PySide6](https://doc.qt.io/qtforpython-6/)) and
[pyqtgraph](https://www.pyqtgraph.org/). solprop does all the physics: the window runs it in
separate processes and shows what it returns.

## Design

The design is [`docs/dev/gui/design.md`](../docs/dev/gui/design.md) in the solprop
repository; the rules for working on this package are in [`AGENTS.md`](AGENTS.md).

## Licence

GPL-3.0-or-later, as solprop.
