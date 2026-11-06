# solprop-gui

`solprop-gui` is a desktop front end for [solprop](https://solprop.readthedocs.io/en/latest/),
a Python package for ultrafast nonlinear pulse propagation. It is meant to let someone set
up, run and understand a solprop simulation without writing Python, and an expert do the
same faster than a script would. It never becomes a dead end: every setup made in it
leaves as a solprop recipe file or a plain Python script, and every results file solprop
writes can be opened in it.

**It is in development: the window opens, but nothing can be set up in it yet.**

The distribution is `solprop-gui` and the import package `solprop_gui`. It is built from
the `gui/` directory of the [solprop repository](https://github.com/LupoLab/solprop) and
released at solprop's own version, which it requires exactly.

## Installing

```sh
pip install "solprop[gui]"
solprop-gui
```

It uses Qt for Python ([PySide6](https://doc.qt.io/qtforpython-6/)) and
[pyqtgraph](https://www.pyqtgraph.org/). solprop does all the physics: the window runs it in
separate processes and shows what it returns.

## Design

The design is [`docs/dev/gui/design.md`](../docs/dev/gui/design.md) in the solprop
repository; the rules for working on this package are in [`AGENTS.md`](AGENTS.md).

## Licence

GPL-3.0-or-later, as solprop.
