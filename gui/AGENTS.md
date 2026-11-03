# AGENTS.md

Contributor guidelines for `solprop-gui`, the desktop front end for solprop, in `gui/`.
They apply to people and to coding agents alike, beside the repository's
[`AGENTS.md`](../AGENTS.md), which governs everything outside `gui/` and whatever this
file does not say. The design is [`docs/dev/gui/design.md`](../docs/dev/gui/design.md);
tested code and configuration are authoritative, and a document that disagrees with
them is corrected.

## Product and architecture

- Distribution `solprop-gui`; import package `solprop_gui`; command `solprop-gui`. It is
  the third distribution of the solprop repository, built from `gui/`: one checkout,
  one uv workspace and one `uv.lock`. All three distributions take one version from the
  repository's tags; solprop's `gui` extra requires this package at that version, and
  this package requires solprop at it (`hatch_build.py`).
- **solprop does the physics; the front end does none.** Every number the window shows
  comes from a public solprop call. It holds no dispersion formulas, compatibility
  rules or defaults of its own: where it needs one that solprop does not expose, the fix
  is a change to solprop, in this repository, not a copy here.
- **Public solprop only.** Nothing here imports a private solprop name (a module or name
  starting with `_`); `tests/test_layers.py` enforces it.
- **The layers.** `model` (the editable recipe), `registry` (presentation over solprop's
  metadata), `service` (the messages, the kernel process, one worker process per run,
  transports) and `help` import no Qt. `ui` and `views` build the window with PySide6
  and pyqtgraph.
- **The window's process never imports simulation code.** It may import from solprop
  only `solprop.recipe.quantity`, `paths`, `document` and `errors`; metadata,
  capabilities, checks, previews and runs come from the kernel and the workers as
  plain data. Only `service` may import the rest of solprop, and only its child
  processes run it. `tests/test_layers.py` enforces both rules, statically and by what
  a fresh interpreter loads.
- **Never block the window.** Anything that can take more than about 50 ms happens in
  another process.
- **Two transports, one interface** (design 13.3). By default a kernel child and a worker
  child per run end with the window (option B); with `SOLPROP_GUI_SERVICE=1` the
  window speaks to this user's solprop service (`service/server.py`, option C) over a
  socket only the user can open, whose address and token are in
  `~/.solprop-gui/service.json`; the service owns the kernel and the runs and ends
  itself when idle.

## Commands

| Task | Command (from the repository root) |
|---|---|
| Sync | `uv sync --extra gui --group gui-test` (add `--extra native` to keep the native backend: a sync removes any extra it does not name) |
| Run the window | `uv run solprop-gui` |
| Run it with the solprop service (runs outlive the window, a new window re-attaches) | `SOLPROP_GUI_SERVICE=1 uv run solprop-gui` |
| The front end's gate | `uv run poe check-gui`: types (`gui-types`), docstrings (`gui-docstrings`) and tests (`gui-test`, offscreen, with coverage) |
| Lint, format, spelling, licences | the repository's: `uv run ruff check .`, `uv run ruff format .`, `uv run codespell`, `uv run poe reuse` |

The tests run without a display (`QT_QPA_PLATFORM=offscreen`, set by `tests/conftest.py`
and the `gui-test` task); set it to your platform's own value to watch a test.

## Definition of done

The repository's rules apply, with one difference in the coverage bar:

- `model`, `registry`, `service` and `help` meet solprop's standard in full: annotated,
  NumPy-style docstrings validated by numpydoc, tested to **100 %** of statements and
  branches (`[tool.coverage]` in `pyproject.toml`).
- `ui`, `views` and the entry points are typed, documented and linted the same way, but
  are left out of the coverage measure. Each page, dock and view has a `pytest-qt`
  smoke test, and the pages an offscreen screenshot test.
- A change that needs something new from solprop changes solprop in the same branch or
  first, through its own rules (an issue, or a SEP for a major change).

CI runs the gate as the `gui` job, on Linux, macOS and Windows, whenever `gui/` or
solprop's own code changes.
