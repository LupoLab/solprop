# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Set a machine up: solprop, at the window's version, in an environment of its own.

The window shows the user the script before anything runs, and runs it only when
asked (design 13.4). The script installs ``uv`` in the user's home folder if the
machine has none, makes an environment for this version under a folder the user may
choose (a home folder with a small quota is not the only place), and installs solprop
there: from PyPI, or from wheels the window copies over first, for a development build
PyPI does not have or a machine that cannot reach PyPI. Its last line names the
environment's Python, which the window then remembers for the machine.

Nothing here needs root, and nothing outside the chosen folder and ``uv``'s own is
touched.
"""

import contextlib
import shlex
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from solprop.service.machines import Machine, ssh_command
from solprop.service.messages import ServiceError

__all__ = [
    "DEFAULT_ROOT",
    "PYTHON_LINE",
    "copy_wheels",
    "run_setup",
    "setup_script",
    "this_version",
]

DEFAULT_ROOT: Final = "~/.solprop/envs"
"""Where environments go on a machine unless the user picks another folder."""

PYTHON_LINE: Final = "SOLPROP_PYTHON="
"""How the script's last line starts: the environment's Python follows."""

PYTHON_VERSION: Final = "3.14"
"""The Python the environment is made with: solprop's ``requires-python``."""


def this_version() -> str:
    """Return this process's solprop version: what a machine is set up with.

    Returns
    -------
    str
        The version.

    Examples
    --------
    >>> import solprop
    >>> this_version() == solprop.__version__
    True
    """
    from solprop import __version__  # noqa: PLC0415 - the package's own

    return __version__


def setup_script(
    version: str, root: str = DEFAULT_ROOT, *, wheels: str | None = None
) -> str:
    """Return the shell script that sets a machine up.

    Parameters
    ----------
    version : str
        The solprop version to install, the window's own.
    root : str, optional
        The folder environments go in, on the machine (``~`` is its home).
    wheels : str or None, optional
        A folder on the machine holding wheels to install from first (copied there
        by :func:`copy_wheels`); ``None`` installs from PyPI alone.

    Returns
    -------
    str
        The script, for a POSIX shell.

    Examples
    --------
    >>> script = setup_script("0.2.0", "/scratch/envs")
    >>> 'pip install --python "$env/bin/python" "solprop==0.2.0"' in script
    True
    """
    base = root.rstrip("/") or "~"
    if base == "~" or base.startswith("~/"):
        folder = '"$HOME"' + shlex.quote(base[1:]) if base != "~" else '"$HOME"'
    else:
        folder = shlex.quote(base)
    links = f" --find-links {shlex.quote(wheels)}" if wheels else ""
    return f"""\
set -eu
env={folder}/{shlex.quote(version)}
uv=$(command -v uv || true)
if [ -z "$uv" ] && [ -x "$HOME/.local/bin/uv" ]; then uv="$HOME/.local/bin/uv"; fi
if [ -z "$uv" ]; then
  echo "Installing uv in $HOME/.local/bin"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  else
    wget -qO- https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  fi
  uv="$HOME/.local/bin/uv"
fi
echo "Making $env"
mkdir -p "$env"
"$uv" venv --allow-existing --python {PYTHON_VERSION} "$env"
echo "Installing solprop {version}"
"$uv" pip install --python "$env/bin/python"{links} "solprop=={version}"
"$env/bin/python" -c "import solprop, solprop_native; print(solprop.__version__)"
echo "{PYTHON_LINE}$env/bin/python"
"""


def copy_wheels(
    machine: Machine,
    wheels: Sequence[Path],
    folder: str,
    *,
    askpass: str | None = None,
) -> None:
    """Copy wheels to a folder on a machine, through its SSH connection.

    Parameters
    ----------
    machine : solprop.service.machines.Machine
        The machine.
    wheels : Sequence of pathlib.Path
        The wheels, on this computer.
    folder : str
        Where they go on the machine.
    askpass : str or None, optional
        A program ``ssh`` asks a password or code of.

    Raises
    ------
    solprop.service.messages.ServiceError
        If a copy fails, with what ``ssh`` said.
    """
    for wheel in wheels:
        target = f"{folder.rstrip('/')}/{wheel.name}"
        remote = (
            f"mkdir -p {shlex.quote(folder)} && cat > {shlex.quote(target)}"
            if not folder.startswith("~")
            else f'mkdir -p "$HOME"{shlex.quote(folder[1:])} && '
            f'cat > "$HOME"{shlex.quote(target[1:])}'
        )
        command, environment = ssh_command(machine, remote, askpass=askpass)
        with wheel.open("rb") as stream:
            result = subprocess.run(
                command, stdin=stream, capture_output=True, env=environment, check=False
            )
        if result.returncode != 0:
            said = result.stderr.decode("utf-8", "replace").strip()
            msg = f"copying {wheel.name} to {machine.name} failed: {said}"
            raise ServiceError(msg)


def run_setup(
    machine: Machine,
    script: str,
    line: Callable[[str], None],
    *,
    askpass: str | None = None,
    command: Callable[[Machine, str], tuple[list[str], dict[str, str]]] | None = None,
) -> str:
    """Run a setup script on a machine, line by line, and return its Python.

    Parameters
    ----------
    machine : solprop.service.machines.Machine
        The machine.
    script : str
        The script (:func:`setup_script`), as the user saw it.
    line : Callable
        Called with each line the script writes, as it comes, from this thread.
    askpass : str or None, optional
        A program ``ssh`` asks a password or code of.
    command : Callable or None, optional
        What gives the command running a shell on the machine; ``ssh`` by default.

    Returns
    -------
    str
        The environment's Python, as the script's last line names it.

    Raises
    ------
    solprop.service.messages.ServiceError
        If the script fails, with its last lines.
    """
    run, environment = (command or _ssh_shell)(machine, askpass or "")
    process = subprocess.Popen(
        run,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        text=True,
    )
    feeder = threading.Thread(target=_feed, args=(process, script), daemon=True)
    feeder.start()
    said: list[str] = []
    assert process.stdout is not None  # a pipe, as asked
    for text in process.stdout:
        said.append(text.rstrip("\n"))
        line(said[-1])
    process.stdout.close()
    status = process.wait()
    feeder.join(timeout=5)
    python = next(
        (
            text[len(PYTHON_LINE) :]
            for text in reversed(said)
            if text.startswith(PYTHON_LINE)
        ),
        None,
    )
    if status != 0 or python is None:
        tail = "\n".join(said[-5:]) or f"status {status}"
        msg = f"setting {machine.name} up failed:\n{tail}"
        raise ServiceError(msg)
    return python


def _ssh_shell(machine: Machine, askpass: str) -> tuple[list[str], dict[str, str]]:
    """Return the command that runs a shell reading its script from its input.

    - ``askpass``: a program ``ssh`` asks a password or code of; empty for none.
    - Returns: the command and its environment.
    """
    return ssh_command(machine, "sh -s", askpass=askpass or None)


def _feed(process: subprocess.Popen[str], script: str) -> None:
    """Write a script to a process's input and close it."""
    assert process.stdin is not None  # a pipe, as asked
    with contextlib.suppress(OSError):
        process.stdin.write(script)
    with contextlib.suppress(OSError):
        process.stdin.close()
