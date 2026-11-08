# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Setting a machine up: the script, and running it with ``sh`` in place of ``ssh``."""

import os
import sys
from pathlib import Path

import pytest

from solprop.service import install
from solprop.service.install import (
    DEFAULT_ROOT,
    PYTHON_LINE,
    copy_wheels,
    run_setup,
    setup_script,
    this_version,
)
from solprop.service.machines import Machine
from solprop.service.messages import ServiceError

MACHINE = Machine("lab", "me@lab")


def local_shell(machine: Machine, askpass: str) -> tuple[list[str], dict[str, str]]:
    """Run the script with this computer's shell, as ssh runs it on the machine."""
    del machine, askpass
    return ["sh", "-s"], dict(os.environ)


@pytest.fixture
def posix() -> None:
    if sys.platform == "win32":
        pytest.skip("a machine's setup script runs in a POSIX shell")


def test_the_script_says_where_and_what(tmp_path: Path):
    home = setup_script("0.3.0")
    assert DEFAULT_ROOT == "~/.solprop/envs"
    assert 'env="$HOME"/.solprop/envs/0.3.0' in home
    assert '"solprop==0.3.0"' in home and "--find-links" not in home
    assert home.rstrip().endswith(f'echo "{PYTHON_LINE}$env/bin/python"')
    elsewhere = setup_script("0.3.0", "/scratch/my envs/", wheels="/scratch/w")
    assert "env='/scratch/my envs'/0.3.0" in elsewhere
    assert "--find-links /scratch/w" in elsewhere
    assert 'env="$HOME"/0.3.0' in setup_script("0.3.0", "~")
    assert this_version()


def test_a_script_that_names_its_python_sets_the_machine_up(posix: None):
    script = f'echo installing\necho "{PYTHON_LINE}/env/bin/python"\n'
    lines: list[str] = []
    python = run_setup(MACHINE, script, lines.append, command=local_shell)
    assert python == "/env/bin/python" and lines[0] == "installing"


def test_a_script_that_fails_says_its_last_lines(posix: None):
    script = "echo one\necho two\nexit 3\n"
    with pytest.raises(ServiceError, match=r"setting lab up failed:\none\ntwo"):
        run_setup(MACHINE, script, lambda line: None, command=local_shell)
    with pytest.raises(ServiceError, match="status 0"):
        run_setup(MACHINE, "", lambda line: None, command=local_shell)


def test_the_real_script_runs_up_to_its_install(
    posix: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A uv that only says what it was asked: the script's own logic, no network.
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    uv = bin_ / "uv"
    uv.write_text(
        '#!/bin/sh\necho "uv $*"\n'
        'if [ "$1" = venv ]; then mkdir -p "$5/bin"; '
        "printf '#!/bin/sh\\necho 0.3.0\\n' > \"$5/bin/python\"; "
        'chmod +x "$5/bin/python"; fi\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    lines: list[str] = []
    script = setup_script("0.3.0", str(tmp_path / "envs"), wheels="/w")
    python = run_setup(MACHINE, script, lines.append, command=local_shell)
    assert python == f"{tmp_path}/envs/0.3.0/bin/python"
    assert any("pip install" in line and "--find-links /w" in line for line in lines)


def test_wheels_are_copied_through_ssh(
    posix: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    wheel = tmp_path / "solprop-0.3.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    target = tmp_path / "there"

    def shell(machine: Machine, remote: str, *, askpass: str | None = None):
        del machine, askpass
        return ["sh", "-c", remote], {**os.environ, "HOME": str(tmp_path)}

    monkeypatch.setattr(install, "ssh_command", shell)
    copy_wheels(MACHINE, [wheel], str(target))
    assert (target / wheel.name).read_bytes() == b"wheel"
    copy_wheels(MACHINE, [wheel], "~/home-wheels")
    assert (tmp_path / "home-wheels" / wheel.name).exists()

    def refusing(machine: Machine, remote: str, *, askpass: str | None = None):
        del machine, remote, askpass
        return ["sh", "-c", "echo 'no space' >&2; exit 1"], dict(os.environ)

    monkeypatch.setattr(install, "ssh_command", refusing)
    with pytest.raises(ServiceError, match=r"copying .* failed: no space"):
        copy_wheels(MACHINE, [wheel], str(target))


def test_the_setup_runs_a_shell_over_ssh():
    command, _ = install._ssh_shell(MACHINE, "")
    assert command[0] == "ssh" and command[-2:] == ["me@lab", "sh -s"]
