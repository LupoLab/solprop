# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What executed a run and which code it was: the backend, the git commit, the versions.

:class:`BackendRecord` is what the backend reported about itself, :class:`GitProvenance`
the commit of a development checkout with its dirty flag, and :func:`package_versions`
the installed versions. All three are outside the resume key
(:meth:`solprop.output.metadata.RunMetadata.hashed`): they say how a run was executed,
not what was computed. **Provenance, not source:** the commit and the versions are
recorded, never the source text.

The manual page is ``docs/guide/results/provenance.md``.
"""

import dataclasses
import functools
import platform
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import cast

from solprop.exceptions import OutputError
from solprop.modes.material import JsonValue

__all__ = [
    "BackendRecord",
    "GitProvenance",
    "package_versions",
]

# The subprocess calls that read the git provenance are bounded: a hung git must not
# stall a run that is otherwise ready to start.
_GIT_TIMEOUT = 5.0


def _run_git(arguments: Sequence[str], root: Path) -> str | None:
    """Return the output of one read-only git command, or ``None`` if it cannot run.

    - Returns: the stripped standard output, or ``None`` when git is absent, the
      directory is not a repository, or the command fails or times out.
    """
    try:
        finished = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.strip()


@dataclass(frozen=True)
class BackendRecord:
    """What executed a run, as the backend's own ``metadata()`` reported it.

    A saved run has to say what produced it and not only what was computed: the thread
    configuration alone moved one step by a factor of nine
    (``docs/dev/backends.md`` section 3.4), and without this record a timing
    comparison has to read the realised kernel pool out of the backend itself.

    It is **outside the resume key**, like the package versions and the git provenance:
    a run continued on the other backend agrees with an uninterrupted one to
    2.2-2.7e-16 and is the same run. What it records is how,
    not what.

    A resumed file keeps the record it was opened with, creation time and all, so the
    backend it names is the one that **started** the run. That is append mode's rule
    for the whole record and not a special case for this group. The name may be one this
    package no longer has, such as ``numba_cpu``: a recorded name is not a request, so
    an old file's record reads as it was written, in its backend's own vocabulary.

    Attributes
    ----------
    name : str
        The registry name of the backend, for example ``"numpy_ref"`` or ``"native"``.
    propagator_mode : str
        Which the backend resolved the contract's ``"auto"`` to, or which was asked
        for: ``"tabulate"`` or ``"recompute"``.
    settings : dict
        Everything else the backend reported, for example its thread counts, its plan
        flag and its wisdom digest. Free-form because
        it is a backend's own vocabulary, and JSON-serialisable because it is written
        to the file.

    Examples
    --------
    >>> from solprop.output.provenance import BackendRecord
    >>> record = BackendRecord.from_metadata(
    ...     {"backend": "native", "propagator_mode": "recompute",
    ...      "thread_count": 4}
    ... )
    >>> record.name, record.propagator_mode, record.settings
    ('native', 'recompute', {'thread_count': 4})
    """

    name: str
    propagator_mode: str
    settings: Mapping[str, JsonValue] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Adopt a plain copy of the free-form settings."""
        object.__setattr__(self, "settings", dict(self.settings))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "BackendRecord":
        """Return the record a backend's ``metadata()`` describes.

        Parameters
        ----------
        metadata : collections.abc.Mapping
            What :meth:`solprop.backends.protocol.Backend.metadata` returned.
            ``backend`` and ``propagator_mode`` are named fields here; everything else
            becomes `settings`.

        Returns
        -------
        BackendRecord
            The record.

        Raises
        ------
        OutputError
            If either named field is missing.
        """
        try:
            name = str(metadata["backend"])
            mode = str(metadata["propagator_mode"])
        except KeyError as error:
            msg = (
                "a backend's metadata must name the backend and the propagator mode; "
                f"this one is missing {error}"
            )
            raise OutputError(msg) from error
        rest = {
            key: cast("JsonValue", value)
            for key, value in metadata.items()
            if key not in {"backend", "propagator_mode"}
        }
        return cls(name=name, propagator_mode=mode, settings=rest)


@dataclass(frozen=True)
class GitProvenance:
    """Which commit of solprop produced a file, and whether it was modified.

    Attributes
    ----------
    commit : str
        Full 40-character hexadecimal commit hash of the running checkout.
    branch : str
        Branch name, or ``'HEAD'`` when the checkout is detached.
    dirty : bool
        Whether the working tree had uncommitted changes to tracked files. A dirty
        run is reproducible only up to those changes, which is the point of recording
        it. It is a separate field rather than a suffix on the hash string, so the
        commit stays parsable.

    Examples
    --------
    >>> from solprop.output.provenance import GitProvenance
    >>> record = GitProvenance(commit="0" * 40, branch="main", dirty=False)
    >>> len(record.commit), record.dirty
    (40, False)
    """

    commit: str
    branch: str
    dirty: bool

    @classmethod
    def detect(cls) -> "GitProvenance | None":
        """Return the provenance of the running solprop checkout, if there is one.

        An installed release is not a checkout, git may be absent, and the package may
        sit inside an unrelated repository's worktree -- a virtual environment inside a
        user's own project, whose ``git rev-parse`` answers with *that* project's
        commit; every one of those returns ``None`` rather than a placeholder string,
        so that a caller can tell "no provenance" from "this commit". The test that
        separates the last case from a development checkout is whether the repository
        found actually tracks this file.

        Returns
        -------
        GitProvenance or None
            The commit, branch and dirty flag, or ``None`` when they cannot be read.

        Examples
        --------
        >>> from solprop.output.provenance import GitProvenance
        >>> record = GitProvenance.detect()
        >>> record is None or len(record.commit) == 40
        True
        """
        here = Path(__file__).resolve()
        root = here.parent
        # A repository was found, but is it solprop's? Inside a virtual environment that
        # lives in some other project's checkout, git answers for that project. Only a
        # repository that tracks this very file is the one to record; that cannot
        # change while the process runs, so it is asked once.
        if not _tracks(here):
            return None
        # One command gives the commit, the branch and the dirty flag: the per-run
        # cost of provenance is a single git process (SEP 25 section 6.3).
        status = _run_git(
            ["status", "--porcelain=v2", "--branch", "--untracked-files=no"], root
        )
        if status is None:
            return None
        headers = dict(
            line[2:].split(" ", 1)
            for line in status.splitlines()
            if line.startswith("# ") and " " in line[2:]
        )
        commit = headers.get("branch.oid", "")
        if len(commit) != 40:
            return None
        branch = headers.get("branch.head", "")
        dirty = any(line and not line.startswith("#") for line in status.splitlines())
        return cls(
            commit=commit,
            branch="HEAD" if branch in {"", "(detached)"} else branch,
            dirty=dirty,
        )


@functools.cache
def _tracks(path: Path) -> bool:
    """Return whether the git repository around `path` tracks it, asked once.

    - Returns: ``True`` when git runs, finds a repository and that repository tracks the
      file.
    """
    return _run_git(["ls-files", "--error-unmatch", path.name], path.parent) is not None


def _version_of(name: str) -> str:
    """Return an installed distribution's version, or ``'unknown'``.

    - Returns: the version string, or ``'unknown'`` when the distribution is not
      installed.
    """
    try:
        return _package_version(name)
    except PackageNotFoundError:  # pragma: no cover - every dependency is declared
        return "unknown"


def package_versions() -> dict[str, str]:
    """Return the versions of solprop and of the packages a stored file depends on.

    Recorded so that a file can be read years later with the same numerics in view. The
    versions are deliberately *outside* the configuration hash: upgrading NumPy must not
    refuse to resume a run. optmat's version is here too, for a reader; a run that takes
    a material from optmat carries it again in its material manifest
    (:attr:`~solprop.output.metadata.RunMetadata.materials`), where it is part of the
    key.

    Returns
    -------
    dict
        Version strings keyed by ``'solprop'``, ``'numpy'``, ``'scipy'``, ``'h5py'``,
        ``'optmat'`` (its full version string) and ``'python'``.

    Examples
    --------
    >>> from solprop.output.provenance import package_versions
    >>> sorted(package_versions())
    ['h5py', 'numpy', 'optmat', 'python', 'scipy', 'solprop']
    """
    return dict(_installed_versions())


@functools.cache
def _installed_versions() -> dict[str, str]:
    """Return :func:`package_versions`, read once per process.

    Every record asks, so every scan point did; what is installed does not change
    under a running process.

    - Returns: the versions; callers copy it.
    """
    return {
        "solprop": _version_of("solprop"),
        "numpy": _version_of("numpy"),
        "scipy": _version_of("scipy"),
        "h5py": _version_of("h5py"),
        "optmat": _version_of("optmat"),
        "python": platform.python_version(),
    }
