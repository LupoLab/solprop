# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Copy as Python: the solprop calls that reproduce what a page shows (design 11.3)."""

__all__ = ["snippet"]

_OPEN = """\
from solprop.output import open_file
{imports}

with open_file({path!r}) as output:
{body}
"""

_SNIPPETS = {
    "overview": ("from solprop import plotting", "    plotting.prop_2d(output)"),
    "plane": (
        "from solprop import plotting",
        "    plotting.spec_1d(output, {z!r}, modes={modes!r})\n"
        "    plotting.time_1d(output, {z!r}, modes={modes!r})",
    ),
    "spectrogram": (
        "from solprop import plotting",
        "    plotting.spectrogram(output, {z!r}, modes={first!r})",
    ),
    "along": ("from solprop import plotting", "    plotting.plot_stats(output)"),
    "modes": (
        "from solprop.processing import get_spectral_density",
        '    energy = output["stats/energy"]  # one column per mode\n'
        "    for mode in range(energy.shape[1]):\n"
        '        x, density = get_spectral_density(output, "wavelength", '
        "z={z!r}, modes=mode)",
    ),
    "compress": (
        "from solprop.processing import compress_taylor",
        "    found = compress_taylor(output, plane={plane!r}, modes={first!r})",
    ),
    "health": (
        "import numpy as np",
        '    accepted = np.asarray(output["step_history/accepted"], dtype=bool)\n'
        '    z = np.asarray(output["step_history/z"])[accepted]\n'
        '    h = np.asarray(output["step_history/h"])[accepted]\n'
        "    print(output.outcome)",
    ),
}


def snippet(
    key: str,
    path: str,
    *,
    z: float | None = None,
    plane: int = -1,
    modes: object = "sum",
) -> str:
    """Return the script that reproduces a page.

    Parameters
    ----------
    key : str
        The page.
    path : str
        The results file.
    z : float or None, optional
        The plane's position, m.
    plane : int, optional
        The plane's index.
    modes : object, optional
        ``"sum"`` or a mode's index.

    Returns
    -------
    str
        The script.

    Raises
    ------
    KeyError
        If the page has none.

    Examples
    --------
    >>> print(snippet("along", "run.h5"))
    from solprop.output import open_file
    from solprop import plotting
    <BLANKLINE>
    with open_file('run.h5') as output:
        plotting.plot_stats(output)
    <BLANKLINE>
    """
    imports, body = _SNIPPETS[key]
    first = 0 if modes == "sum" else modes
    return _OPEN.format(
        imports=imports,
        path=path,
        body=body.format(z=z, plane=plane, modes=modes, first=first),
    )
