# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Identify reference receipts and evaluate the declared worst-plane field metric."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.records import (
    Record,
    SuiteError,
    file_hash,
    identity,
    read_record,
)

__all__ = [
    "ADVANCED_ENVELOPES",
    "CORE_ENVELOPES",
    "EXERCISE_ENVELOPES",
    "MODAL_ENVELOPES",
    "AccuracyEnvelope",
    "ReferenceReceipt",
    "approved_envelope",
    "check",
    "field_errors",
    "measurement_identity",
    "reference_path",
    "spectral_errors",
]

if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import Complex, Float


@dataclass(frozen=True)
class AccuracyEnvelope:
    """Freeze a reviewed acceptance ceiling independently of solver tolerance.

    Attributes
    ----------
    case : str
        Public case alias.
    revision : int
        Accuracy-policy revision, independent of the case and result schemas.
    bound : float
        Dimensionless relative field and retained-band error ceiling.
    measurements : tuple of str
        Exact workload/measurement identities covered by this review.
    """

    case: str
    revision: int
    bound: float
    measurements: tuple[str, ...]


# SEP 30 Part II records the independent reference/refinement pilot and its
# review. The ceilings use source-campaign target levels with headroom, not a fit
# to the fastest candidate. Hashes prevent a changed window inheriting this review.
# CtC revision 2 (0.3 m) and reduced CtC revision 2 (one mode, 2 ps) were requalified
# on 2026-10-24 against fresh tight references: field errors 5.8e-8 and 2.4e-7
# (Multistep, RK) for CtC and 3.0e-7 and 4.6e-8 for reduced CtC, with the references'
# own coarse-to-tight difference 4e-11, so the reviewed ceilings carry over.
# SEP 21 phase 2 (2026-10-25) re-gridded the cases at their new revisions
# (benchmarks/registry.py); the measurement hashes below name those revisions. The
# ceilings are carried over from the review of the former grids and have not been
# requalified on the new ones.
CORE_ENVELOPES = (
    AccuracyEnvelope(
        "rdw",
        1,
        1e-4,
        (
            "535191adef32c0151a07f4a27b23ad54a553814ff20eba70a88f10f255f10007",
            "cd6a383a6a760157e2caaa300ad8bffce5bb7889316617528c400ba70e3ae9de",
        ),
    ),
    AccuracyEnvelope(
        "ctc",
        1,
        1e-5,
        (
            "f9119a6f4a12f5610f07898465baff64e9c81a458a17df7a84a15b1a46930608",
            "4d055999b7d0bd0cccbfc62c19f0000a376b48974dabad2e9162fababe59dafd",
        ),
    ),
    AccuracyEnvelope(
        "radial_filament_256",
        1,
        1e-4,
        (
            "06ea9b0501632011742be494d81afe398485af3a23780a786c99134851c30261",
            "79805b76638c6ddbdc8aa141fe6081579651e2e282368051782ba48a1bf8d260",
        ),
    ),
)


# Reviewed phase 2 ceilings; see SEP 30 Part II for measured evidence.
MODAL_ENVELOPES = (
    AccuracyEnvelope(
        "rdw_one_mode",
        1,
        1e-4,
        (
            "ea018e193a4cbc0398a2cd53d81a3289f2e7ba86bfd2f97af1fdad10d64cdfac",
            "6030641f91a88e938316a4ef72e8ad7030ca5485dcda00cd1adf374e373f3746",
        ),
    ),
    # Workload revision 2 (the 2 ps window): the same 1e-4 ceiling, qualified anew
    # (candidates 3.3e-7 and 1.1e-7 against tight RK, refinement witness 1.1e-11).
    AccuracyEnvelope(
        "ctc_reduced",
        1,
        1e-4,
        (
            "9e01c4dcd39183d4d7308674a5db6750453dcb51e3878a0450a705766a2491f3",
            "a2aa2221a8500e1309d918525556100b59de1f6e4a44ed4c04589915cd6166bd",
        ),
    ),
    AccuracyEnvelope(
        "argon_blueshift",
        1,
        1e-4,
        (
            "a08c1f2f09b8755d8b1f8ce99223f0b154021887ed3fa36f4f3faaadae9198f5",
            "1c9d67a55ae2f343376e56508c204bf65f1033fc3e6b67dbc7a16e93d597b5d1",
        ),
    ),
    AccuracyEnvelope(
        "h2_raman_6ps",
        1,
        1e-4,
        (
            "5b72fa31b826c801b7cf7e86c9bfa9251e47f8c3ffd1b689ae91cec003596de0",
            "f044a7a680e4371b21da6b195a8bd72c8296fde3d0dda73cd35158a196dbd4e2",
        ),
    ),
    AccuracyEnvelope(
        "he_rdw_averaged",
        1,
        1e-4,
        (
            "2e9b2768bf28969faee1e2e83e7b5c5e24c860577d0e883a1cd99ec350549c6d",
            "75fca12e36f6baeb88201a161a64a8029ae3282cafedd85de7a62938ce60aba8",
        ),
    ),
    AccuracyEnvelope(
        "argon_blueshift_averaged",
        1,
        1e-4,
        (
            "bb0e3cc3aad09187afbc2abe309412ed1064e4b6f0729412f4fee4d46b6cf32a",
            "0a6098dfa8ca092a9304e6fa0511467204c6b2877b9e8bf426f29677ef205449",
        ),
    ),
    AccuracyEnvelope(
        "h2_raman_6ps_averaged",
        1,
        1e-4,
        (
            "5e856d2ee20cc5103670ce61a54776acd0d5380f61e55c95577ec3ed2dc99e2b",
            "5d72fa4851658d2954c41a58ab72d9f340e439fc2bdc1c43fa2fb91d83d0a853",
        ),
    ),
    AccuracyEnvelope(
        "gnlse_soliton",
        1,
        1e-4,
        (
            "e31d67b8797c00180db3944955b21ef7c8b6be60d2b4eff5767b88685193e572",
            "9a9da42bd94689463581c94500f1abd943ee8ecbd94535b9f2bd9b9fdf97049d",
        ),
    ),
    AccuracyEnvelope(
        "gnlse_raman_shock",
        1,
        1e-4,
        (
            "5b2d72a9ce3a1f012b192fee0c9b30f23c9df95909d47531e55f5bf0c0f4689e",
            "54202dd6cda53db7c5189d434b737995ba94dd23021cf5f810cedd131fad40e0",
        ),
    ),
)


# Reviewed advanced and spatial ceilings, with measured headroom.
ADVANCED_ENVELOPES = (
    AccuracyEnvelope(
        "scalar_wall",
        1,
        1e-4,
        (
            "a0642916ad9e5d57b684c26250750b6d64883bc796f2eaff51810b56f22c5fa8",
            "348b426520876847f7fd8df2e19614e95fa0019908ecefb13e946f61c660f87d",
        ),
    ),
    AccuracyEnvelope(
        "vector_wall",
        1,
        1e-4,
        (
            "581f5e525d21700b212a23effebe7ff6ca698851181405f78cd133aedc02e1b3",
            "fb307c7d114a4ea4c277a558626d0f4120b8f6b8bd030ca415a2a4cf03a3ef7c",
        ),
    ),
    AccuracyEnvelope(
        "taper",
        1,
        1e-4,
        (
            "759ff7d3c95925fe23b9997a9cd6706b92a916727234028231d0522ed9e50c8b",
            "dabe45cc3a22056cb3053f4a93f2236c53336b0396f4ef2244347eff54c8cf56",
        ),
    ),
    AccuracyEnvelope(
        "taper_wall",
        1,
        1e-4,
        (
            "c941817a9254188e4a6ab0003dec748ff39939eb6ce4fcf4cd4ee5d385c542cc",
            "784fc26a3265d24199a666a735444881909479b114484b04637b51f3402e9b13",
        ),
    ),
    AccuracyEnvelope(
        "step_index_pair",
        1,
        1e-4,
        (
            "8ba57ed541529a90e8e03723f9ade9c0bcdedda1c9e3d3898eea005304f0cbed",
            "c609923210bf9e6535685affe4d53bb0d8368e97109345a888e09468a46feca9",
        ),
    ),
    AccuracyEnvelope(
        "grin_linear",
        1,
        1e-4,
        (
            "967d370e58f4c9f737177826d7ccae67adaa042206847d3715767dcabaecd47e",
            "fb5ab666bacc8d6927571dabcdad939b344d1a65cbb5e7d35041285272a38c5f",
        ),
    ),
    AccuracyEnvelope(
        "grin_nonlinear",
        1,
        1e-4,
        (
            "13f3513e35a7c5facabecbc8b5917c55ea9d239db82a7851cc6c93de20ac7184",
            "2ea66a1214e9d6777c0e536c2f3c62e9cd8fef238c69c03cd7e0c4242d30dc1f",
        ),
    ),
    AccuracyEnvelope(
        "moving_solid_guide",
        1,
        1e-4,
        (
            "0c042beeb6689a36ace6236146a5ae33dd2b4c8274fb0ef997c1ca7612c8e655",
            "06a10ab3f5ffbcc7b4df3c768864d261738ef9a0687b0b73c3d1f22c0a941a6b",
        ),
    ),
    AccuracyEnvelope(
        "line",
        1,
        1e-4,
        (
            "6f947cd489b8ca4620292bac87027aeaca455c446eb07f18ee2cf8d3cca7bf9d",
            "0614197f59685f70f758d22ddebf512e013c53696ed0ef5e5b7950970d2956c4",
        ),
    ),
    AccuracyEnvelope(
        "planar",
        1,
        1e-4,
        (
            "c5233b5bfdbc9152fdd88ec88ec3494744a14c0f6162a87109a194fbd8be2957",
            "ed6d92503416e0fabac2f2c6f203b587ade9bfb463a7102c2065d26b3ec78ea9",
        ),
    ),
    AccuracyEnvelope(
        "spatial_absorber",
        1,
        1e-4,
        (
            "e0890b8c1076f8ee0a6c121a989335e545aedec07157cf8f15c82df6841ddf8a",
            "87afcc77d2c04d18b839070c906b03baf3d6ba3e65398cc1a7833bd9ec128007",
        ),
    ),
)


# Reviewed backend-exercise ceilings, independent of solver settings.
EXERCISE_ENVELOPES = (
    AccuracyEnvelope(
        "vector_raman",
        1,
        1e-4,
        (
            "042691e4d922727e1d39a965c420c4253c56e42e352dca742316e20869486f98",
            "ea3f908fc767ced99db8b7b5eedb6ef0c76c80b829dc82619d995680c09228ee",
        ),
    ),
    AccuracyEnvelope(
        "envelope_plasma",
        1,
        1e-4,
        (
            "a7192b6564f5080e6b6c1b8547664724c37831bb401061ccc7f2fcb84df37948",
            "5018094657908f68afe83e89a27fb3ad869c8008ed6a250a4d81b8baddc1caae",
        ),
    ),
    AccuracyEnvelope(
        "mixture_gradient",
        1,
        1e-4,
        (
            "d53f4aacad210a9c530eadb8c520bf10d72aee72d811b2feb470627a5a9bb53a",
            "e476a16527ab9f2ec6ad2471656bf598eaa0e293ee10ad8c54caec2121575b20",
        ),
    ),
    AccuracyEnvelope(
        "dispersive_kerr",
        1,
        1e-4,
        (
            "563fab8b6ca31f1988892b1ed49e7a7cc8329e8249842eb8ef27c7e645828d1e",
            "c9548440f1d9ba7701401f33edccabb36f97db5fd085f328620912155b65f24f",
        ),
    ),
    AccuracyEnvelope(
        "plasma_band",
        1,
        1e-4,
        (
            "f3c3a476d422e9f9a6185bde3c5c6bfdac1de4813eb8856df3d7935ed7450ead",
            "b5a55c882ee996fcaf7916cfb3609b0944333ead811a8a1de010ff531c8d6fa5",
        ),
    ),
    AccuracyEnvelope(
        "plasma_trapezoid",
        1,
        1e-4,
        (
            "96c5ba349d53e7013172d9b783ad596fd9527a6dfd32093e70dcc6178baf2a43",
            "5cb640050f6aa36d17433d198c56a8cb433a915cc5a725b855054f60450767b5",
        ),
    ),
    AccuracyEnvelope(
        "noise",
        1,
        1e-4,
        (
            "6eb5dd34de4404092d6f08723a87dbaaef24fd67a369f5555e377c73edcb7645",
            "bf18c4ae7b01e904c6afe63422a60b7569359fca67e904cb5f6f12ea0ca14314",
        ),
    ),
    AccuracyEnvelope(
        "dense_output",
        1,
        1e-4,
        (
            "1cf95ca74e3ba9640f42fb73fd4902ea423a0be815b732374dbfe2a89b93f1f9",
            "7d74cf7591ee889f87428690bd3cbe9d3784e373f0b8084dd3ec9f6a135e5b74",
        ),
    ),
    AccuracyEnvelope(
        "absorber_schemes",
        1,
        1e-4,
        (
            "ec07ca4f09b515bc248662da37421c438acb20747c10d993878fcf3ba07fc97f",
            "ca6ffbd2daef828d232f0925b9194787deb72173ea682752c3835211e84e3688",
        ),
    ),
    AccuracyEnvelope(
        "ordinary_stats",
        1,
        1e-4,
        (
            "5da7c8c23ad39f15a52f66fec5b61d5e7ddf680d518d1174e78aa6e23a30eedc",
            "ec5e08975186a3079d873039e98e21dd7c2ecb0b39886cdce37a8d00054a7ad2",
        ),
    ),
    AccuracyEnvelope(
        "far_field_stats",
        1,
        1e-4,
        (
            "9ea383414cc72b76935faba42b4ed178c22ed84c3d039358b1b86c69fad9882c",
            "f717d6d4aa19f51c2acfa4090955d06714448d27fb4c640f0e3bfa24fbf0a707",
        ),
    ),
)


@dataclass(frozen=True)
class ReferenceReceipt:
    """Freeze a numerical reference independently of timing source/build identity."""

    revision: int
    case_hash: str
    measurement_hash: str
    fields_file: str
    fields_hash: str
    band_file: str
    band_hash: str
    accuracy_bound: float | None
    refinement_floor: float | None
    evidence_file: str
    evidence_hash: str
    protocol: str
    qualified_for_adoption: bool = False
    omega_file: str | None = None
    omega_hash: str | None = None
    reference_revision: int = 1
    policy_revision: int = 1
    supersedes: str | None = None
    revision_note: str | None = None


def measurement_identity(configuration: Record) -> str:
    """Exclude execution choices while retaining method, tolerance and window."""
    return identity(
        {
            key: value
            for key, value in configuration.items()
            if key not in ("backend", "threads", "profile_hash", "policy")
        }
    )


def approved_envelope(configuration: Record) -> AccuracyEnvelope | None:
    """Return a reviewed bound only for its exact frozen measurement."""
    key = measurement_identity(configuration)
    return next(
        (
            item
            for item in (
                *CORE_ENVELOPES,
                *MODAL_ENVELOPES,
                *ADVANCED_ENVELOPES,
                *EXERCISE_ENVELOPES,
            )
            if key in item.measurements
        ),
        None,
    )


def field_errors(
    fields: Complex[np.ndarray, "n_z ..."], truth: Complex[np.ndarray, "n_z ..."]
) -> list[float]:
    """Return relative L2 errors for each plane, refusing malformed references."""
    import numpy as np  # noqa: PLC0415 - qualification/worker only

    if fields.shape != truth.shape or len(fields) == 0:
        raise SuiteError("reference shape mismatch or empty fields")
    errors = []
    for found, expected in zip(fields, truth, strict=True):
        denominator = float(np.linalg.norm(expected.ravel()))
        if denominator == 0 or not np.isfinite(denominator):
            raise SuiteError("reference field norm is zero or non-finite")
        error = float(np.linalg.norm((found - expected).ravel())) / denominator
        if not np.isfinite(error):
            raise SuiteError("non-finite field error")
        errors.append(error)
    return errors


def spectral_errors(
    fields: Complex[np.ndarray, "n_z n_modes n_pol n_omega"],
    truth: Complex[np.ndarray, "n_z n_modes n_pol n_omega"],
    omega: Float[np.ndarray, " n_omega"],
    case: str,
) -> Record:
    """Retain the source campaigns' weak-bin, UV and modal-energy diagnostics.

    The weak bins are frozen from the reference's final-plane power, between
    1e-9 and 1e-3 of its peak. Band norms have the source campaign's 1e-3 floor
    across planes. RDW additionally uses 100--150 nm. See SEP 16's
    ``campaign.py::score`` and ``varying_campaign.py::score``. At a common grid,
    Parseval's constant cancels from the relative modal-energy difference, whose
    endpoint weights follow ``solprop.fourier.energy_omega`` (conventions §4).
    Energy is a diagnostic, not a separate acceptance threshold.
    """
    import numpy as np  # noqa: PLC0415

    from solprop.constants import C  # noqa: PLC0415 - worker/qualification only
    from solprop.fourier import energy_omega  # noqa: PLC0415

    if case not in ("rdw", "ctc"):
        return {}
    if (
        fields.ndim != 4
        or fields.shape != truth.shape
        or not len(fields)
        or omega.shape != (fields.shape[-1],)
        or not np.all(np.isfinite(omega))
        or not np.all(np.isfinite(fields))
        or not np.all(np.isfinite(truth))
    ):
        raise SuiteError("spectral metric shape or finite-value contract is invalid")
    if np.any(np.linalg.norm(truth.reshape(len(truth), -1), axis=1) == 0):
        raise SuiteError("spectral reference has a zero-norm plane")
    found, reference = fields.reshape(len(fields), -1), truth.reshape(len(truth), -1)
    power = np.abs(reference[-1]) ** 2
    masks = {"weak": (power > 1e-9 * power.max()) & (power < 1e-3 * power.max())}
    if case == "rdw":
        ultraviolet = (omega >= 2 * np.pi * C / 150e-9) & (
            omega <= 2 * np.pi * C / 100e-9
        )
        masks["uv"] = np.broadcast_to(ultraviolet, truth.shape[1:]).reshape(-1)
    result: Record = {}
    for name, mask in masks.items():
        norms = np.linalg.norm(reference[:, mask], axis=1)
        scale = float(norms.max())
        if not scale > 0:
            result[name] = {"value": None, "reason": "reference band has zero norm"}
            continue
        differences = np.linalg.norm((found - reference)[:, mask], axis=1)
        values = differences / np.maximum(norms, 1e-3 * scale)
        result[name] = {
            "value": float(values.max()),
            "planes": values.tolist(),
            "selected_bins": int(mask.sum()),
        }
    # Both core modal cases use field grids and saved true-power amplitudes.
    # The common dt cancels; the endpoint/interior weights must still be retained.
    reference_energy = energy_omega(truth, 1.0, kind="field").sum(axis=(1, 2))
    energy = energy_omega(fields, 1.0, kind="field").sum(axis=(1, 2))
    values = np.abs(energy / reference_energy - 1)
    result["energy"] = {
        "value": float(values.max()),
        "planes": values.tolist(),
        "role": "diagnostic",
    }
    return result


def reference_path(receipt: Record, key: str, directory: Path) -> Path:
    """Validate a stored artifact's relative path and content identity."""
    path = (directory / receipt[f"{key}_file"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise SuiteError("reference artifact escapes its directory")
    if file_hash(path) != receipt[f"{key}_hash"]:
        raise SuiteError(f"reference {key} hash mismatch")
    return path


def check(fields: np.ndarray, configuration: Record, path: Path) -> Record:
    """Compare after measurement, retaining full-band and scored-band errors."""
    import numpy as np  # noqa: PLC0415

    receipt = read_record(path)
    if receipt.get("revision") != 1:
        raise SuiteError("unsupported reference receipt revision")
    if receipt["case_hash"] != configuration["case_hash"] or receipt[
        "measurement_hash"
    ] != measurement_identity(configuration):
        raise SuiteError("reference measurement identity mismatch")
    truth = np.load(
        reference_path(receipt, "fields", path.parent),
        allow_pickle=False,
        mmap_mode="r",
    )
    band = np.load(reference_path(receipt, "band", path.parent), allow_pickle=False)
    reference_path(receipt, "evidence", path.parent)
    if band.dtype != np.bool_ or band.shape != (fields.shape[-1],) or not band.any():
        raise SuiteError("reference band shape or values invalid")
    full_errors = field_errors(fields, truth)
    errors = field_errors(fields[..., band], truth[..., band])
    bound = receipt.get("accuracy_bound")
    if bound is not None and (not np.isfinite(bound) or bound <= 0):
        raise SuiteError("reference accuracy bound must be positive and finite")
    worst = max(errors)
    metrics = {}
    if receipt.get("omega_file") is not None:
        omega = np.load(
            reference_path(receipt, "omega", path.parent), allow_pickle=False
        )
        metrics = spectral_errors(fields, truth, omega, configuration["case"])
    missing_metrics = configuration["case"] in ("rdw", "ctc") and not metrics
    metrics_pass = bound is not None and all(
        item["value"] is not None and item["value"] <= bound
        for name, item in metrics.items()
        if name != "energy"
    )
    return {
        "status": "not-qualified"
        if bound is None or missing_metrics
        else "passed"
        if worst <= bound and metrics_pass
        else "invalid",
        "worst_plane_relative_field_error": worst,
        "all_bins_error": max(full_errors),
        "plane_errors": errors,
        "accuracy_bound": bound,
        "reference_id": identity(receipt),
        "reference_revision": receipt.get("reference_revision", 1),
        "policy_revision": receipt.get("policy_revision", 1),
        "spectral_metrics": metrics,
        "qualified_for_adoption": receipt.get("qualified_for_adoption", False)
        and not missing_metrics
        and metrics_pass
        and bound is not None
        and worst <= bound,
    }
