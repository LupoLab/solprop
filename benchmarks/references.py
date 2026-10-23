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
CORE_ENVELOPES = (
    AccuracyEnvelope(
        "rdw",
        1,
        1e-4,
        (
            "6ea41171e90bd551e8b3afcf05dcf1d75f570f8db36f75f193302b2ea5e31732",
            "a2c01b6756f5257aecfbdfa1d801a27d790df0c033fbfc3f80a2a6f6195d64af",
        ),
    ),
    AccuracyEnvelope(
        "ctc",
        1,
        1e-5,
        (
            "b9bb84f300848a0fb3a7132c92039697f977f6c1ebf9eb08f98f98a8c1c547de",
            "e43c31b1ed8784d9d85e210336a7484752aa29782da5439928c1ddfb3f5eb21e",
        ),
    ),
    AccuracyEnvelope(
        "radial_filament_256",
        1,
        1e-4,
        (
            "fd4f0b9918208908821ebad776d54b34ff476f752bb17ad06b5c4c8d9d1c7982",
            "27a930d6d7849dc5422a6cb053def410743b2d5890adbbf8dcc951d95ae9abdb",
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
            "84398731ca1744959ce10c617def158c21c3103589fa2d1708c0af7d79dabcd1",
            "bc171d288de1c2889b0023888f375b4d2327ddf6fc64d49da27fb72757a30a85",
        ),
    ),
    AccuracyEnvelope(
        "ctc_reduced",
        1,
        1e-4,
        (
            "565159afbdb832475b42b84e381998dffd57e12acfa55f7185acb775e2466909",
            "29e3e85513c29fd5ecebec59d1933c836c0d00524cf2339aad9f2b42b8ffe664",
        ),
    ),
    AccuracyEnvelope(
        "argon_blueshift",
        1,
        1e-4,
        (
            "535113368d35b09ad660d445aee2233020f9c374a876142b35c873378569fc7d",
            "a98e65bf58f35de0f4a456c42b6a7f149166f5e3b495f6fa143fc4f0812b2ce8",
        ),
    ),
    AccuracyEnvelope(
        "h2_raman_6ps",
        1,
        1e-4,
        (
            "6a17ad8758d995c1458fec33ad01536b26d2fe03a3fb46ea9964ac30e7b7f2c6",
            "0679ec539ff730fe2aae493e9d807cffb5f012179b8c98be7b3e5c1bf608298e",
        ),
    ),
    AccuracyEnvelope(
        "he_rdw_averaged",
        1,
        1e-4,
        (
            "888f3999189c641288bcc860fbddcf4624db391e5578805d4c494496d2919c83",
            "9c8963f38ea41269e16a523be941159d9c75cb5baa8607b1077a9c9abcea46ff",
        ),
    ),
    AccuracyEnvelope(
        "argon_blueshift_averaged",
        1,
        1e-4,
        (
            "ab37a5d92bd7e06c1df1a0a14796f69af327d1992033acfe209cca544db43114",
            "74837207a40ba6312b0c495771993123e6e70a6d86bb4e51032b7c2a98a76636",
        ),
    ),
    AccuracyEnvelope(
        "h2_raman_6ps_averaged",
        1,
        1e-4,
        (
            "518d3484e2f2f3da16db6819464eaf6d08976574b13e89d9f08bde274aa0d851",
            "980f036d6e7bedfab73abc814f03edd1d5cfedaf5145db8cfd0287dfbb19306c",
        ),
    ),
    AccuracyEnvelope(
        "gnlse_soliton",
        1,
        1e-4,
        (
            "aa895940a728b6e3b57472987b29dc740e06087cd694b58342436cfc39af9296",
            "0780a2bbbf0b678986662eda981391524d0385524a7d8a8fb365e0711e1166a0",
        ),
    ),
    AccuracyEnvelope(
        "gnlse_raman_shock",
        1,
        1e-4,
        (
            "eadafdd81e90c9038c378fb450e4eb6092d4dd8100cf371bc0a4a8cc28519320",
            "aded17e77fe4c1a7838d8aca5cfc1729301dda9fbd6cf3c17d242f76f8e4ea95",
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
            "9ba5e7e873ed929e2b7db006932ebee35ca873548f29851fc18aaa0422ba9caa",
            "86687e068e8218247971ae5623cc969ce18f3de1ba1ca66a27e282f4fd7aa9e1",
        ),
    ),
    AccuracyEnvelope(
        "vector_wall",
        1,
        1e-4,
        (
            "1c38a9c2d6a6a39caf2b8b06cb6cdf4b1f4bb785c305a20ec1f3342f0484ba85",
            "94433388be568d010b2800fbd3fa0d17c3319069e3eb7fd6077050f10f65c48c",
        ),
    ),
    AccuracyEnvelope(
        "taper",
        1,
        1e-4,
        (
            "9deef7413f17aba0e269726bb6a54900d588bc9dff6ed96979dc2b9114522f35",
            "592a601bd2e5504a13c345227947ae10ce6a1ac809b793f158b179ed2f5458aa",
        ),
    ),
    AccuracyEnvelope(
        "taper_wall",
        1,
        1e-4,
        (
            "3bbe22681b246cb1a77257208090f8d2bc20046a139b697035bb63ecacd3b6ae",
            "473f17d1d87579fe1eeda62eaffc5bf54e3f7135043fd9977aff0f28131f774b",
        ),
    ),
    AccuracyEnvelope(
        "step_index_pair",
        1,
        1e-4,
        (
            "81627ea4c59490d57102307eed5671ca58418f6f53722388a9a78d60ee6902b1",
            "a7864758611f50c7cc5fa8d08cffa9d3995933c21edcf6b322d34bc2fcf6b92d",
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
            "dc66ca6ccc1a65baadec48f7c96e26e37b2d199d57dba41e4500e119fc0121c7",
            "d49290aca313beff4ebc73243cd2d9d186cba2bf9d52669d1ff65e16659d9bb3",
        ),
    ),
    AccuracyEnvelope(
        "moving_solid_guide",
        1,
        1e-4,
        (
            "129cd4088e49bf0c1c5331e9749cd36cd46ec312727e35409238daef177b78e4",
            "f0b8f82edd4d13115be421e1810c7ef209aa8e81007507812a076a3f85e7ec63",
        ),
    ),
    AccuracyEnvelope(
        "line",
        1,
        1e-4,
        (
            "8bf25ceed3203b27ec13865165ef6cea225c1286cc7525429e24fec86ed612c1",
            "980eecce15e9ec1945afeceb1f78cb664b8772244d31d494eeccd056af778c5b",
        ),
    ),
    AccuracyEnvelope(
        "planar",
        1,
        1e-4,
        (
            "0e4d9b2bf5cdc30fad613f55f907c4f33afdf1d4ace3f11233149024223d1c81",
            "a41e0224f29efeb3de357cf4dc9cf5f16bf5c3ad0c11e21df44be5073b7be14e",
        ),
    ),
    AccuracyEnvelope(
        "spatial_absorber",
        1,
        1e-4,
        (
            "7de22f4914a486c40a822928232299064908a0f3debb5af1aaf623b68d729ee1",
            "0c007223550acdbb8c4eeb6a997af893bd955f0470a687a3954f2a58d9f37f80",
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
            "0b4c6af3f20ade9ecec1072f1eac075928ad22c14e6318a2da9f95e5031fe3ba",
            "b1a6e6fdd496a92a335aa8007567145fa26320c3f0221a3bed57c2a4bcc584d0",
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
            "2d1c9a98977d3e4839cc5558e318c10548b9bad9d8df16581fb5772c89138e60",
            "6dea7e88f7cfdffc39be63c097a2966a05134e9e71d0e7471c4c55ad6dd6973c",
        ),
    ),
    AccuracyEnvelope(
        "dispersive_kerr",
        1,
        1e-4,
        (
            "588813bb72fc29f6b3834ded9c7078a8a1a5746eef2680c75ad65e5e5a272c96",
            "97688f847d3f35cb2be89782cd16aef024201a814da67bf4b4da67a19cc67f19",
        ),
    ),
    AccuracyEnvelope(
        "plasma_band",
        1,
        1e-4,
        (
            "7a9effb9c27b84f72195703d8e302ba5684861461af079ba1af26d1e66b7c8cf",
            "d19a47f0d7595f813c452a8d21ae826c606d9b179f7e7963a2efddbd18abffab",
        ),
    ),
    AccuracyEnvelope(
        "plasma_trapezoid",
        1,
        1e-4,
        (
            "abda1dacc9a31f448e8b3690bc8c55f6fb1954e652d1d20cc9e081bc6aed41df",
            "1bcf695c1d051f8b5c211292642f0e726412067f22f8dd79bdda801336f5d286",
        ),
    ),
    AccuracyEnvelope(
        "noise",
        1,
        1e-4,
        (
            "5d3495b4fd41c594e540e16df660d76e7550d68a93e923b1fb5a916f0fe9a002",
            "ab15d2ce3fdb08db0e9da45629045f6d895fb619d2bdbe3c18b7f3073adaccf2",
        ),
    ),
    AccuracyEnvelope(
        "dense_output",
        1,
        1e-4,
        (
            "c6c68a6abd1d74cb374c726452ca8e44a9ac752fd3278ce50c94a54fcadcac1b",
            "8bd833422a9dab440b53eb4e7fa10972b9d4db9e184dace1908745bfcd429255",
        ),
    ),
    AccuracyEnvelope(
        "absorber_schemes",
        1,
        1e-4,
        (
            "a7d905d7c129c6019fd9deee3ab6dc45b30f75d2dc3bbe5337d03f72a5f32ff1",
            "8d3fd4c34ccbbeba9a7858521de43bd338139c89619e6a2a8c322a96d4de25c0",
        ),
    ),
    AccuracyEnvelope(
        "ordinary_stats",
        1,
        1e-4,
        (
            "4dabb07cb9e633d7ac63577fb19172c86848d5d865078d0feaa8f46397ab6334",
            "0900e26546e1db9bc6c852a3fe9bc8794a79e8e309d4fa48a09801235982547f",
        ),
    ),
    AccuracyEnvelope(
        "far_field_stats",
        1,
        1e-4,
        (
            "d1de6a66bc4d147aabe4d590f7255e7d0c55ebf8682b55c58b88052af3d5a989",
            "0d107f4509d4d8854093958451882b59b6a0e04c4e52274d175d49f7e76c34aa",
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
