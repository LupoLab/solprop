# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Topology, allocation classes, fingerprints and the profile a process applies
(solprop SEP 46 sections 5 and 6.4)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from solprop_native import _profile
from solprop_native._allocation import AllocationClass, classify, match
from solprop_native._execution import builtin
from solprop_native._fingerprint import CALIBRATION_VERSION, build_identity, fingerprint
from solprop_native._profile import Process, choose
from solprop_native._store import Generation, Row, Store, StoreWarning
from solprop_native._topology import Topology, parse_cpu_list, read_topology


def _sysfs(
    root: Path,
    *,
    sockets: int,
    cores: int,
    smt: bool,
    l2_per: int = 1,
    capacity: Mapping[int, int] | None = None,
) -> Path:
    """Write a fake ``/sys/devices/system/cpu`` with `sockets` x `cores` cores."""
    base = root / "sys" / "devices" / "system" / "cpu"
    threads = 2 if smt else 1
    total = sockets * cores
    count = total * threads
    base.mkdir(parents=True)
    (base / "online").write_text(f"0-{count - 1}\n", encoding="utf-8")
    for cpu in range(count):
        core = cpu % total
        package = core // cores
        local = core % cores
        siblings = [core + total * index for index in range(threads)]
        directory = base / f"cpu{cpu}"
        (directory / "topology").mkdir(parents=True)
        (directory / "topology" / "physical_package_id").write_text(f"{package}\n")
        (directory / "topology" / "core_id").write_text(f"{local}\n")
        (directory / f"node{package}").mkdir()
        first_l2 = package * cores + (local // l2_per) * l2_per
        l2 = [
            c + total * index
            for c in range(first_l2, first_l2 + l2_per)
            for index in range(threads)
        ]
        l3 = [
            c + total * index
            for c in range(package * cores, (package + 1) * cores)
            for index in range(threads)
        ]
        for index, (level, members) in enumerate(
            [("1", siblings), ("2", l2), ("3", l3)]
        ):
            cache = directory / "cache" / f"index{index}"
            cache.mkdir(parents=True)
            (cache / "level").write_text(level + "\n")
            (cache / "type").write_text("Unified\n" if level != "1" else "Data\n")
            (cache / "shared_cpu_list").write_text(
                ",".join(str(c) for c in sorted(members)) + "\n"
            )
        if capacity is not None:
            (directory / "cpu_capacity").write_text(f"{capacity[core]}\n")
    return base


def test_cpu_lists_parse_ranges_and_singletons() -> None:
    assert parse_cpu_list("0-2,5,7-8\n") == (0, 1, 2, 5, 7, 8)
    assert parse_cpu_list(",") == ()


def test_a_two_socket_smt_machine_reads_its_caches_and_nodes(tmp_path: Path) -> None:
    root = _sysfs(tmp_path, sockets=2, cores=4, smt=True)
    topology = read_topology("Linux", root=root)
    assert topology.cpus == tuple(range(16))
    assert len(set(topology.core.values())) == 8
    assert topology.core[0] == topology.core[8]
    assert topology.socket[3] == 0
    assert topology.socket[4] == 1
    assert topology.numa[4] == 1
    assert topology.l3[1] == topology.l3[3] != topology.l3[4]
    record = topology.record()
    assert record["sockets"] == 2
    assert record["smt"] is True
    assert record["kinds"] == [8]
    # The whole machine; one socket's cores alone; across sockets without SMT.
    whole = classify(topology, topology.cpus)
    assert whole == AllocationClass(
        cores=8, kinds=(8,), l2_groups=0, l3_groups=2, numa_nodes=2, sockets=2, smt=True
    )
    one = classify(topology, [0, 1, 2, 3])
    assert (one.cores, one.sockets, one.smt) == (4, 1, False)
    spread = classify(topology, [0, 1, 4, 5])
    assert (spread.cores, spread.sockets, spread.numa_nodes) == (4, 2, 2)
    assert one != spread
    assert one.key() != spread.key()
    assert classify(topology, [], quota=2).quota == 2
    assert classify(topology, [0, 1], quota=4).quota is None


def test_shared_l2_groups_and_capacity_kinds_are_read(tmp_path: Path) -> None:
    capacity = {core: 1024 if core < 2 else 512 for core in range(6)}
    root = _sysfs(tmp_path, sockets=1, cores=6, smt=False, l2_per=2, capacity=capacity)
    topology = read_topology("Linux", root=root)
    assert topology.kind[0] == 0
    assert topology.kind[5] == 1
    allocation = classify(topology, topology.cpus)
    assert allocation.kinds == (2, 4)
    assert allocation.l2_groups == 3
    assert classify(topology, [0, 1]).l2_groups == 1


def test_an_intel_hybrid_reads_its_kinds_from_the_pmus(tmp_path: Path) -> None:
    root = _sysfs(tmp_path, sockets=1, cores=4, smt=False)
    devices = root.parent.parent
    (devices / "cpu_core").mkdir()
    (devices / "cpu_core" / "cpus").write_text("0-1\n")
    (devices / "cpu_atom").mkdir()
    (devices / "cpu_atom" / "cpus").write_text("2-3\n")
    assert classify(read_topology("Linux", root=root), [0, 1, 2, 3]).kinds == (2, 2)


def test_macos_numbers_its_cores_by_kind() -> None:
    values = {
        "hw.nperflevels": 2,
        "hw.perflevel0.physicalcpu": 6,
        "hw.perflevel0.cpusperl2": 6,
        "hw.perflevel1.physicalcpu": 12,
        "hw.perflevel1.cpusperl2": 6,
    }
    topology = read_topology("Darwin", sysctl=values.get)
    assert topology.cpus == tuple(range(18))
    assert [topology.kind[cpu] for cpu in (0, 5, 6, 17)] == [0, 0, 1, 1]
    allocation = classify(topology, ())
    assert allocation.kinds == (6, 12)
    assert allocation.l2_groups == 3


def test_unreadable_sources_give_a_flat_machine(tmp_path: Path) -> None:
    flat = read_topology("Linux", root=tmp_path / "missing")
    assert len(set(flat.core.values())) == len(flat.cpus)
    assert read_topology("Darwin", sysctl=lambda name: None).cpus == flat.cpus
    assert read_topology("Plan9").cpus == flat.cpus


def _cls(cores: int, sockets: int = 1, smt: bool = False) -> AllocationClass:
    return AllocationClass(
        cores=cores,
        kinds=(cores,),
        l2_groups=0,
        l3_groups=sockets,
        numa_nodes=sockets,
        sockets=sockets,
        smt=smt,
    )


def test_rows_match_exactly_or_by_the_nearest_smaller_class() -> None:
    rows = [(_cls(4), "four"), (_cls(8), "eight"), (_cls(16, sockets=2), "two")]
    exact = match(_cls(8), rows)
    assert exact is not None
    assert (exact.row, exact.exact, exact.scope) == ("eight", True, "all")
    nearer = match(_cls(12), rows)
    assert nearer is not None
    assert (nearer.row, nearer.exact, nearer.scope) == ("eight", False, "workspace")
    # A class of another span structure, or smaller than every row, has no row.
    assert match(_cls(12, sockets=2), rows) is None
    assert match(_cls(2), rows) is None
    assert match(_cls(8, smt=True), rows) is None
    # A later row of an equal class replaces an earlier one.
    later = match(_cls(8), [*rows, (_cls(8), "upgraded")])
    assert later is not None
    assert later.row == "upgraded"


def test_allocation_classes_round_trip_and_refuse_malformed_records() -> None:
    cls = _cls(8)
    assert AllocationClass.read(cls.record()) == cls
    for broken in (
        {},
        cls.record() | {"kinds": "8"},
        cls.record() | {"cores": 8.0},
        cls.record() | {"smt": 0},
        cls.record() | {"quota": "2"},
    ):
        with pytest.raises(ValueError, match="not an allocation class"):
            AllocationClass.read(broken)


def _package(tmp_path: Path, extension: bytes = b"binary") -> Path:
    package = tmp_path / "solprop_native"
    package.mkdir()
    (package / "_core.abi3.so").write_bytes(extension)
    (package / "_build.json").write_text(
        json.dumps({"compiler_id": "GNU", "openmp_libraries": "gomp,pthread"})
    )
    return package


def test_the_fingerprint_digests_machine_build_and_version(tmp_path: Path) -> None:
    package = _package(tmp_path)
    identity = build_identity([package])
    assert identity["record"] == {
        "compiler_id": "GNU",
        "openmp_libraries": "gomp,pthread",
    }
    assert isinstance(identity["extension"], str)
    topology = read_topology(
        "Linux", root=_sysfs(tmp_path, sockets=1, cores=2, smt=False)
    )
    first = fingerprint(topology, identity, "Linux", "x86_64")
    assert first == fingerprint(topology, identity, "Linux", "x86_64")
    assert first.calibration == CALIBRATION_VERSION
    assert first.record()["digest"] == first.digest
    # A rebuilt extension, another system or another machine is another fingerprint.
    (package / "_core.abi3.so").write_bytes(b"rebuilt")
    assert fingerprint(topology, build_identity([package]), "Linux", "x86_64") != first
    assert fingerprint(topology, identity, "Windows", "x86_64") != first
    other = read_topology(
        "Linux", root=_sysfs(tmp_path / "b", sockets=1, cores=4, smt=False)
    )
    assert fingerprint(other, identity, "Linux", "x86_64").digest != first.digest
    # No build record or extension: still a fingerprint, of what is there.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build_identity([empty])["record"] is None
    assert build_identity([empty])["extension"] is None


def _machine(tmp_path: Path) -> Topology:
    return read_topology("Linux", root=_sysfs(tmp_path, sockets=1, cores=4, smt=False))


def _publish(
    tmp_path: Path, topology: Topology, package: Path, rows: tuple[Row, ...]
) -> Store:
    store = Store(tmp_path / "store")
    digest = fingerprint(topology, build_identity([package])).digest
    store.publish(
        Generation(fingerprint=digest, name="g1", created="2026-11-08", rows=rows)
    )
    return store


def test_no_profile_or_off_leaves_the_built_in_table(tmp_path: Path) -> None:
    package = _package(tmp_path)
    topology = _machine(tmp_path)
    off = choose({"SOLPROP_CALIBRATION": "off"}, packages=[package])
    assert off.fingerprint is None
    assert off.threads() is None
    assert off.wait() is None
    table = builtin("Linux")
    process = Process(threads=4, wait="spin")
    assert off.workspace(process, table) == (table, "SOLPROP_CALIBRATION=off")
    empty = choose(
        {}, packages=[package], topology=topology, searched=[Store(tmp_path)]
    )
    assert empty.generation is None
    record = empty.record(process, table, table, "no profile")
    assert record["profile"] == "built-in"
    assert record["mode"] == "auto"
    assert record["workspace"]["stage_threshold"] == {  # type: ignore[index]
        "applied": 4096,
        "built_in": 4096,
    }


def test_an_exact_row_supplies_process_and_workspace_settings(tmp_path: Path) -> None:
    package = _package(tmp_path)
    topology = _machine(tmp_path)
    row = Row(
        allocation=classify(topology, topology.cpus).record(),
        level="standard",
        process={"threads": 3, "wait": "passive"},
        workspace={"stage_threshold": 1024, "stage_full": 2048},
        fitted_under={"threads": 3, "wait": "passive"},
    )
    store = _publish(tmp_path, topology, package, (row,))
    chosen = choose(
        {}, packages=[package], cpus=topology.cpus, topology=topology, searched=[store]
    )
    assert chosen.threads() == 3
    assert chosen.wait() == "passive"
    table = builtin("Linux")
    applied, reason = chosen.workspace(Process(3, "passive"), table)
    assert reason == "profile"
    assert (applied.stage_threshold, applied.stage_full) == (1024, 2048)
    # Under other process settings the workspace settings do not hold.
    assert chosen.workspace(Process(3, "spin"), table)[0] == table
    assert chosen.workspace(Process(4, "passive"), table)[0] == table
    assert (
        chosen.workspace(Process(3, "environment: OMP_WAIT_POLICY=passive"), table)[0]
        == table
    )
    record = chosen.record(Process(3, "passive"), applied, table, reason)
    profile = record["profile"]
    assert isinstance(profile, dict)
    assert profile["level"] == "standard"
    assert profile["exact"] is True
    assert record["workspace"]["stage_threshold"] == {  # type: ignore[index]
        "applied": 1024,
        "built_in": 4096,
    }


def test_a_nearer_row_supplies_workspace_settings_only(tmp_path: Path) -> None:
    package = _package(tmp_path)
    topology = _machine(tmp_path)
    row = Row(
        allocation=classify(topology, [0, 1]).record(),
        level="standard",
        process={"threads": 2, "wait": "spin"},
        workspace={"medium_threads": 2},
        fitted_under={"threads": 2, "wait": "spin"},
    )
    store = _publish(tmp_path, topology, package, (row,))
    chosen = choose(
        {}, packages=[package], cpus=topology.cpus, topology=topology, searched=[store]
    )
    assert chosen.found is not None
    assert not chosen.found.exact
    assert chosen.threads() is None
    assert chosen.wait() is None
    applied, reason = chosen.workspace(Process(4, "spin"), builtin("Linux"))
    assert (applied.medium_threads, reason) == (2, "profile")
    # A row for another class only: none applies.
    unrelated = choose(
        {}, packages=[package], cpus=[0], topology=topology, searched=[store]
    )
    assert unrelated.found is None
    assert unrelated.note == "no row for this allocation"


def test_malformed_rows_warn_and_leave_the_table(tmp_path: Path) -> None:
    package = _package(tmp_path)
    topology = _machine(tmp_path)
    allocation = classify(topology, topology.cpus).record()
    rows = (
        Row(allocation={"cores": "many"}, level="quick"),
        Row(
            allocation=allocation,
            level="quick",
            workspace={"stage_threshold": 0},
            fitted_under={"wait": "spin"},
        ),
    )
    store = _publish(tmp_path, topology, package, rows)
    _profile._WARNED.clear()  # pyright: ignore[reportPrivateUsage]
    with pytest.warns(StoreWarning, match="ignoring a calibration row"):
        chosen = choose(
            {},
            packages=[package],
            cpus=topology.cpus,
            topology=topology,
            searched=[store],
        )
    table = builtin("Linux")
    with pytest.warns(StoreWarning, match="workspace settings"):
        assert chosen.workspace(Process(4, "spin"), table)[0] == table
    empty = Row(allocation=allocation, level="quick")
    assert (
        "no workspace"
        in choose(
            {},
            packages=[package],
            cpus=topology.cpus,
            topology=topology,
            searched=[_publish(tmp_path / "x", topology, package, (empty,))],
        ).workspace(Process(4, "spin"), table)[1]
    )


def test_an_unreadable_store_warns_and_leaves_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)

    def broken(*args: object, **kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(_profile, "lookup", broken)
    _profile._WARNED.clear()  # pyright: ignore[reportPrivateUsage]
    with pytest.warns(StoreWarning, match="denied"):
        chosen = choose({}, packages=[package], topology=_machine(tmp_path))
    assert chosen.generation is None
    assert "unreadable" in chosen.note
