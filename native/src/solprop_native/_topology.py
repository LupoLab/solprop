# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The machine's processor topology, read without starting the native runtime.

solprop SEP 46 keys a calibration's rows by *allocation class* (section 6.4): which
cores a process holds, of which kinds, and how many caches, NUMA nodes and sockets
they span. This module reads what that needs:

- Linux: sysfs (``/sys/devices/system/cpu``): each logical CPU's core and package, its
  L2 and L3 groups (``cache/index*/shared_cpu_list``), its NUMA node, and its kind
  where the kernel reports one (``cpu_capacity`` on Arm hybrids, the ``cpu_core`` and
  ``cpu_atom`` PMUs on Intel hybrids).
- macOS: ``hw.perflevel*`` from ``sysctl``: cores of each kind, fastest first, and
  how many share an L2. macOS exposes no CPU affinity, so its CPUs are numbered here
  by kind, fastest first.
- Windows: ``GetLogicalProcessorInformation``: cores, caches, NUMA nodes and packages
  of the first processor group (64 logical CPUs). Core kinds are not read.

Elsewhere, or when a source is unreadable, every CPU is its own core on one socket.

It reads ``sysctl`` through solprop's own reader (``solprop._process``) and loads
nothing of this package.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from solprop._process import sysctl_int

__all__ = ["Topology", "parse_cpu_list", "read_topology"]


@dataclass(frozen=True)
class Topology:
    """Each logical CPU's place in the machine.

    Every mapping is keyed by logical CPU and holds a group number, equal for CPUs in
    the same group.

    Attributes
    ----------
    cpus : tuple of int
        Every online logical CPU.
    core, socket, numa, l2, l3 : Mapping[int, int]
        The physical core, package, NUMA node, and L2 and L3 cache groups.
    kind : Mapping[int, int]
        The core's kind, 0 the fastest; all 0 on a machine with one kind.
    model : str
        The processor model, as the platform names it.
    """

    cpus: tuple[int, ...]
    core: Mapping[int, int]
    socket: Mapping[int, int]
    numa: Mapping[int, int]
    l2: Mapping[int, int]
    l3: Mapping[int, int]
    kind: Mapping[int, int] = field(default_factory=dict[int, int])
    model: str = ""

    def record(self) -> dict[str, object]:
        """Return the topology as JSON data, the part a fingerprint digests."""
        return {
            "model": self.model,
            "cpus": len(self.cpus),
            "cores": len(set(self.core.values())),
            "sockets": len(set(self.socket.values())),
            "numa": len(set(self.numa.values())),
            "l2": _shares(self.l2, self.cpus),
            "l3": _shares(self.l3, self.cpus),
            "kinds": _kind_counts(self, self.cpus),
            "smt": any(
                sum(1 for other in self.cpus if self.core[other] == self.core[cpu]) > 1
                for cpu in self.cpus
            ),
        }


def _shares(groups: Mapping[int, int], cpus: Sequence[int]) -> list[int]:
    """Return the sizes of `groups` over `cpus`, sorted, largest first."""
    sizes: dict[int, int] = {}
    for cpu in cpus:
        sizes[groups[cpu]] = sizes.get(groups[cpu], 0) + 1
    return sorted(sizes.values(), reverse=True)


def _kind_counts(topology: Topology, cpus: Sequence[int]) -> list[int]:
    """Return the physical cores of each kind among `cpus`, fastest first."""
    cores: dict[int, set[int]] = {}
    for cpu in cpus:
        cores.setdefault(topology.kind.get(cpu, 0), set()).add(topology.core[cpu])
    return [len(cores[kind]) for kind in sorted(cores)]


def parse_cpu_list(text: str) -> tuple[int, ...]:
    """Parse a kernel CPU list such as ``"0-3,8,10-11"``."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return tuple(cpus)


def read_topology(
    system: str | None = None,
    *,
    root: Path = Path("/sys/devices/system/cpu"),
    sysctl: Callable[[str], int | None] | None = None,
) -> Topology:
    """Return this machine's topology.

    `root` and `sysctl` replace the Linux sysfs root and the macOS reader, for tests.
    """
    system = system or platform.system()
    if system == "Linux":
        found = _linux(root)
        if found is not None:
            return found
    elif system == "Darwin":
        found = _macos(sysctl or sysctl_int)
        if found is not None:
            return found
    elif system == "Windows":  # pragma: no cover - Windows
        found = _windows()
        if found is not None:
            return found
    count = os.cpu_count() or 1
    return _flat(tuple(range(count)), _model(system))


def _flat(cpus: tuple[int, ...], model: str) -> Topology:
    """Every CPU its own core, on one socket, node and cache."""
    every = dict.fromkeys(cpus, 0)
    return Topology(
        cpus=cpus,
        core={cpu: cpu for cpu in cpus},
        socket=every,
        numa=every,
        l2={cpu: cpu for cpu in cpus},
        l3=every,
        model=model,
    )


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _linux(root: Path) -> Topology | None:
    """Read sysfs; ``None`` when it holds no CPUs."""
    online = _read(root / "online")
    cpus = (
        parse_cpu_list(online)
        if online
        else tuple(
            sorted(
                int(path.name[3:])
                for path in root.glob("cpu[0-9]*")
                if path.name[3:].isdigit()
            )
        )
    )
    if not cpus:
        return None
    core: dict[int, int] = {}
    socket: dict[int, int] = {}
    numa: dict[int, int] = {}
    caches: dict[int, dict[int, int]] = {2: {}, 3: {}}
    capacity: dict[int, int] = {}
    cores: dict[tuple[int, int], int] = {}
    for cpu in cpus:
        base = root / f"cpu{cpu}"
        package = int(_read(base / "topology" / "physical_package_id") or 0)
        core_id = int(_read(base / "topology" / "core_id") or cpu)
        socket[cpu] = package
        core[cpu] = cores.setdefault((package, core_id), len(cores))
        nodes = [path.name for path in base.glob("node[0-9]*")]
        numa[cpu] = int(nodes[0][4:]) if nodes else 0
        for index in base.glob("cache/index[0-9]*"):
            level = _read(index / "level")
            kind = _read(index / "type")
            shared = _read(index / "shared_cpu_list")
            if level in {"2", "3"} and kind in {"Unified", "Data"} and shared:
                caches[int(level)][cpu] = min(parse_cpu_list(shared))
        value = _read(base / "cpu_capacity")
        if value and value.isdigit():
            capacity[cpu] = int(value)
    kind = _linux_kinds(cpus, capacity, root.parent.parent)
    return Topology(
        cpus=cpus,
        core=core,
        socket=socket,
        numa=numa,
        l2={cpu: caches[2].get(cpu, core[cpu]) for cpu in cpus},
        l3={cpu: caches[3].get(cpu, socket[cpu]) for cpu in cpus},
        kind=kind,
        model=_linux_model(),
    )


def _linux_kinds(
    cpus: Sequence[int], capacity: Mapping[int, int], devices: Path
) -> dict[int, int]:
    """Rank each CPU's kind, 0 the fastest: by capacity, or Intel's hybrid PMUs."""
    if len(set(capacity.values())) > 1:
        ranks = sorted(set(capacity.values()), reverse=True)
        return {cpu: ranks.index(capacity.get(cpu, ranks[-1])) for cpu in cpus}
    big = _read(devices / "cpu_core" / "cpus")
    little = _read(devices / "cpu_atom" / "cpus")
    if big and little:
        fast = set(parse_cpu_list(big))
        return {cpu: 0 if cpu in fast else 1 for cpu in cpus}
    return dict.fromkeys(cpus, 0)


def _linux_model() -> str:
    text = _read(Path("/proc/cpuinfo")) or ""
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name.strip() in {"model name", "Model", "CPU part"} and value.strip():
            return value.strip()
    return platform.processor() or platform.machine()


def _macos(sysctl: Callable[[str], int | None]) -> Topology | None:
    """Number the cores by kind, fastest first; ``None`` without ``hw.perflevel*``."""
    levels = sysctl("hw.nperflevels") or 1
    cpus: list[int] = []
    core: dict[int, int] = {}
    l2: dict[int, int] = {}
    kind: dict[int, int] = {}
    for level in range(levels):
        count = sysctl(f"hw.perflevel{level}.physicalcpu")
        if count is None:
            if level == 0:
                count = sysctl("hw.physicalcpu")
            if count is None:
                return None
        per_l2 = sysctl(f"hw.perflevel{level}.cpusperl2") or count
        first = len(cpus)
        for offset in range(count):
            cpu = first + offset
            cpus.append(cpu)
            core[cpu] = cpu
            kind[cpu] = level
            l2[cpu] = first + (offset // max(per_l2, 1)) * per_l2
    every = dict.fromkeys(cpus, 0)
    return Topology(
        cpus=tuple(cpus),
        core=core,
        socket=every,
        numa=every,
        l2=l2,
        l3=every,
        kind=kind,
        model=_macos_model(),
    )


def _macos_model() -> str:  # pragma: no cover - macOS
    import ctypes

    size = ctypes.c_size_t(0)
    name = b"machdep.cpu.brand_string"
    try:
        query = ctypes.CDLL(None).sysctlbyname
    except AttributeError, OSError:
        return platform.processor()
    if query(name, None, ctypes.byref(size), None, 0) != 0:
        return platform.processor()
    buffer = ctypes.create_string_buffer(size.value)
    if query(name, buffer, ctypes.byref(size), None, 0) != 0:
        return platform.processor()
    return buffer.value.decode(errors="replace")


def _model(system: str) -> str:
    if system == "Linux":
        return _linux_model()
    return platform.processor() or platform.machine()


def _windows() -> Topology | None:  # pragma: no cover - Windows
    """Read ``GetLogicalProcessorInformation``'s first processor group."""
    if sys.platform != "win32":
        return None
    import ctypes

    class _Cache(ctypes.Structure):
        _fields_ = (
            ("level", ctypes.c_ubyte),
            ("associativity", ctypes.c_ubyte),
            ("line_size", ctypes.c_ushort),
            ("size", ctypes.c_ulong),
            ("type", ctypes.c_int),
        )

    class _Union(ctypes.Union):
        _fields_ = (
            ("flags", ctypes.c_ubyte),
            ("node", ctypes.c_ulong),
            ("cache", _Cache),
            ("reserved", ctypes.c_ulonglong * 2),
        )

    class _Information(ctypes.Structure):
        _fields_ = (
            ("mask", ctypes.c_size_t),
            ("relationship", ctypes.c_int),
            ("union", _Union),
        )

    query = ctypes.windll.kernel32.GetLogicalProcessorInformation
    size = ctypes.c_ulong(0)
    query(None, ctypes.byref(size))
    table = (_Information * (size.value // ctypes.sizeof(_Information)))()
    if not query(table, ctypes.byref(size)):
        return None

    def members(mask: int) -> list[int]:
        return [bit for bit in range(mask.bit_length()) if mask >> bit & 1]

    core: dict[int, int] = {}
    socket: dict[int, int] = {}
    numa: dict[int, int] = {}
    l2: dict[int, int] = {}
    l3: dict[int, int] = {}
    packages = 0
    for entry in table:
        cpus = members(entry.mask)
        if not cpus:
            continue
        if entry.relationship == 0:  # RelationProcessorCore
            for cpu in cpus:
                core[cpu] = min(cpus)
        elif entry.relationship == 1:  # RelationNumaNode
            for cpu in cpus:
                numa[cpu] = int(entry.union.node)
        elif entry.relationship == 2:  # RelationCache
            target = {2: l2, 3: l3}.get(int(entry.union.cache.level))
            if target is not None:
                for cpu in cpus:
                    target[cpu] = min(cpus)
        elif entry.relationship == 3:  # RelationProcessorPackage
            for cpu in cpus:
                socket[cpu] = packages
            packages += 1
    all_cpus = tuple(sorted(core))
    if not all_cpus:
        return None
    return Topology(
        cpus=all_cpus,
        core=core,
        socket={cpu: socket.get(cpu, 0) for cpu in all_cpus},
        numa={cpu: numa.get(cpu, 0) for cpu in all_cpus},
        l2={cpu: l2.get(cpu, core[cpu]) for cpu in all_cpus},
        l3={cpu: l3.get(cpu, 0) for cpu in all_cpus},
        model=platform.processor(),
    )
