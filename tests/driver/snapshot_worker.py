# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise real process signals and fresh TDSE restoration outside pytest's process."""

import json
import os
import signal
import sys
from pathlib import Path

import pytest
from tests.driver.test_snapshots import Requests, _run, _simulation
from tests.hacc_cases import alpha_law, install, plain

from solprop.driver import run as run_module
from solprop.exceptions import RunTerminated
from solprop.output import (
    HDF5Output,
    SnapshotController,
    TDSEArchive,
    read_snapshot,
    restore_snapshot,
    resume,
)
from solprop.output import snapshots as module
from solprop.output.tdse_archive import TDSEArchiveWriter


def install_signal_phase(patch, phase, signal_once):
    if phase == "checkpoint":
        save_march = HDF5Output.save_march
        # Force history persistence into the snapshot transaction itself.
        patch.setattr(run_module, "MARCH_SNAPSHOT_AUTO_BYTES", 0)

        def write(self, snapshot):
            signal_once()
            return save_march(self, snapshot)

        patch.setattr(HDF5Output, "save_march", write)
    elif phase in ("copy", "interrupt", "kill"):
        original_copy = module._copy_file
        copies = []

        def copy(*args):
            copies.append(True)
            if phase == "kill":
                if len(copies) == 3:
                    os.kill(os.getpid(), signal.SIGKILL)
            else:
                signal_once()
            return original_copy(*args)

        patch.setattr(module, "_copy_file", copy)
    else:
        original_reopen = TDSEArchiveWriter.reopen

        def reopen(self):
            signal_once()
            return original_reopen(self)

        patch.setattr(TDSEArchiveWriter, "reopen", reopen)


def main():
    """Deliver a real signal from a chosen transaction operation exactly once."""
    directory, phase = Path(sys.argv[1]), sys.argv[2]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SOLPROP_CACHE_DIR", str(directory / "cache"))
        fake = install(patch)
        fake.response = alpha_law(1.0, lambda5=2.0**-200)
        if phase == "restore":
            files = restore_snapshot(sys.argv[3], directory / "restored")
            before = resume(files.output)
            _run(
                _simulation(),
                files.output,
                resume=True,
                tdse_archive=files.tdse_archive,
            )
            after = resume(files.output)
            assert before.outcome is not None and after.outcome is not None
            assert after.outcome.status == "complete"
            assert (
                plain(after.outcome.atomic_calls)["attempted"]
                >= plain(before.outcome.atomic_calls)["attempted"]
            )
            return
        controller = SnapshotController(directory / "snapshots")
        calls_at_signal = []

        def signal_once():
            if not calls_at_signal:
                calls_at_signal.append(len(fake.calls))
                os.kill(
                    os.getpid(),
                    signal.SIGINT if phase == "interrupt" else signal.SIGTERM,
                )

        install_signal_phase(patch, phase, signal_once)
        try:
            _run(
                _simulation(),
                directory / "working.h5",
                snapshots=controller,
                tdse_archive=TDSEArchive(directory / "working.tdse.h5"),
                progress=Requests(controller),
            )
        except RunTerminated, KeyboardInterrupt:
            pass
        else:
            raise AssertionError("signal did not stop the run")
        assert len(fake.calls) == calls_at_signal[0]
        assert controller.last_result is not None
        assert controller.last_result.generation is not None
        assert controller.last_result.disposition == "published"
        info = read_snapshot(controller.last_result.generation)
        stopped = resume(directory / "working.h5")
        assert stopped.outcome is not None
        assert stopped.outcome.status == "failed" and stopped.march is not None
        assert info.z == stopped.outcome.z
        (directory / "signal.json").write_text(
            json.dumps(
                {
                    "calls": len(fake.calls),
                    "accounting": plain(stopped.outcome.atomic_calls),
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
