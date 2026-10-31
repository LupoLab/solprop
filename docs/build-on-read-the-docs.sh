#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Build the manual on Read the Docs, ending before the service's time limit.
#
# Read the Docs stops a build at its time limit (about 1500 s, set-up included) and
# discards its log. Since 2026-10-29 some builds have run into it, though the same tools
# build in about 600 s otherwise. So Sphinx runs in a session of its own with its log in
# a file, and this script always ends first: at WATCHDOG seconds Sphinx writes every
# thread's stack to the log (SOLPROP_DOCS_WATCHDOG, docs/conf.py); at LIMIT seconds the
# process table is printed with each process's state and kernel wait, the session is
# killed and the log printed. A failed or stalled build is not retried.
#
# Usage: docs/build-on-read-the-docs.sh PYTHON OUTPUT_DIR

set -u

python=$1
output=$2
watchdog=1200
limit=1260
log=$(mktemp)

echo "builder: $(nproc) CPUs"
free -m

SOLPROP_DOCS_WATCHDOG=$watchdog setsid "$python" -m sphinx -b html -W -n --keep-going \
    docs "$output" >"$log" 2>&1 &
pid=$!

while kill -0 "$pid" 2>/dev/null && ((SECONDS < limit)); do
    sleep 5
done

if kill -0 "$pid" 2>/dev/null; then
    echo "sphinx-build still running after $SECONDS s; load $(cut -d' ' -f1-3 /proc/loadavg)"
    free -m
    ps -eo pid,ppid,pgid,stat,etime,time,wchan:24,args --sort=pid
    kill -KILL -- "-$pid" 2>/dev/null
    cat "$log"
    echo "sphinx-build stopped after $SECONDS s; the stacks and the process table above show where"
    exit 124
fi

wait "$pid"
status=$?
cat "$log"
exit "$status"
