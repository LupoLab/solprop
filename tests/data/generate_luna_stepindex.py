# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate corrected and stale Luna step-index witnesses in a read-only Julia child.

Run with ``uv run python tests/data/generate_luna_stepindex.py``. The corrected
solver brackets Luna's characteristic branches between Bessel poles, with the TE/TM
names exchanged. Source files in the Luna checkout are never modified.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
OUTPUT = Path(__file__).with_name("luna_stepindex.json")
PROGRAM = r"""
using Luna
const LUNA_DIR = get(ENV, "LUNA_DIR", joinpath(homedir(), ".julia/dev/Luna"))
@assert realpath(pathof(Luna)) == realpath(joinpath(LUNA_DIR, "src", "Luna.jl"))
const S = Luna.StepIndexFibre
const besselj = S.besselj
const besselk = S.besselk
const Roots = parentmodule(S.find_zeros)
const find_zero = Roots.find_zero
const Bisection = Roots.Bisection
function corrected(a,k,nc,nl,n,m,kind)
    branch = kind == :TE ? S.char_TM : kind == :TM ? S.char_TE :
             kind == :HE ? S.char_HE : S.char_EH
    v = a*k*sqrt(nc^2-nl^2)
    index(u) = sqrt(nc^2-(u/(a*k))^2)
    char(u) = n == 0 ? branch(a,k,nc,nl,index(u)) : branch(a,k,nc,nl,n,index(u))
    # Roots of J_n are boundaries, never sign changes fed to the characteristic.
    mesh = range(1e-7, v*(1-1e-10), length=max(1000,ceil(Int,v/0.01)))
    poles = Float64[]
    for (left,right) in zip(mesh[1:end-1],mesh[2:end])
        if signbit(besselj(n,left)) != signbit(besselj(n,right))
            push!(poles,find_zero(u->besselj(n,u),(left,right),Bisection()))
        end
    end
    edges = [1e-7; poles; v*(1-1e-10)]
    roots = Float64[]
    for (lo,hi) in zip(edges[1:end-1],edges[2:end])
        nodes = range(lo+1e-8*(hi-lo),hi-1e-8*(hi-lo),length=129)
        for (left0,right0) in zip(nodes[1:end-1],nodes[2:end])
            left,right=left0,right0
            fl,fr=char(left),char(right)
            if isfinite(fl) && isfinite(fr) && signbit(fl) != signbit(fr)
                for _ in 1:80
                    max(abs(fl),abs(fr)) < 1 && break
                    mid=(left+right)/2
                    fm=char(mid)
                    if signbit(fm)==signbit(fl)
                        left,fl=mid,fm
                    else
                        right,fr=mid,fm
                    end
                end
                max(abs(fl),abs(fr)) < 1 || error("unqualified branch bracket")
                push!(roots,index(find_zero(char,(left,right),Bisection())))
            end
        end
    end
    sort!(roots,rev=true)
    return roots[m]
end
for line in eachline(stdin)
    parts=split(line,'\t')
    id=parse(Int,parts[1]); a,nc,nl,wl=parse.(Float64,parts[2:5])
    kind=Symbol(parts[6]); n,m=parse.(Int,parts[7:8])
    omega=2pi*Luna.PhysData.c/wl
    fixed=corrected(a,omega/Luna.PhysData.c,nc,nl,n,m,kind)
    stale=S.findneff(a,omega/Luna.PhysData.c,nc,nl,n,m,kind)
    println("I\t",id,"\t",fixed,"\t",stale)
    # Use Luna's field construction with the independently corrected eigenpair.
    if n>0
        profile_omega=2pi*Luna.PhysData.c/1030e-9
        v=a*profile_omega/Luna.PhysData.c*sqrt(nc^2-nl^2)
        try
            nr=corrected(a,profile_omega/Luna.PhysData.c,nc,nl,n,m,kind)
            f1,f2,u,w,v=S.f1f2(a,profile_omega/Luna.PhysData.c,nc,nl,nr,n)
            for (p,r) in enumerate((0.,0.1,0.3,0.5,0.8,1.,1.01,1.2,2.,3.))
                theta=0.37
                args=((f2-1)/2,(f2+1)/2,cos(n*theta),-sin(n*theta))
                er,et = r<=1 ? S.corefields(n,u,r,args...) :
                               S.cladfields(n,u,w,r,args...)
                println("F\t",id,"\t",p,"\t",r,"\t",
                        er*cos(theta)-et*sin(theta),"\t",er*sin(theta)+et*cos(theta))
            end
        catch exception
            # A profile can be below cutoff even though the row wavelength guides.
            exception isa BoundsError || rethrow()
        end
    end
end
"""


def main():
    """Run the Julia child and write its numeric output with source provenance."""
    rows = json.loads(Path(__file__).with_name("symbolic_stepindex.json").read_text())[
        "cases"
    ]
    rows = [row for row in rows if not row["material_slope"]]
    request = "\n".join(
        "\t".join(
            map(
                str,
                (
                    index,
                    row["radius"],
                    row["core"],
                    row["cladding"],
                    row["wavelength"],
                    row["kind"],
                    row["n"],
                    row["m"],
                ),
            )
        )
        for index, row in enumerate(rows)
    )
    child = subprocess.run(
        ["julia", "--startup-file=no", "-e", PROGRAM],
        input=request,
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stderr.write(child.stderr)
    if child.returncode:
        return child.returncode
    for line in child.stdout.splitlines():
        parts = line.split("\t")
        row = rows[int(parts[1])]
        if parts[0] == "I":
            row["corrected"] = float(parts[2])
            row["stale"] = float(parts[3])
        else:
            row.setdefault("fields", []).append(
                [float(parts[3]), float(parts[4]), float(parts[5])]
            )
    commit = subprocess.run(
        ["git", "-C", str(LUNA), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(LUNA), "branch", "--show-current"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    julia_version = subprocess.run(
        ["julia", "--startup-file=no", "--version"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    OUTPUT.write_text(
        json.dumps(
            {
                "provenance": {
                    "luna_path": str(LUNA),
                    "luna_commit": commit,
                    "luna_branch": branch,
                    "julia_version": julia_version,
                    "generated": datetime.now(UTC).isoformat(),
                    "generator": Path(__file__).name,
                    "corrections": (
                        "TE/TM exchange and independent pole-excluding bracketing; "
                        "hybrid fields use corrected roots; TE/TM hybrid field "
                        "singularity (a zero tangential field for n = 0) not "
                        "carried over"
                    ),
                },
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    sys.stdout.write(f"wrote {len(rows)} Luna cases to {OUTPUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
