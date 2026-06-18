#!/usr/bin/env python3
"""
run_experiments.py
==================

Driver + parser + plotter for the IETI-DP vs (geometric) multigrid comparison on
isogeometric Poisson problems, built on ``solver_benchmark_example``.

Goal of these experiments
-------------------------
We look for a discretization regime in which the IETI-DP solver is already
competitive with -- or better than -- a multigrid-preconditioned Krylov method,
to be used as the *reference case* for the neural-preconditioner study. We then
sweep, one at a time, the three discretization parameters that control the IETI
subdomain structure and the local problem size:

  * ``SplitPatches``  -- number of subdomains (each uniform split -> x4 patches),
  * ``Degree``        -- local polynomial degree p in every patch,
  * ``Refinements``   -- uniform h-refinement, i.e. the mesh resolution per patch,

and record the setup/solve time of IETI-DP and of CG+multigrid.

The measured quantity that matters for the neural preconditioner is the *solve*
time: the learned operator replaces the local Dirichlet Schur complement applied
once per CG iteration, i.e. it acts inside the solve phase, not the assembly.

Paths
-----
By default the script assumes it lives in the G+Smo ``examples/`` directory, so
the compiled binary is at ``../build/bin/solver_benchmark_example`` and the
geometries are under ``../filedata/``. Override with the environment variables
``GISMO_BUILD`` (build directory) and ``GISMO_FILEDATA`` (filedata root directory)
if your layout differs.

Usage
-----
    python3 run_experiments.py [--domain teapot|yeti] run     # run binary -> results_{domain}.json
    python3 run_experiments.py [--domain teapot|yeti] plot    # generate figures from results_{domain}.json
    python3 run_experiments.py [--domain teapot|yeti] all     # run then plot (default)

Default domain is ``teapot`` for backward compatibility.
Re-running ``run`` overwrites results_{domain}.json; ``plot`` only needs the JSON.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# examples/ -> repo root is one level up.
ROOT = os.path.normpath(os.path.join(HERE, ".."))
BUILD = os.environ.get("GISMO_BUILD", os.path.join(ROOT, "build"))
BIN = os.path.join(BUILD, "bin", "solver_benchmark_example")
FILEDATA = os.environ.get("GISMO_FILEDATA", os.path.join(ROOT, "filedata"))

TEAPOT      = os.path.join(FILEDATA, "surfaces", "teapot.xml")
YETI        = os.path.join(FILEDATA, "domain2d", "yeti_mp2.xml")
HUMMINGBIRD = os.path.join(FILEDATA, "surfaces", "hummingbird_nurbs_only_v2.igs")

FIG_DIR = os.path.join(ROOT, "latex_experiments", "figures")

# Restrict the global-system solvers to the multigrid baseline (CG + multigrid).
# IETI-DP is always run by the benchmark regardless of this list.
COMMON_ARGS = ["--Solvers", "CG,Multigrid", "--Preconditioners", "no prec,Jacobi,symm. Gauss-Seidel,multigrid"]

# ---------------------------------------------------------------------------
# Domain configurations
# ---------------------------------------------------------------------------
DOMAINS = {
    "teapot": dict(
        geometry=TEAPOT,
        reference=dict(geometry=TEAPOT, splitpatches=0, degree=2, refinements=3),
        plan=dict(
            refinement=[dict(geometry=TEAPOT, splitpatches=0, degree=2, refinements=r) for r in (1, 2, 3, 4)],
            degree    =[dict(geometry=TEAPOT, splitpatches=0, degree=p, refinements=2) for p in (1, 2, 3, 4)],
            splitpatches=[dict(geometry=TEAPOT, splitpatches=sp, degree=2, refinements=2) for sp in (0, 1, 2)],
        ),
        results_json=os.path.join(HERE, "results_teapot.json"),
        fig_prefix="",
        title="Teapot",
    ),
    "yeti": dict(
        geometry=YETI,
        reference=dict(geometry=YETI, splitpatches=0, degree=2, refinements=2),
        plan=dict(
            refinement=[dict(geometry=YETI, splitpatches=0, degree=2, refinements=r) for r in (1, 2, 3, 4)],
            degree    =[dict(geometry=YETI, splitpatches=0, degree=p, refinements=2) for p in (1, 2, 3, 4)],
            splitpatches=[dict(geometry=YETI, splitpatches=sp, degree=2, refinements=2) for sp in (0, 1, 2)],
        ),
        results_json=os.path.join(HERE, "results_yeti.json"),
        fig_prefix="yeti_",
        title="Yeti",
    ),
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# A benchmark table row: "<name>  <setup> <solve> <iters|-> <l2> <yes|NO>"
_ROW = re.compile(
    r"^(?P<name>.+?)\s+"
    r"(?P<setup>\d+\.\d+)\s+"
    r"(?P<solve>\d+\.\d+)\s+"
    r"(?P<iters>-|\d+)\s+"
    r"(?P<l2>[0-9.eE+\-]+)\s+"
    r"(?P<conv>yes|NO)\s*$"
)
_DOFS = re.compile(r"(\d+)\s+coupled dofs")
_PATCHES = re.compile(r"^(\d+)\s+patches")
_ASM = re.compile(r"done \(([0-9.]+) s,")
_IETI = re.compile(
    r"IETI sizes:\s*(\d+)\s+Lagrange multipliers.*?(\d+)\s+primal dofs,\s*(\d+)\s+local patch solves"
)


def parse_output(text):
    rec = {"dofs": None, "npatches": None, "asm_time": None,
           "ieti_lagrange": None, "ieti_primal": None, "ieti_solves": None,
           "methods": {}}
    for line in text.splitlines():
        m = _DOFS.search(line)
        if m:
            rec["dofs"] = int(m.group(1))
        m = _PATCHES.match(line.strip())
        if m:
            rec["npatches"] = int(m.group(1))
        m = _ASM.search(line)
        if m and rec["asm_time"] is None:
            rec["asm_time"] = float(m.group(1))
        m = _IETI.search(line)
        if m:
            rec["ieti_lagrange"] = int(m.group(1))
            rec["ieti_primal"] = int(m.group(2))
            rec["ieti_solves"] = int(m.group(3))
        m = _ROW.match(line.rstrip())
        if m:
            rec["methods"][m.group("name").strip()] = dict(
                setup=float(m.group("setup")),
                solve=float(m.group("solve")),
                iters=(None if m.group("iters") == "-" else int(m.group("iters"))),
                l2=float(m.group("l2")),
                converged=(m.group("conv") == "yes"),
            )
    return rec


# Fixed thread count for every run, so the comparison is "this 12-core machine".
# Timings are taken as the MIN over a few repeats: the true compute time is a
# lower bound that transient OS/CPU contention can only inflate, so the minimum
# is the least-contaminated estimate (and the iteration counts are deterministic).
N_THREADS = "12"
N_REPEATS = 3


def run_one(geometry, splitpatches, degree, refinements, timeout=600, extra=None):
    args = [BIN, "-g", geometry,
            "--SplitPatches", str(splitpatches),
            "-p", str(degree),
            "-r", str(refinements)] + COMMON_ARGS + (extra or [])
    env = dict(os.environ, OMP_NUM_THREADS=N_THREADS)
    label = f"{os.path.basename(geometry)} sp={splitpatches} p={degree} r={refinements}"
    print(f"  running {label} (x{N_REPEATS}) ...", flush=True)

    best = None
    for _ in range(N_REPEATS):
        t0 = time.time()
        try:
            out = subprocess.run(args, capture_output=True, text=True,
                                 timeout=timeout, env=env).stdout
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {timeout}s", flush=True)
            return dict(geometry=os.path.basename(geometry), splitpatches=splitpatches,
                        degree=degree, refinements=refinements, timed_out=True, methods={})
        rec = parse_output(out)
        rec["wall"] = round(time.time() - t0, 1)
        if best is None:
            best = rec
        else:
            # keep the per-method minimum setup and solve across repeats
            for name, m in rec["methods"].items():
                if name in best["methods"]:
                    best["methods"][name]["setup"] = min(best["methods"][name]["setup"], m["setup"])
                    best["methods"][name]["solve"] = min(best["methods"][name]["solve"], m["solve"])
                else:
                    best["methods"][name] = m

    best.update(geometry=os.path.basename(geometry), splitpatches=splitpatches,
                degree=degree, refinements=refinements, timed_out=False)
    ieti = best["methods"].get("IETI-DP (CG on Schur)", {})
    mg = best["methods"].get("CG + multigrid", {})
    cg_no = best["methods"].get("CG + no prec", {})
    print(f"    dofs={best['dofs']} patches={best['npatches']} "
          f"IETI={ieti.get('solve')} MG={mg.get('solve')} CG_no={cg_no.get('solve')}", flush=True)
    return best


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------
def collect(cfg):
    results = {"reference": cfg["reference"], "sweeps": {}}
    for name, configs in cfg["plan"].items():
        print(f"[sweep: {name}]", flush=True)
        results["sweeps"][name] = [run_one(**c) for c in configs]
    with open(cfg["results_json"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {cfg['results_json']}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _series(records, xkey, method):
    xs, solve, total = [], [], []
    for r in records:
        if r.get("timed_out") or method not in r["methods"]:
            continue
        xs.append(r[xkey])
        s = r["methods"][method]["solve"]
        setup = r["methods"][method]["setup"]
        solve.append(s)
        total.append(s + setup)
    return xs, solve, total


def plot(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    with open(cfg["results_json"]) as f:
        results = json.load(f)

    METHODS = [
        ("IETI-DP (CG on Schur)",   "o-",  "#c0392b", "IETI-DP"),
        ("CG + multigrid",          "s--", "#2c3e50", "CG + Multigrid"),
        ("CG + no prec",            "v:",  "#7f8c8d", "CG (no prec)"),
        ("CG + Jacobi",             "d:",  "#2980b9", "CG + Jacobi"),
        ("CG + symm. Gauss-Seidel", "^:",  "#8e44ad", "CG + GS sym"),
        ("Multigrid (standalone)",  "p-.", "#27ae60", "Multigrid alone"),
    ]

    sweep_meta = {
        "refinement":   ("refinements", "uniform refinement level $r$ (mesh resolution per patch)"),
        "degree":       ("degree",      "local polynomial degree $p$"),
        "splitpatches": ("splitpatches", "uniform splits (number of subdomains $\\propto 4^{\\,\\mathrm{splits}}$)"),
    }

    for sweep, (xkey, xlabel) in sweep_meta.items():
        recs = results["sweeps"][sweep]

        fig, ax = plt.subplots(figsize=(7, 5))

        for method_name, style, color, label in METHODS:
            xs, solve, total = _series(recs, xkey, method_name)
            if not xs:
                continue
            ax.plot(xs, solve, style, color=color, label=label)

        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("solve time [s]")
        # ax.set_title("Solve time")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize="small")

        xs_all = set()
        for method_name, _, _, _ in METHODS:
            xs, _, _ = _series(recs, xkey, method_name)
            xs_all.update(xs)
        if xs_all:
            ax.set_xticks(sorted(xs_all))

        # fig.suptitle(f"{cfg['title']}: Solver Comparison, sweep over {sweep}")
        fig.tight_layout()
        path = os.path.join(FIG_DIR, f"{cfg['fig_prefix']}time_vs_{sweep}.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"), dpi=140)
        plt.close(fig)
        print(f"wrote {path}")

    print("done plotting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run and/or plot IETI-DP vs multigrid sweep experiments."
    )
    parser.add_argument(
        "--domain", choices=list(DOMAINS), default="teapot",
        help="geometry domain to use (default: teapot)",
    )
    parser.add_argument(
        "cmd", nargs="?", default="all", choices=["run", "plot", "all"],
        help="run: execute binary; plot: generate figures; all: both (default)",
    )
    args = parser.parse_args()
    cfg = DOMAINS[args.domain]
    if args.cmd in ("run", "all"):
        collect(cfg)
    if args.cmd in ("plot", "all"):
        plot(cfg)
