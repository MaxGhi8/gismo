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

Timings
-------
Each configuration is run N_REPEATS=10 times.  The JSON stores, for each method,
top-level ``setup`` / ``solve`` / ``iters`` scalars (means, for backward compat)
plus ``*_stats`` sub-dicts with mean, median, min, max, std, and n for setup, solve,
total (setup+solve), and iterations.  All raw repetition records are kept under the
``runs`` key for later variance analysis.

Figures
-------
All sweeps are plotted with N (number of coupled DOFs) on the x-axis in log scale
so that asymptotic slopes are visible (log-log for times, semilog-x for iterations).
Separate *iters_vs_* figures are saved for each sweep showing iteration counts vs N.

Theoretical scalings (2-D surface problems)
--------------------------------------------
Refinement sweep  (fixed degree p, mesh size h -> 0,  N ~ h^{-2}):

  GMRES (no prec):     kappa ~ h^{-2} ~ N       iters ~ N^{1/2}    time ~ N^{3/2}
  GMRES + Jacobi:      same asymptotics, better constant
  GMRES + symm. GS:    same asymptotics, better constant than Jacobi
  CG (no prec):        kappa ~ h^{-2} ~ N       iters ~ N^{1/2}    time ~ N^{3/2}
  CG + Jacobi:         same asymptotics, better constant
  CG + symm. GS:       same asymptotics, better constant than Jacobi
  IETI-DP:             kappa ~ (1 + log(H/h))^2 ~ log(N)^2
                       iters ~ O(log N)         time ~ O(N log N)

Degree sweep  (fixed h, polynomial degree p -> inf,  N ~ p^2):

  CG (no prec):        kappa grows super-algebraically in p
  CG + Multigrid:      not p-robust -- deteriorates with p
  IETI-DP:             kappa ~ O(p^3);  primal space must cover vertex dofs

Subdomain sweep  (fixed p, h per patch;  K ~ 4^splits subdomains):

  CG + Multigrid:      governed by global N_total
  IETI-DP:             kappa ~ (1 + log(H/h))^2  improves as K grows (H decreases)
"""

import argparse
import json
import math
import os
import re
import statistics
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
COMMON_ARGS = ["--Solvers", "CG,GMRES,Multigrid", "--Preconditioners", "no prec,Jacobi,symm. Gauss-Seidel"]

# ---------------------------------------------------------------------------
# Domain configurations
# ---------------------------------------------------------------------------
DOMAINS = {
    "teapot": dict(
        geometry=TEAPOT,
        reference=dict(geometry=TEAPOT, splitpatches=0, degree=2, refinements=4),
        plan=dict(
            refinement   =[dict(geometry=TEAPOT, splitpatches=0, degree=2, refinements=r) for r in (2, 3, 4, 5, 6)],
            degree       =[dict(geometry=TEAPOT, splitpatches=0, degree=p, refinements=2) for p in (1, 2, 3, 4, 5, 6)],
            splitpatches =[dict(geometry=TEAPOT, splitpatches=sp, degree=2, refinements=2) for sp in (0, 1, 2, 3, 4)],
        ),
        results_json=os.path.join(HERE, "results_teapot.json"),
        fig_prefix="",
        title="Teapot",
    ),
    "yeti": dict(
        geometry=YETI,
        reference=dict(geometry=YETI, splitpatches=0, degree=2, refinements=4),
        plan=dict(
            refinement   =[dict(geometry=YETI, splitpatches=0, degree=2, refinements=r) for r in (2, 3, 4, 5, 6)],
            degree       =[dict(geometry=YETI, splitpatches=0, degree=p, refinements=2) for p in (1, 2, 3, 4, 5, 6)],
            splitpatches =[dict(geometry=YETI, splitpatches=sp, degree=2, refinements=2) for sp in (0, 1, 2, 3, 4)],
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


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
# Fixed thread count for every run.  Timings are taken as the MEAN over
# N_REPEATS=5 repeats; standard deviations are saved for later analysis but
# are not plotted.  Iteration counts are deterministic so their std will be 0.
N_THREADS = "12"
N_REPEATS = 10


def _stats(vals):
    """Return a dict of descriptive statistics for a list of floats."""
    n = len(vals)
    if n == 0:
        return {"mean": None, "median": None, "min": None, "max": None,
                "std": None, "n": 0}
    mean   = sum(vals) / n
    median = statistics.median(vals)
    return {
        "mean":   mean,
        "median": median,
        "min":    min(vals),
        "max":    max(vals),
        "std":    statistics.stdev(vals) if n > 1 else 0.0,
        "n":      n,
    }


def compute_mean_rec(all_recs):
    """Aggregate timing/iteration fields across repeated runs."""
    out = {k: all_recs[0].get(k) for k in (
        "dofs", "npatches", "asm_time",
        "ieti_lagrange", "ieti_primal", "ieti_solves")}
    out["methods"] = {}
    method_names = {name for r in all_recs for name in r.get("methods", {})}
    for name in method_names:
        vals = [r["methods"][name] for r in all_recs if name in r.get("methods", {})]
        if not vals:
            continue
        setup_vals = [v["setup"] for v in vals]
        solve_vals = [v["solve"] for v in vals]
        total_vals = [v["setup"] + v["solve"] for v in vals]
        iters_list = [v["iters"] for v in vals if v.get("iters") is not None]
        l2_vals    = [v["l2"]    for v in vals if v.get("l2")    is not None]
        n_conv     = sum(1 for v in vals if v.get("converged"))
        setup_s = _stats(setup_vals)
        solve_s = _stats(solve_vals)
        total_s = _stats(total_vals)
        iters_s = _stats(iters_list) if iters_list else None
        l2_s    = _stats(l2_vals)    if l2_vals    else None
        out["methods"][name] = {
            # primary scalars (means) kept at top level for backward compat
            "setup":          setup_s["mean"],
            "solve":          solve_s["mean"],
            "iters":          iters_s["mean"] if iters_s else None,
            "l2":             l2_s["mean"]    if l2_s    else None,
            "converged":      n_conv == len(vals),
            # convergence bookkeeping
            "n_converged":    n_conv,
            "n_runs":         len(vals),
            "converge_rate":  n_conv / len(vals),
            # full stats sub-dicts
            "setup_stats":    setup_s,
            "solve_stats":    solve_s,
            "total_stats":    total_s,
            "iters_stats":    iters_s,
            "l2_stats":       l2_s,
        }
    return out


def run_one(geometry, splitpatches, degree, refinements, timeout=600, extra=None):
    args = [BIN, "-g", geometry,
            "--SplitPatches", str(splitpatches),
            "-p", str(degree),
            "-r", str(refinements)] + COMMON_ARGS + (extra or [])
    env = dict(os.environ, OMP_NUM_THREADS=N_THREADS)
    label = f"{os.path.basename(geometry)} sp={splitpatches} p={degree} r={refinements}"
    print(f"  running {label} (x{N_REPEATS}) ...", flush=True)

    all_recs = []
    for rep in range(N_REPEATS):
        t0 = time.time()
        try:
            out = subprocess.run(args, capture_output=True, text=True,
                                 timeout=timeout, env=env).stdout
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {timeout}s (rep {rep+1})", flush=True)
            return dict(geometry=os.path.basename(geometry), splitpatches=splitpatches,
                        degree=degree, refinements=refinements, timed_out=True,
                        methods={}, runs=[])
        except FileNotFoundError:
            # Binary may be momentarily unavailable (rebuild in progress); retry once.
            time.sleep(5)
            out = subprocess.run(args, capture_output=True, text=True,
                                 timeout=timeout, env=env).stdout
        rec = parse_output(out)
        rec["wall"] = round(time.time() - t0, 1)
        all_recs.append(rec)

    mean_rec = compute_mean_rec(all_recs)
    mean_rec.update(geometry=os.path.basename(geometry), splitpatches=splitpatches,
                    degree=degree, refinements=refinements, timed_out=False)
    mean_rec["runs"] = all_recs  # all 5 raw records for later variance analysis

    ieti    = mean_rec["methods"].get("IETI-DP (CG on Schur)", {})
    mg      = mean_rec["methods"].get("Multigrid (standalone)", {})
    cg_no   = mean_rec["methods"].get("CG + no prec", {})
    gmres_no = mean_rec["methods"].get("GMRES + no prec", {})
    _fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
    print(f"    dofs={mean_rec['dofs']} patches={mean_rec['npatches']} "
          f"IETI={_fmt(ieti.get('solve'))} MG={_fmt(mg.get('solve'))} "
          f"CG_no={_fmt(cg_no.get('solve'))} GMRES_no={_fmt(gmres_no.get('solve'))}", flush=True)
    return mean_rec


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
def _series(records, method, xkey="dofs"):
    """Extract (x, solve_time, total_time, iters) for one method across records."""
    xs, solve, total, iters_list = [], [], [], []
    for r in records:
        if r.get("timed_out") or method not in r.get("methods", {}):
            continue
        x_val = r.get(xkey)
        if x_val is None:
            continue
        xs.append(x_val)
        m = r["methods"][method]
        solve.append(m["solve"])
        total.append(m["solve"] + m["setup"])
        iters_list.append(m.get("iters"))
    return xs, solve, total, iters_list


_SLOPE_COLOR = "darkorange"

def _slope_line(ax, slope, label, xs_data, ys_data,
                y_frac=0.12, x_margin=0.06):
    """Draw a reference-slope segment spanning the full x-range with small margins.

    The line starts at x_margin (fraction of log-range) from the left edge and
    ends at x_margin from the right edge.  y_frac sets the vertical position of
    the line's left endpoint relative to the data's log y-range.
    """
    xs_pos = [x for x in xs_data if x and x > 0]
    ys_pos = [y for y in ys_data if y and y > 0]
    if not xs_pos or not ys_pos:
        return
    lx_min, lx_max = math.log10(min(xs_pos)), math.log10(max(xs_pos))
    ly_min, ly_max = math.log10(min(ys_pos)), math.log10(max(ys_pos))
    lx_span = lx_max - lx_min
    lx0 = lx_min + x_margin * lx_span
    lx1 = lx_max - x_margin * lx_span
    if lx1 <= lx0:
        return
    ly0 = ly_min + y_frac * (ly_max - ly_min)
    ly1 = ly0 + slope * (lx1 - lx0)
    # skip if line exits the data range vertically
    if ly1 > ly_max + 0.6 or ly1 < ly_min - 0.6:
        return
    ax.plot([10**lx0, 10**lx1], [10**ly0, 10**ly1], '-',
            color=_SLOPE_COLOR, lw=1.1, alpha=0.55, zorder=0)
    ax.annotate(label, xy=(10**lx1, 10**ly1), xytext=(4, 0),
                textcoords='offset points', fontsize=7, color=_SLOPE_COLOR,
                va='center', alpha=0.9)


def plot(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    with open(cfg["results_json"]) as f:
        results = json.load(f)

    METHODS = [
        ("IETI-DP (CG on Schur)",       "o-",  "#c0392b", "IETI-DP"),
        ("GMRES + no prec",              "v--", "#aab7b8", "GMRES (no prec)"),
        ("GMRES + Jacobi",               "d--", "#1a5276", "GMRES + Jacobi"),
        ("GMRES + symm. Gauss-Seidel",  "^--", "#6c3483", "GMRES + GS sym"),
        ("CG + no prec",                 "v:",  "#7f8c8d", "CG (no prec)"),
        ("CG + Jacobi",                  "d:",  "#2980b9", "CG + Jacobi"),
        ("CG + symm. Gauss-Seidel",     "^:",  "#8e44ad", "CG + GS sym"),
        ("Multigrid (standalone)",       "p-.", "#27ae60", "Multigrid"),
    ]

    # Reference slopes: (label, exponent, y_frac).
    # y_frac positions the line's left endpoint in the data's log y-range [0=bottom, 1=top].
    # All lines span the full x-range (minus a small margin) and use _SLOPE_COLOR.
    SLOPE_REFS = {
        "refinement": {
            "time":  [("$\\propto N$",       1.0, 0.08),
                      ("$\\propto N^{3/2}$", 1.5, 0.32)],
            "iters": [("$\\propto N^0$",     0.0, 0.10),
                      ("$\\propto N^{1/2}$", 0.5, 0.52)],
        },
        "degree": {
            "time":  [("$\\propto N$",       1.0, 0.04),
                      ("$\\propto N^2$",     2.0, 0.30),
                      ("$\\propto N^3$",     3.0, 0.60)],
            "iters": [("$\\propto N^{1/2}$", 0.5, 0.15),
                      ("$\\propto N^{3/2}$", 1.5, 0.62)],
        },
        "splitpatches": {
            "time":  [("$\\propto N$",       1.0, 0.08),
                      ("$\\propto N^2$",     2.0, 0.38)],
            "iters": [("$\\propto N^0$",     0.0, 0.10),
                      ("$\\propto N^{1/2}$", 0.5, 0.52)],
        },
    }

    sweeps = ["refinement", "degree", "splitpatches"]

    for sweep in sweeps:
        if sweep not in results.get("sweeps", {}):
            continue
        recs = results["sweeps"][sweep]

        # ---- solve-time figure: log-log, x = N (DOFs) ----
        fig, ax = plt.subplots(figsize=(7, 5))
        all_xs, all_ys = [], []
        for method_name, style, color, label in METHODS:
            xs, solve, _, _ = _series(recs, method_name, xkey="dofs")
            if not xs:
                continue
            ax.plot(xs, solve, style, color=color, label=label)
            all_xs.extend(xs)
            all_ys.extend(solve)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of DOFs $N$")
        ax.set_ylabel("solve time [s]  (mean of 5 runs)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize="small")
        for lbl, exp, yf in SLOPE_REFS.get(sweep, {}).get("time", []):
            _slope_line(ax, exp, lbl, all_xs, all_ys, yf)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, f"{cfg['fig_prefix']}time_vs_{sweep}.pdf")
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"), dpi=140)
        plt.close(fig)
        print(f"wrote {path}")

        # ---- iteration-count figure: log-log, x = N (DOFs) ----
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        has_data = False
        all_xs2, all_ys2 = [], []
        for method_name, style, color, label in METHODS:
            xs, _, _, iters_list = _series(recs, method_name, xkey="dofs")
            valid = [(x, it) for x, it in zip(xs, iters_list)
                     if it is not None and it > 0]
            if not valid:
                continue
            xi, yi = zip(*valid)
            ax2.plot(xi, yi, style, color=color, label=label)
            all_xs2.extend(xi)
            all_ys2.extend(yi)
            has_data = True
        if has_data:
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel("number of DOFs $N$")
            ax2.set_ylabel("iteration count (mean of 5 runs)")
            ax2.grid(True, which="both", ls=":", alpha=0.5)
            ax2.legend(fontsize="small")
            for lbl, exp, yf in SLOPE_REFS.get(sweep, {}).get("iters", []):
                _slope_line(ax2, exp, lbl, all_xs2, all_ys2, yf)
            fig2.tight_layout()
            path2 = os.path.join(FIG_DIR, f"{cfg['fig_prefix']}iters_vs_{sweep}.pdf")
            fig2.savefig(path2)
            fig2.savefig(path2.replace(".pdf", ".png"), dpi=140)
            print(f"wrote {path2}")
        plt.close(fig2)

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
