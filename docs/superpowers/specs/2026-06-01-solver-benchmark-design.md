# Solver Benchmark Example — Design

**Date:** 2026-06-01
**File to create:** `examples/solver_benchmark_example.cpp`

## Goal

Solve **one** fixed Poisson problem on **one** fixed isogeometric discretization with
many different solvers, and report a fair side-by-side comparison (iterations,
timing, accuracy). The whole point is fairness: every solver sees the same
geometry, the same basis, the same boundary conditions, and the same stopping
criteria.

The problem mirrors `ieti_example.cpp`:

- PDE: `-Δu = f` on the multipatch domain.
- Right-hand side: `f = 2 sin(x) cos(y)`.
- Dirichlet data: `u = sin(x) cos(y)` on boundaries marked `d`; optional Neumann
  on `n`.
- **Manufactured exact solution:** `u_exact = sin(x) cos(y)` (since
  `-Δ(sin x cos y) = 2 sin x cos y`). This gives a numbering-independent
  accuracy yardstick shared by every solver.

Coefficient is fixed to 1 (standard Poisson), so the global system can be
assembled with `gsPoissonAssembler`, which pairs natively with
`gsGridHierarchy` for multigrid.

## Solvers benchmarked

1. **IETI-DP** — CG on the Schur complement with the scaled Dirichlet
   preconditioner (the `ieti_example.cpp` method). Reference domain-decomposition
   solver.
2. **Preconditioned CG on the global system**, with:
   - no preconditioner (`gsIdentityOp`),
   - Jacobi (`makeJacobiOp`),
   - symmetric Gauss-Seidel = SSOR (`makeSymmetricGaussSeidelOp`; symmetric so
     CG stays valid),
   - Richardson (`makeRichardsonOp`),
   - ILU (`makeIncompleteLUOp`).
3. **Geometric multigrid** (`gsGridHierarchy::buildByCoarsening` +
   `gsMultiGridOp`, Gauss-Seidel smoother by default, CLI-selectable, Cholesky
   coarse solver):
   - standalone stationary iteration (`gsGradientMethod`),
   - as a preconditioner for CG.
4. **Direct sparse Cholesky** — timing baseline and correctness anchor.

`gismo` has no dedicated SOR class; Gauss-Seidel is SOR with ω=1 and symmetric
Gauss-Seidel is SSOR with ω=1, which is what the "SOR/Gauss-Seidel" request maps
to. We do not reinvent a relaxed SOR.

## Architecture

Single source of truth for the discretization. `main` builds, exactly once:

- `gsMultiPatch<> mp` (after split / stretch / degree / uniform refine /
  optional square-equalization block carried over verbatim from
  `ieti_example.cpp`),
- `gsMultiBasis<> mb`,
- `gsBoundaryConditions<> bc`.

Every solver consumes copies of these. The global system `K x = F` is assembled
once via `gsPoissonAssembler` (elimination Dirichlet strategy). All non-IETI
solvers operate on that single `K, F`.

### Components (helpers in the one file)

- `struct BenchResult { std::string name; index_t iters; double setupTime,
  solveTime; real_t l2err; bool converged; };` — common result record. `iters`
  is `-1` for the direct solver.
- `assembleGlobal(mp, mb, bc, f) -> gsPoissonAssembler<>` — owns `K`, `F`, and
  `constructSolution` for field reconstruction.
- `l2ErrorVsExact(assembler, x, exactExpr) -> real_t` — reconstructs the field
  and integrates `‖u_h − u_exact‖_L2` with `gsExprEvaluator`. Numbering
  independent; used by every method including IETI.
- `runGlobalCG(name, K, F, prec, opt) -> (x, BenchResult)` — `gsConjugateGradient`
  on `K` with the given `gsLinearOperator` preconditioner, via `solveDetailed`
  (the error history length gives the iteration count). Reused for
  identity / Jacobi / sym-GS / Richardson / ILU and for MG-as-CG-preconditioner.
- `buildMultigrid(mb_copy, bc, cmd) -> gsMultiGridOp<>::Ptr` — hierarchy built
  once via `gsGridHierarchy::buildByCoarsening` on a copy of `mb`; Cholesky
  coarse solver; smoother set per level from the CLI choice. The returned
  operator is reused both standalone and as the CG preconditioner.
- `runMultigridStandalone(K, F, mg, opt)` — drives `mg` with `gsGradientMethod`.
- `runDirect(K, F)` — sparse Cholesky; produces reference timing and accuracy.
- `solveIeti(mp, mb, bc, f, cmd) -> (global field, BenchResult)` — the
  `ieti_example.cpp` per-patch assembly + scaled Dirichlet prec + CG-on-Schur
  loop, ported verbatim. Accuracy measured with the same `l2ErrorVsExact` on its
  reconstructed multipatch field.

### Data flow

```
main
 ├─ build mp / mb / bc  (once)
 ├─ assembleGlobal -> K, F
 ├─ runDirect              -> BenchResult  (+ reference field)
 ├─ runGlobalCG × {identity, Jacobi, symGS, Richardson, ILU}
 ├─ buildMultigrid; runMultigridStandalone; runGlobalCG(prec = mg)
 ├─ solveIeti              -> BenchResult
 └─ print aligned table
```

## Output

Console table only. Columns:

```
method | setup (s) | solve (s) | iters | L2 error | converged
```

A note printed below the table clarifies that IETI iterations count CG steps on
the Schur complement (a different operator and dimension), so iteration counts
are directly comparable only within the global-system group; the L2-error column
confirms all methods solved the same discretization.

## Fairness / correctness guarantees

- Identical `mp / mb / bc` for every solver.
- Identical `K, F` for every global solver.
- Identical `--Solver.Tolerance` and `--Solver.MaxIterations` for every iterative
  method; identical (identically seeded) random initial guess.
- All converged L2 errors should agree to discretization accuracy. A divergent
  L2 error flags a real bug, not an unfair comparison.

## CLI

Reuse the established flags: `-g/--Geometry`, `-r/--Refinements`, `-p/--Degree`,
`-b/--BoundaryConditions`, `-c/--Primals` (IETI), `--SplitPatches`,
`--StretchGeometry`, `-e/--EliminateCorners` (IETI), `--Solver.Tolerance`,
`--Solver.MaxIterations`, `--MG.*` (levels, cycles, pre/post-smooth, smoother).
No `--plot` needed for a benchmark; optional and low priority.

## Non-goals (YAGNI)

- No CSV export (console table only).
- No relaxed-ω SOR (use Gauss-Seidel / symmetric Gauss-Seidel).
- No subspace-corrected mass smoother (Gauss-Seidel smoother suffices; the
  complex SCMS setup from `multiGrid_example.cpp` is not ported).
- No coefficient `a(x)` (fixed to 1 for the standard Poisson operator).
