# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About G+Smo

G+Smo (Geometry plus Simulation modules) is a C++ library for isogeometric analysis (IGA). It bridges Computer-Aided Design (CAD) and Finite Element Analysis (FEA) using spline-based geometries as both the design and simulation domain.

## Build System

In-source builds are strictly forbidden. The project uses CMake; a convenience `Makefile` in the root directory wraps the standard CMake workflow:

```bash
# Configure and build (creates ./build automatically)
make

# Equivalent manual steps
mkdir build && cd build && cmake .. && make
```

Build artifacts land in `build/bin/` (executables) and `build/lib/` (shared library).

### Key CMake options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel` |
| `GISMO_BUILD_LIB` | `ON` | Build dynamic library |
| `GISMO_BUILD_EXAMPLES` | `ON` | Compile examples to `build/bin/` |
| `GISMO_BUILD_UNITTESTS` | `OFF` | Compile unit tests |
| `GISMO_COEFF_TYPE` | `double` | Arithmetic type: `double`, `long double`, `float`, `mpq_class`, `mpfr::mpreal` |
| `GISMO_OPTIONAL` | `` | Semicolon-separated optional modules (e.g., `gsSpectra;gsOpenCascade`) |
| `GISMO_WITH_XDEBUG` | `OFF` | Enable checked iterators and stack traces |

### Running examples and tests

```bash
# Run a specific example
./build/bin/bSplineCurve_example --plot

# Enable and run unit tests
cmake -DGISMO_BUILD_UNITTESTS=ON build/
make -C build unittests
cd build && ctest
# Or run the single test binary:
./build/bin/unittests

# Generate Doxygen docs
make doc  # output at build/doc/html/index.html
```

## Architecture

### Core abstraction hierarchy

The central abstraction is `gsFunction` → `gsGeometry`. A geometry is a basis paired with a coefficient matrix, mapping parameter space to physical space:

- `gsGeometry` (abstract) — base for all geometry types; is-a `gsFunction`
  - `gsCurve` — 1D parameter domain
  - `gsSurface` — 2D parameter domain
  - `gsVolume` / `gsBulk` — 3D/4D parameter domain

Each geometry has an associated basis type in a one-to-one static relationship (e.g., `gsBSplineBasis` ↔ `gsBSpline`, `gsTHBSplineBasis` ↔ `gsTHBSpline`).

### Source module layout (`src/`)

| Module | Purpose |
|---|---|
| `gsCore` | Base classes (`gsGeometry`, `gsBasis`, `gsFunction`, `gsFunctionSet`, `gsBoundary`), memory, debug macros |
| `gsMatrix` | Thin wrappers over Eigen; `gsMatrix<T>`, `gsVector<T>`, `gsSparseMatrix<T>` |
| `gsNurbs` | B-splines and NURBS: `gsBSpline`, `gsNurbs`, `gsKnotVector`, `gsBSplineBasis` |
| `gsHSplines` | Hierarchical splines: `gsHTensorBasis`, `gsTHBSpline` |
| `gsMSplines` | Multi-patch splines |
| `gsModeling` | Surface fitting, mesh processing, `gsFitting`, `gsTrimSurface` |
| `gsMesh2` | Half-edge surface mesh (`gsSurfMesh`), subdivision, I/O for OFF/OBJ/STL |
| `gsAssembler` | FEM assembler framework, boundary conditions (`gsBoundaryConditions`) |
| `gsExpressions` | Expression template system for weak-form assembly |
| `gsSolver` | Linear solvers, preconditioners |
| `gsPde` | PDE problem definitions |
| `gsTensor` | Tensor-product structure utilities |
| `gsHSplines` / `gsIeti` / `gsMultiGrid` | Advanced solvers and multilevel methods |
| `gsIO` | File I/O: XML (native), Paraview VTK (`gsWriteParaview`), and others |
| `gsUtils` | Utilities: `gsCmdLine` (command-line parsing), `gsFileManager`, combinatorics |

Optional modules live under `optional/` (e.g., `gsOpenCascade`, `gsSpectra`, `gsIpOpt`).

### Header/implementation split

Template implementations are split across three file types:
- `.h` — class declaration
- `.hpp` — template method definitions (included at end of `.h`)
- `_<ClassName>.cpp` — explicit instantiation for the default `real_t` type (compiled into the shared lib)

The single public entry point for clients is `#include <gismo.h>`.

### Logging and debugging

```cpp
gsInfo << "standard output\n";     // like std::cout, for program output
gsWarn << "something off\n";       // warnings (colored on Linux)
gsDebug << "only in Debug\n";      // compiled out in Release (NDEBUG)
gsDebugVar(myVar);                 // prints file:line, variable name and value
```

### Command-line arguments in examples

All examples follow the same pattern using `gsCmdLine`:

```cpp
gsCmdLine cmd("Description.");
cmd.addSwitch("plot", "Generate ParaView output", plot);
cmd.addInt("d", "degree", "Spline degree", degree);
try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }
```

Pass `--plot` to generate `.pvd`/`.vtp` files viewable in Paraview.

## Code Style

Formatting is defined by `.clang-format` (LLVM base style, 4-space indent, Allman braces, left pointer alignment). All code lives in `namespace gismo`. Class names use `gs` prefix (e.g., `gsBSpline`, `gsMatrix`).

Unit tests use UnitTest++ with `CHECK`, `CHECK_EQUAL`, `CHECK_CLOSE` macros.
