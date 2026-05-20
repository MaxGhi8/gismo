# G+Smo (Geometry plus Simulation modules)

G+Smo is a C++ library for isogeometric analysis (IGA). It provides a modular and extensible framework for geometry processing, modeling, and simulation, focusing on the seamless integration of Computer-Aided Design (CAD) and Finite Element Analysis (FEA).

## Project Overview

- **Main Technologies:** C++ (standard library, templates), CMake (build system).
- **Core Library:** Eigen (for linear algebra), Doxygen (for documentation).
- **Architecture:** Modular C++ library with components like `gsCore`, `gsMatrix`, `gsNurbs`, `gsAssembler`, `gsSolver`, etc.
- **Interfaces:** Support for Python (via `cppyy` or `pybind11`), Julia, and MATLAB.

## Building and Running

The project uses CMake for configuration and building. In-source builds are disabled.

### Basic Build (Linux/macOS)

A convenience `Makefile` is provided in the root directory.

```bash
# Create build directory and compile everything
make
```

Alternatively, use the standard CMake workflow:

```bash
mkdir build
cd build
cmake ..
make
```

### Build Options

Key CMake options (can be passed with `-D OPTION=VALUE`):

- `GISMO_BUILD_LIB`: Build the dynamic library (default: `ON`).
- `GISMO_BUILD_EXAMPLES`: Build the example programs (default: `ON`).
- `GISMO_BUILD_UNITTESTS`: Build the unit tests (default: `OFF`).
- `GISMO_OPTIONAL`: Semicolon-separated list of optional modules (e.g., `gsSpectra;gsOpenCascade`).
- `GISMO_COEFF_TYPE`: The arithmetic type for computations (e.g., `double`, `long double`, `float`; default: `double`).
- `GISMO_EXTRA_INSTANCE`: Compile the library with extra arithmetic types enabled.
- `GISMO_WITH_XDEBUG`: Enable additional debugging tools (default: `OFF`).
- `GISMO_PLUGIN_AXL`: Compile the plugin for Axel modeler (default: `OFF`).
- `CMAKE_BUILD_TYPE`: Build configuration (e.g., `Release`, `Debug`, `RelWithDebInfo`).
- `CMAKE_INSTALL_PREFIX`: Installation directory for the library.

### Running Examples

After compilation, executables are located in the `build/bin` directory.

```bash
./build/bin/bSplineCurve_example
```

### Documentation

To generate the Doxygen documentation (requires Doxygen):

```bash
make doc
```
The output will be in `build/doc/html/index.html`.

## Testing

Unit tests are located in the `unittests/` directory.

1. Enable tests during configuration: `cmake -DGISMO_BUILD_UNITTESTS=ON ..`
2. Build the tests: `make unittests` (or just `make`)
3. Run tests using CTest: `ctest` (inside the build directory) or execute the binary: `./build/bin/unittests`

## Development Conventions

- **In-source builds:** Strictly disabled. Always use a separate build directory.
- **Code Style:** The project uses `.clang-format`. Please adhere to the existing styling.
- **Language:** C++ with extensive use of templates.
- **Header Files:** Modules are organized into subdirectories in `src/`. Headers are typically separated into `.h` (declarations) and `.hpp` (template implementations).

## Directory Structure

- `src/`: Core library source code, organized by modules.
- `examples/`: Usage examples, small programs, and tutorials.
- `unittests/`: Unit tests for various parts of the library.
- `filedata/`: XML data files used by examples and tests.
- `extensions/`: Optional features and modules.
- `cmake/`: CMake configuration files and modules.
- `doc/`: Doxygen documentation configuration and snippets.
- `plugins/`: Plugins for external software (e.g., Axel, Rhinoceros).
- `external/`: Third-party dependencies (e.g., Eigen).

## IETI and Geometry Workflows

### Working with Geometries
- **Location:** Common geometries are stored in `filedata/domain2d/` and `filedata/volumes/`.
- **Visualization:** Use `geometry_example` to convert XML to ParaView format:
  ```bash
  ./build/bin/geometry_example -i filedata/domain2d/yeti_mp2.xml -o my_geometry
  ```
- **Splitting Patches:** Use the `--SplitPatches N` flag in many examples to turn a single-patch geometry into a multi-patch domain suitable for IETI.

### Running IETI Examples
- **Schur Complement (CG):** `ieti_example`
  ```bash
  ./build/bin/ieti_example -g domain2d/yeti_mp2.xml --plot
  ```
- **Saddle Point (MINRES):** `ieti2_example`
  ```bash
  ./build/bin/ieti2_example -g domain2d/yeti_mp2.xml --plot
  ```
- **Visualizing Results:** In ParaView, use the "Surface With Edges" representation to see both the solution and the patch/knot-line discretization.

## Contact and Support

- **Wiki:** [https://github.com/gismo/gismo/wiki](https://github.com/gismo/gismo/wiki)
- **Issues:** [https://github.com/gismo/gismo/issues](https://github.com/gismo/gismo/issues)
- **Discussions:** [https://github.com/gismo/gismo/discussions](https://github.com/gismo/gismo/discussions)
