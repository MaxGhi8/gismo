/** @file neuralPrec_nn_example.cpp

    @brief Minimal smoke test for gsNeuralPrec: loads an ONNX model and
    evaluates it on dummy inputs.

    The bundled model
        filedata/onnx_models/best_model_Transformer_homogeneous_neumann_l_0_deg_2_crazygeom_h40_new.onnx
    has the following I/O:

        input  'input' : float[1, 625]      <-- residual / dof vector
        input  'obj.1' : float[1, 625, 4]   <-- geometry features per dof
        output 'output': float[1, 625]      <-- preconditioned vector

    This example creates two dummy tensors with those shapes, hands them
    to gsNeuralPrec, runs apply() a few times, and prints a small slice
    of the output so you can verify the wiring end-to-end.

    Build: requires ONNX Runtime; configure with
        cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
    Files matching *_nn_example.cpp are skipped automatically otherwise.

    Author(s): M. Ghiotto
*/

#include <gismo.h>
#include "gsNeuralPrec.h"

#include <fstream>
#include <sstream>

using namespace gismo;

// Read a single-row CSV file into a gsMatrix<real_t> column vector.
static gsMatrix<real_t> loadCsvRow(const std::string & path)
{
    std::ifstream f(path);
    GISMO_ENSURE(f.is_open(), "Cannot open CSV file: " << path);
    std::string line;
    std::getline(f, line);
    std::istringstream ss(line);
    std::vector<real_t> vals;
    std::string tok;
    while (std::getline(ss, tok, ','))
        vals.push_back(static_cast<real_t>(std::stod(tok)));
    gsMatrix<real_t> out(vals.size(), 1);
    for (size_t i = 0; i < vals.size(); ++i) out(i, 0) = vals[i];
    return out;
}

int main(int argc, char * argv[])
{
    std::string model_path =
        GISMO_DATA_DIR "onnx_models/"
        "best_model_Transformer_homogeneous_neumann_l_0_deg_2_crazygeom_h40_H3_realSPD.onnx";

    bool use_cuda  = false;
    int  n_repeats = 3;

    gsCmdLine cmd("Smoke test for gsNeuralPrec on a bundled ONNX model.");
    cmd.addString("m", "model",   "Path to the ONNX model file", model_path);
    cmd.addSwitch("cuda",         "Use CUDA execution provider",  use_cuda);
    cmd.addInt   ("n", "repeats", "Number of apply() calls",      n_repeats);
    try { cmd.getValues(argc, argv); } catch (int rv) { return rv; }

    gsInfo << "Model : " << model_path << "\n";
    gsInfo << "CUDA  : " << (use_cuda ? "yes" : "no") << "\n";

    // The model expects:
    //   primary input  "input"  : 625 values   (dofs / residual)
    //   aux     input  "obj.1"  : 625 * 4 = 2500 values (geometry features)
    //   output         "output" : 625 values
    gsNeuralPrec<real_t> nnPrec(
        model_path, "input", "output", use_cuda
    );

    gsInfo << "Loaded. rows=" << nnPrec.rows()
           << "  cols=" << nnPrec.cols() << "\n";

    // --- Build the auxiliary geometry tensor ---------------------------------
    // ONNX shape [1, 625, 4] is row-major: for each of the 625 points, the
    // 4 features are contiguous in memory. gsMatrix is column-major, so to
    // match that layout we use a (4 x 625) matrix where each COLUMN is one
    // point's 4-feature vector. The flat buffer .data() then iterates
    // point-by-point, feature-by-feature, which is exactly what the model
    // expects.
    const index_t n_points   = 625;
    const index_t n_features = 4;
    gsMatrix<real_t> geom(n_features, n_points);
    geom.setRandom();   // dummy geometry features in [-1, 1]

    nnPrec.setAuxiliaryInput("obj.1", geom);

    // --- Build the primary input (dof vector) and apply ---------------------
    gsMatrix<real_t> dofs(n_points, 1);
    gsMatrix<real_t> out;

    // Warm-up (JIT compilation, GPU transfer, etc. happen on first call)
    dofs.setRandom();
    nnPrec.apply(dofs, out);
    gsInfo << "Warm-up done.\n";

    gsStopwatch timer;
    for (int k = 0; k < n_repeats; ++k)
    {
        dofs.setRandom();

        timer.restart();
        nnPrec.apply(dofs, out);
        const double elapsed_ms = timer.stop() * 1e3;

        gsInfo << "[call " << k << "]  time=" << elapsed_ms << " ms"
               << "   first 5: " << out.topRows(5).transpose() << "\n";
    }

    // -------------------------------------------------------------------------
    // Sanity check: compare against saved Python reference outputs
    // -------------------------------------------------------------------------
    gsInfo << "\n--- Sanity check against saved CSV reference ---\n";

    const std::string csv_dir = GISMO_DATA_DIR "onnx_models/";
    const std::string prefix  = "model_GeometryConditionedLinearOperator_best_AFIETI_transformer_";

    gsMatrix<real_t> ref_input0  = loadCsvRow(csv_dir + prefix + "input_0.csv");  // 625x1
    gsMatrix<real_t> ref_input1  = loadCsvRow(csv_dir + prefix + "input_1.csv");  // 2500x1
    gsMatrix<real_t> ref_output  = loadCsvRow(csv_dir + prefix + "output.csv");   // 625x1

    gsInfo << "Loaded CSV: input_0=" << ref_input0.size()
           << "  input_1=" << ref_input1.size()
           << "  output=" << ref_output.size() << "\n";

    // obj.1 is stored flat as [1,625,4] row-major → 2500 values.
    // setAuxiliaryInput expects a (4 x 625) column-major matrix (same memory layout).
    gsMatrix<real_t> ref_geom = ref_input1;   // already 2500x1, castIn reads .data() sequentially
    ref_geom.resize(n_features, n_points);    // reshape in-place: (4 x 625), same data order

    nnPrec.setAuxiliaryInput("obj.1", ref_geom);

    gsMatrix<real_t> ref_out_computed;
    nnPrec.apply(ref_input0, ref_out_computed);

    // Compare element-wise
    const real_t abs_tol = 1e-4;
    gsMatrix<real_t> diff = (ref_out_computed - ref_output).cwiseAbs();
    const real_t max_err  = diff.maxCoeff();
    const real_t mean_err = diff.mean();

    gsInfo << "Max  |computed - reference| = " << max_err  << "\n";
    gsInfo << "Mean |computed - reference| = " << mean_err << "\n";

    if (max_err < abs_tol)
        gsInfo << "PASSED (tolerance " << abs_tol << ")\n";
    else
        gsInfo << "FAILED: max error " << max_err << " exceeds tolerance " << abs_tol << "\n";

    return 0;
}
