/** @file neuralPrec_nn_example.cpp

    @brief Minimal smoke test for gsNeuralPrec: loads an ONNX model and
    evaluates it on dummy inputs.

    The bundled model
        filedata/onnx_models/model_GeometryConditionedLinearOperator_best_YetiSchurTransformer.onnx
    has the following I/O:

        input  'input'    : float[1, 48]       <-- residual on skeleton DOFs (f)
        input  'obj.1'    : float[1, 304, 4]   <-- geometry features per local DOF
        output 'u'        : float[1, 48]       <-- preconditioned vector on skeleton DOFs
        output 'Q'        : float[1, 48, 20]   <-- attention weights
        output 'K_scaled' : float[1, 48, 20]   <-- scaled keys
        output 'epsilon'  : float[]             <-- scalar residual-skip coefficient

    gsNeuralPrec only ever asks for the 'u' output -- that's the model's
    actual preconditioning result and the only thing apply() needs. The
    other three outputs are exposed by the graph so the identity

        Kt_f = K_scaled^T @ f
        Qf   = Q @ Kt_f
        u    = Qf + epsilon * f

    can be checked directly against saved reference CSVs, without a
    second ONNX Run.

    This example creates two dummy tensors with the 'input'/'obj.1' shapes,
    hands them to gsNeuralPrec, runs apply() a few times, and prints a small
    slice of the output so you can verify the wiring end-to-end. It then
    reruns on saved reference inputs and checks both gsNeuralPrec's own
    output and the u = Qf + epsilon*f formula against saved reference CSVs.

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
        "model_GeometryConditionedLinearOperator_best_YetiSchurTransformer.onnx";

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
    //   primary input  "input"  : 48 values    (residual on skeleton DOFs, f)
    //   aux     input  "obj.1"  : 304 * 4 = 1216 values (geometry features per local DOF)
    //   output         "u"      : 48 values    (skeleton DOFs)
    // (the model also exposes "Q", "K_scaled", "epsilon" -- see the formula
    // check below; gsNeuralPrec itself only ever reads "u")
    gsNeuralPrec<real_t> nnPrec(
        model_path, "input", "u", use_cuda
    );

    gsInfo << "Loaded. rows=" << nnPrec.rows()
           << "  cols=" << nnPrec.cols() << "\n";

    // --- Build the auxiliary geometry tensor ---------------------------------
    // ONNX shape [1, 304, 4] is row-major: for each of the 304 local DOFs, the
    // 4 features are contiguous in memory. gsMatrix is column-major, so to
    // match that layout we use a (4 x 304) matrix where each COLUMN is one
    // DOF's 4-feature vector. The flat buffer .data() then iterates
    // DOF-by-DOF, feature-by-feature, which is exactly what the model
    // expects.
    const index_t n_local     = 304;   // local DOFs (geometry conditioning)
    const index_t n_skeleton  = 48;    // skeleton DOFs (primary input/output)
    const index_t n_features  = 4;
    gsMatrix<real_t> geom(n_features, n_local);
    geom.setRandom();   // dummy geometry features in [-1, 1]

    nnPrec.setAuxiliaryInput("obj.1", geom);

    // --- Build the primary input (dof vector) and apply ---------------------
    gsMatrix<real_t> dofs(n_skeleton, 1);
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
    const std::string prefix  = "model_GeometryConditionedLinearOperator_best_YetiSchurTransformer_";

    gsMatrix<real_t> ref_input0  = loadCsvRow(csv_dir + prefix + "input_0.csv");  // 48x1  (f)
    gsMatrix<real_t> ref_input1  = loadCsvRow(csv_dir + prefix + "input_1.csv");  // 1216x1
    gsMatrix<real_t> ref_output  = loadCsvRow(csv_dir + prefix + "u.csv");        // 48x1  (u)
    gsMatrix<real_t> ref_Q       = loadCsvRow(csv_dir + prefix + "Q.csv");        // 960x1
    gsMatrix<real_t> ref_K       = loadCsvRow(csv_dir + prefix + "K_scaled.csv"); // 960x1
    gsMatrix<real_t> ref_epsilon = loadCsvRow(csv_dir + prefix + "epsilon.csv");  // 1x1

    gsInfo << "Loaded CSV: input_0=" << ref_input0.size()
           << "  input_1=" << ref_input1.size()
           << "  u=" << ref_output.size()
           << "  Q=" << ref_Q.size()
           << "  K_scaled=" << ref_K.size()
           << "  epsilon=" << ref_epsilon.size() << "\n";

    // obj.1 is stored flat as [1,304,4] row-major → 1216 values.
    // setAuxiliaryInput expects a (4 x 304) column-major matrix (same memory layout).
    gsMatrix<real_t> ref_geom = ref_input1;   // already 1216x1, castIn reads .data() sequentially
    ref_geom.resize(n_features, n_local);     // reshape in-place: (4 x 304), same data order

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

    // -------------------------------------------------------------------------
    // Formula check: u = Q @ (K_scaled^T @ f) + epsilon * f, using the Q /
    // K_scaled / epsilon outputs the model now also exposes (loaded above
    // from CSV, since those CSVs already came from a real forward pass).
    // -------------------------------------------------------------------------
    gsInfo << "\n--- Formula check: u = Q @ (K_scaled^T @ f) + epsilon * f ---\n";

    const index_t n_heads = 20;

    // Q and K_scaled are [1, 48, 20] row-major in ONNX (48 rows of 20
    // contiguous values each). Reshaping the flat buffer into a (20 x 48)
    // column-major gsMatrix puts each ONNX row into one column, which is
    // exactly the transpose -- same reshape trick as 'obj.1' above, so the
    // reshaped buffers already ARE K_scaled^T / Q^T.
    gsMatrix<real_t> K_scaled_T = ref_K;
    K_scaled_T.resize(n_heads, n_skeleton);   // reshape in-place: (20 x 48) == K_scaled^T

    gsMatrix<real_t> Q_T = ref_Q;
    Q_T.resize(n_heads, n_skeleton);          // reshape in-place: (20 x 48) == Q^T

    gsMatrix<real_t> Kt_f     = K_scaled_T * ref_input0;                   // (20x48)*(48x1) = 20x1
    gsMatrix<real_t> Qf       = Q_T.transpose() * Kt_f;                    // (48x20)*(20x1) = 48x1
    gsMatrix<real_t> u_manual = Qf + ref_epsilon(0, 0) * ref_input0;       // 48x1

    gsMatrix<real_t> formula_diff = (u_manual - ref_output).cwiseAbs();
    const real_t formula_max_err  = formula_diff.maxCoeff();
    const real_t formula_mean_err = formula_diff.mean();

    gsInfo << "Max  |u_manual - u_reference| = " << formula_max_err  << "\n";
    gsInfo << "Mean |u_manual - u_reference| = " << formula_mean_err << "\n";

    if (formula_max_err < abs_tol)
        gsInfo << "PASSED (tolerance " << abs_tol << ")\n";
    else
        gsInfo << "FAILED: max error " << formula_max_err << " exceeds tolerance " << abs_tol << "\n";

    return 0;
}
