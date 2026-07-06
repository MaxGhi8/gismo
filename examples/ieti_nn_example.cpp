/** @file ieti_nn_example.cpp

    @brief IETI solver with a cached neural-network Schur complement.

    Mirrors ieti_example.cpp (CG on the Schur complement formulation), but
    replaces each patch's local Schur complement operator S_k inside the
    scaled Dirichlet preconditioner with a learned model's cached linear
    decomposition. One ONNX session is loaded ONCE and shared across all
    patches; each patch runs the model exactly ONCE, during setup, to
    extract three small per-patch tensors, then never touches ONNX again.

    Discretization: --SquareDiscretization (default "global") uses the same
    makeBenchBasis()/makeInterfacesConforming() helpers as
    solver_benchmark_example.cpp and ieti_dataset_generation_example.cpp
    (see gsBenchDiscretization.h), so every patch gets an identical raw
    basis (mb[k].size()). Even so, per-patch free-dof / skeleton-dof counts
    (N, S) after Dirichlet elimination can still differ slightly, since
    they depend on how many sides of a patch are an interface vs. the outer
    Dirichlet boundary (topology), not on mesh resolution.

    Because a given ONNX export has a fixed input shape, every patch's
    actual (N, S) is zero-padded up to the model's declared max (Nmax,
    Smax) before calling it -- see gsCachedNeuralSchurOp below. Nmax/Smax
    are read from the model's own declared input shapes.

    NOTE: the bundled model
        filedata/onnx_models/model_GeometryConditionedLinearOperator_best_YetiSchurTransformer.onnx
    was retrained on the yeti domain's 21 patches under the current
    --SquareDiscretization=global default at degree 2 / 2 refinements
    (Nmax=304, Smax=48). Using a coarser/finer discretization, a different
    degree, or --SquareDiscretization off (native, non-square patches) may
    produce per-patch (N, S) that this model was not trained for -- values
    exceeding Nmax/Smax trigger the padding-capacity GISMO_ENSURE below;
    values well below it will still run but are out-of-distribution for
    the model.

    Model contract: this file requires a model exposing the specific
    4-output decomposition below (NOT a generic single-output model):

        input  'input'    : float[1, Smax]     <-- residual on skeleton
                                                     DOFs (f), zero-padded
                                                     from the patch's
                                                     actual S <= Smax
        input  'obj.1'    : float[1, Nmax, 4]  <-- geometry features per
                                                     free local DOF
                                                     (x, y, z, w),
                                                     zero-padded from
                                                     N <= Nmax
                                                     x,y,z : physical coords
                                                     w     : NURBS weight
                                                             (=1 for
                                                             B-splines)
        output 'u'        : float[1, Smax]     <-- preconditioned vector
        output 'Q'        : float[1, Smax, H]  <-- attention weights
        output 'K_scaled' : float[1, Smax, H]  <-- scaled keys
        output 'epsilon'  : float[]             <-- scalar residual-skip
                                                     coefficient

    where u = Q @ (K_scaled^T @ f) + epsilon * f, and Q/K_scaled/epsilon
    depend only on the geometry input ('obj.1'), NOT on f -- verified
    empirically (see
    docs/superpowers/specs/2026-07-06-ieti-nn-cached-schur-operator-design.md).
    H (the head count) is read from the model's own 'Q' output shape, not
    hardcoded.

    Because of this, each patch's Q/K_scaled/epsilon are computed ONCE
    during setup (computeCachedSchurTensors, using one random probe f
    purely to cross-check the formula against the model's own 'u' output)
    and cached in a gsCachedNeuralSchurOp, which does only two small dense
    matvecs per apply() -- no ONNX Run() during the CG solve at all.

    Build: requires ONNX Runtime; configure with
        cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
    Files matching *_nn_example.cpp are skipped automatically otherwise.

    Author(s): M. Ghiotto
*/

#include <ctime>
#include <set>
#include <gismo.h>
#include "gsNeuralPrec.h"
#include "gsBenchDiscretization.h"

namespace gismo {

/// @brief Build the (4 x n_free) per-free-DOF geometry feature matrix for
/// patch k, where n_free is the number of DOFs left after Dirichlet
/// elimination (matches ieti_dataset_generation_example.cpp's "zeta").
///
/// Rows are (x, y, z, w):
///   x,y,z : physical coordinates of the basis function's Greville abscissa
///           (z = 0 in 2D);
///   w     : NURBS weight, = 1 for B-spline geometries.
///
/// Columns are ordered by free-dof index (0 .. n_free-1), NOT by raw local
/// basis index -- Dirichlet-eliminated basis functions are skipped
/// entirely, exactly as in the training data.
///
/// The (4 x n_free) column-major layout matches an ONNX [1, n_free, 4]
/// row-major tensor (per-DOF feature vectors contiguous in memory), which
/// is what gsNeuralPrec::setAuxiliaryInput expects.
template <class T>
gsMatrix<T> computePatchGeometryFeatures(const gsMultiPatch<T> & mp,
                                         const gsMultiBasis<T> & mb,
                                         const gsDofMapper & dofMapperLocal,
                                         index_t k)
{
    gsMatrix<T> anchors = mb[k].anchors();         // (paramDim, n_basis)
    gsMatrix<T> phys;
    mp.patch(k).eval_into(anchors, phys);          // (geoDim, n_basis)

    gsMatrix<T> features = gsMatrix<T>::Zero(4, dofMapperLocal.freeSize());
    for (index_t j = 0; j < mb[k].size(); ++j)
    {
        if (!dofMapperLocal.is_free(j, 0))
            continue;
        const index_t i = dofMapperLocal.index(j, 0);
        features(0, i) = phys(0, j);
        features(1, i) = (phys.rows() > 1) ? phys(1, j) : T(0);
        features(2, i) = (phys.rows() > 2) ? phys(2, j) : T(0);
        features(3, i) = T(1);                     // w = 1 (B-spline)
    }
    return features;
}

/// @brief The three per-patch tensors cached from one forward pass of the
/// model, sufficient to reproduce u = Q @ (K_scaled^T @ f) + epsilon * f
/// for any f, without ever calling the model again.
template <class T>
struct gsCachedSchurTensors
{
    gsMatrix<T> Q_T;          // (n_heads x maxSkeleton), == Q^T
    gsMatrix<T> K_scaled_T;   // (n_heads x maxSkeleton), == K_scaled^T
    T           epsilon;
};

/// @brief Runs the model ONCE for one patch's geometry to extract and cache
/// Q, K_scaled, epsilon (verified independent of the residual input), and
/// cross-checks that u = Q @ (K_scaled^T @ f) + epsilon * f reproduces the
/// model's own 'u' output for a random probe f, within `tol`. Throws
/// (naming `patchIndex`) if the cross-check fails.
///
/// `featuresPadded` is the (4 x maxLocal) geometry tensor, already
/// zero-padded (see computePatchGeometryFeatures), in the same layout
/// gsNeuralPrec::setAuxiliaryInput used to expect. `maxSkeleton`/`nHeads`
/// are read once in main() from the model's declared input/output shapes.
template <class T>
gsCachedSchurTensors<T> computeCachedSchurTensors(
    typename gsNeuralModel<T>::Ptr model,
    const std::string & inputName, const std::string & auxName,
    const std::string & outputName, const std::string & qName,
    const std::string & kName, const std::string & epsName,
    const gsMatrix<T> & featuresPadded,
    index_t maxSkeleton, index_t nSkeleton, index_t nHeads,
    T tol, index_t patchIndex)
{
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Random (non-zero) probe residual: Q/K_scaled/epsilon are geometry-only
    // (verified independent of 'input' -- see the design doc), so any
    // nontrivial vector works here. It exists only to make the cross-check
    // below non-vacuous; its value has no bearing on the cached tensors.
    gsMatrix<T> fProbe(maxSkeleton, 1);
    fProbe.setRandom();

    std::vector<float> fBuf(static_cast<size_t>(maxSkeleton));
    for (index_t i = 0; i < maxSkeleton; ++i)
        fBuf[static_cast<size_t>(i)] = static_cast<float>(fProbe(i, 0));

    std::vector<float> auxBuf(static_cast<size_t>(featuresPadded.size()));
    for (index_t i = 0; i < featuresPadded.size(); ++i)
        auxBuf[static_cast<size_t>(i)] = static_cast<float>(featuresPadded.data()[i]);

    std::vector<int64_t> fShape   = { 1, maxSkeleton };
    std::vector<int64_t> auxShape = { 1, featuresPadded.cols(), featuresPadded.rows() };

    Ort::Value inF = Ort::Value::CreateTensor<float>(
        memInfo, fBuf.data(), fBuf.size(), fShape.data(), fShape.size());
    Ort::Value inAux = Ort::Value::CreateTensor<float>(
        memInfo, auxBuf.data(), auxBuf.size(), auxShape.data(), auxShape.size());

    const char * inputNamesArr[]  = { inputName.c_str(), auxName.c_str() };
    const char * outputNamesArr[] = { outputName.c_str(), qName.c_str(), kName.c_str(), epsName.c_str() };
    Ort::Value inputs[] = { std::move(inF), std::move(inAux) };

    auto outputs = model->session().Run(
        Ort::RunOptions{nullptr},
        inputNamesArr, inputs, 2,
        outputNamesArr, 4);

    const float * uPtr   = outputs[0].template GetTensorData<float>();
    const float * qPtr   = outputs[1].template GetTensorData<float>();
    const float * kPtr   = outputs[2].template GetTensorData<float>();
    const float   epsVal = *outputs[3].template GetTensorData<float>();

    // Q and K_scaled are [1, Smax, H] row-major in ONNX (Smax rows of H
    // contiguous values each). Filling a (H x Smax) column-major gsMatrix
    // directly from that flat buffer lands each ONNX row in one column,
    // which is exactly the transpose -- same reshape trick as
    // neuralPrec_nn_example.cpp, so these ARE Q^T / K_scaled^T already.
    gsMatrix<T> Q_T(nHeads, maxSkeleton), K_scaled_T(nHeads, maxSkeleton);
    for (index_t i = 0; i < nHeads * maxSkeleton; ++i)
    {
        Q_T.data()[i]        = static_cast<T>(qPtr[i]);
        K_scaled_T.data()[i] = static_cast<T>(kPtr[i]);
    }

    gsMatrix<T> Kt_f     = K_scaled_T * fProbe;                  // (H x Smax)*(Smax x1)
    gsMatrix<T> Qf       = Q_T.transpose() * Kt_f;               // (Smax x H)*(H x1)
    gsMatrix<T> u_manual = (Qf + static_cast<T>(epsVal) * fProbe).topRows(nSkeleton);

    gsMatrix<T> u_model(nSkeleton, 1);
    for (index_t i = 0; i < nSkeleton; ++i)
        u_model(i, 0) = static_cast<T>(uPtr[i]);

    const T maxErr = (u_manual - u_model).cwiseAbs().maxCoeff();
    gsInfo << "  patch " << patchIndex << ": cached-formula cross-check max error = " << maxErr << "\n";
    GISMO_ENSURE(maxErr < tol,
        "computeCachedSchurTensors: patch " << patchIndex
        << ": cached-formula cross-check failed, max |u_manual - u_model| = "
        << maxErr << " exceeds tolerance " << tol
        << ". The model's Q/K_scaled/epsilon decomposition may not match "
        "the assumed formula for this patch.");

    return gsCachedSchurTensors<T>{ std::move(Q_T), std::move(K_scaled_T), static_cast<T>(epsVal) };
}

/// @brief Zero-pads a patch's actual skeleton-sized residual up to the
/// model's fixed max skeleton size, applies the cached
/// u = Q @ (K_scaled^T @ f) + epsilon * f formula, and truncates the
/// result back down -- see computeCachedSchurTensors above for where
/// Q/K_scaled/epsilon come from. No ONNX Run() ever happens here.
template <class T>
class gsCachedNeuralSchurOp : public gsLinearOperator<T>
{
public:
    typedef memory::shared_ptr<gsCachedNeuralSchurOp> Ptr;

    gsCachedNeuralSchurOp(gsCachedSchurTensors<T> tensors,
                         index_t maxSkeleton, index_t nSkeleton)
    : m_t(std::move(tensors)), m_maxSkeleton(maxSkeleton), m_nSkeleton(nSkeleton)
    {
        GISMO_ENSURE(m_nSkeleton <= m_maxSkeleton,
            "gsCachedNeuralSchurOp: patch skeleton size " << m_nSkeleton
            << " exceeds the model's maximum of " << m_maxSkeleton
            << ". Use a coarser discretization or a model trained with a "
            "larger maximum skeleton size.");
    }

    void apply(const gsMatrix<T> & input, gsMatrix<T> & x) const override
    {
        GISMO_ASSERT(input.rows() == m_nSkeleton,
            "gsCachedNeuralSchurOp::apply: expected " << m_nSkeleton
            << " skeleton entries, got " << input.rows());

        gsMatrix<T> padded = gsMatrix<T>::Zero(m_maxSkeleton, 1);
        padded.topRows(m_nSkeleton) = input;

        gsMatrix<T> Kt_f = m_t.K_scaled_T * padded;             // (H x Smax)*(Smax x1)
        gsMatrix<T> Qf   = m_t.Q_T.transpose() * Kt_f;          // (Smax x H)*(H x1)
        gsMatrix<T> out  = Qf + m_t.epsilon * padded;

        x = out.topRows(m_nSkeleton);
    }

    index_t rows() const override { return m_nSkeleton; }
    index_t cols() const override { return m_nSkeleton; }

private:
    gsCachedSchurTensors<T> m_t;
    index_t m_maxSkeleton, m_nSkeleton;
};

} // namespace gismo

using namespace gismo;

int main(int argc, char *argv[])
{
    /************** Define command line options *************/

    std::string geometry("domain2d/yeti_mp2.xml");
    std::string coeff("1.0");
    index_t splitPatches = 0;
    real_t stretchGeometry = 1;
    index_t refinements = 2;
    index_t degree = 2;
    std::string squareDiscr("global");
    std::string boundaryConditions("d");
    std::string primals("c");
    bool eliminateCorners = false;
    real_t tolerance = 1.e-8;
    index_t maxIterations = 100;
    bool calcEigenvalues = false;
    index_t numRun = 1;
    std::string out;
    bool plot = false;

    // NN-specific options
    std::string modelPath =
        GISMO_DATA_DIR "onnx_models/"
        "model_GeometryConditionedLinearOperator_best_YetiSchurTransformer.onnx";
    std::string nnInputName    = "input";
    std::string nnOutputName   = "u";
    std::string nnAuxName      = "obj.1";
    std::string nnQName        = "Q";
    std::string nnKScaledName  = "K_scaled";
    std::string nnEpsilonName  = "epsilon";
    bool useCuda = false;

    gsCmdLine cmd("Solves a PDE with IETI, using a neural-network Schur complement inside the scaled Dirichlet preconditioner.");
    cmd.addString("g", "Geometry",              "Geometry file", geometry);
    cmd.addString("a", "Coeff",                 "Coefficient function a(x)", coeff);
    cmd.addInt   ("",  "SplitPatches",          "Split every patch that many times in 2^d patches", splitPatches);
    cmd.addReal  ("",  "StretchGeometry",       "Stretch geometry in x-direction by the given factor", stretchGeometry);
    cmd.addInt   ("r", "Refinements",           "Number of uniform h-refinement steps to perform before solving", refinements);
    cmd.addInt   ("p", "Degree",                "Degree of the B-spline discretization space", degree);
    cmd.addString("",  "SquareDiscretization",  "Element equalisation: off (native, patches differ) | global (all patches identical)", squareDiscr);
    cmd.addString("b", "BoundaryConditions",    "Boundary conditions", boundaryConditions);
    cmd.addString("c", "Primals",               "Primal constraints (c=corners, e=edges, f=faces)", primals);
    cmd.addSwitch("e", "EliminateCorners",      "Eliminate corners (if they are primals)", eliminateCorners);
    cmd.addReal  ("t", "Solver.Tolerance",      "Stopping criterion for linear solver", tolerance);
    cmd.addInt   ("",  "Solver.MaxIterations",  "Maximum iterations for linear solver", maxIterations);
    cmd.addSwitch("",  "Solver.CalcEigenvalues","Estimate eigenvalues based on Lanczos", calcEigenvalues);
    cmd.addInt   ("",  "num_run",               "Number of times the solve is repeated; the reported solve time is the mean over the repeats (setup is measured once)", numRun);
    cmd.addString("m", "model",                 "Path to the ONNX model file", modelPath);
    cmd.addString("",  "nnInput",               "ONNX primary input tensor name", nnInputName);
    cmd.addString("",  "nnOutput",              "ONNX output tensor name (used only for the one-time setup-time formula cross-check)", nnOutputName);
    cmd.addString("",  "nnAux",                 "ONNX auxiliary (geometry) input tensor name", nnAuxName);
    cmd.addString("",  "nnQ",                   "ONNX 'Q' output tensor name", nnQName);
    cmd.addString("",  "nnKScaled",             "ONNX 'K_scaled' output tensor name", nnKScaledName);
    cmd.addString("",  "nnEpsilon",             "ONNX 'epsilon' output tensor name", nnEpsilonName);
    cmd.addSwitch(     "cuda",                  "Use CUDA execution provider", useCuda);
    cmd.addString("",  "out",                   "Write solution and used options to file", out);
    cmd.addSwitch(     "plot",                  "Plot the result with Paraview", plot);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

    if (numRun < 1) numRun = 1;

    if ( ! gsFileManager::fileExists(geometry) )
    {
        gsInfo << "Geometry file could not be found.\n";
        gsInfo << "I was searching in the current directory and in: " << gsFileManager::getSearchPaths() << "\n";
        return EXIT_FAILURE;
    }

    gsInfo << "Run ieti_nn_example with options:\n" << cmd << std::endl;

    /******************* Define geometry ********************/

    gsInfo << "Define geometry... " << std::flush;

    gsMultiPatch<>::uPtr mpPtr = gsReadFile<>(geometry);
    if (!mpPtr)
    {
        gsInfo << "No geometry found in file " << geometry << ".\n";
        return EXIT_FAILURE;
    }
    gsMultiPatch<>& mp = *mpPtr;

    // Ensure a usable topology BEFORE splitting (see solver_benchmark_example.cpp
    // for the rationale): CAD formats like IGES/STEP arrive with no registered
    // interfaces, so recover the topology by geometric matching first, or a
    // subsequent uniformSplit()/IETI setup would silently treat every patch side
    // as an unglued boundary.
    if (mp.nInterfaces() == 0)
    {
        gsInfo << "(no interfaces registered, computing topology) " << std::flush;
        mp.computeTopology();
    }

    for (index_t i=0; i<splitPatches; ++i)
    {
        gsInfo << "split patches uniformly... " << std::flush;
        mp = mp.uniformSplit();
    }

    if (stretchGeometry!=1)
    {
       gsInfo << "and stretch it... " << std::flush;
       for (size_t i=0; i!=mp.nPatches(); ++i)
           const_cast<gsGeometry<>&>(mp[i]).scale(stretchGeometry,0);
    }

    gsInfo << "done.\n";

    /************** Define boundary conditions **************/

    gsInfo << "Define right-hand-side and boundary conditions... " << std::flush;

    gsFunctionExpr<> a( coeff, mp.geoDim() );
    gsFunctionExpr<> f( "2*sin(x)*cos(y)", mp.geoDim() );
    gsFunctionExpr<> gD( "sin(x)*cos(y)", mp.geoDim() );
    gsConstantFunction<> gN( 1.0, mp.geoDim() );

    gsBoundaryConditions<> bc;
    {
        const index_t len = boundaryConditions.length();
        index_t i = 0;
        for (gsMultiPatch<>::const_biterator it = mp.bBegin(); it < mp.bEnd(); ++it)
        {
            char b_local;
            if ( len == 1 )
                b_local = boundaryConditions[0];
            else if ( i < len )
                b_local = boundaryConditions[i];
            else
            {
                gsInfo << "\nNot enough boundary conditions given.\n";
                return EXIT_FAILURE;
            }

            if ( b_local == 'd' )
                bc.addCondition( *it, condition_type::dirichlet, &gD );
            else if ( b_local == 'n' )
                bc.addCondition( *it, condition_type::neumann, &gN );
            else
            {
                gsInfo << "\nInvalid boundary condition given; only 'd' (Dirichlet) and 'n' (Neumann) are supported.\n";
                return EXIT_FAILURE;
            }

            ++i;
        }
        if ( len > i )
            gsInfo << "\nToo many boundary conditions have been specified. Ignoring the remaining ones.\n";
        gsInfo << "done. "<<i<<" boundary conditions set.\n";
    }


    /************ Setup bases and adjust degree *************/

    gsInfo << "Setup bases and adjust degree... " << std::flush;

    // --SquareDiscretization global (default, matches solver_benchmark_example.cpp
    // and ieti_dataset_generation_example.cpp) makes every patch's raw basis
    // identical. makeInterfacesConforming then unions interface knots so the
    // dof-mapper/IETI matchWith still succeeds if patches were non-conforming.
    gsMultiBasis<> mb = makeBenchBasis(mp, degree, refinements, squareDiscr);
    makeInterfacesConforming(mp, mb);

    gsInfo << "done.\n";

    for ( size_t i = 0; i < mb.nBases(); ++ i )
    {
        gsInfo << "Patch " << i << ": Degree " << mb[i].degree(0);
        for (short_t d = 1; d < mb[i].domainDim(); ++d) gsInfo << "x" << mb[i].degree(d);
        gsInfo << ", " << mb[i].size() << " basis functions.\n";
    }

    /********* Setup assembler and assemble matrix **********/

    gsInfo << "Setup assembler and assemble matrix... " << std::flush;

    const index_t nPatches = mp.nPatches();

    gsIetiMapper<> ietiMapper;

    {
        typedef gsExprAssembler<>::space  space;
        gsExprAssembler<> assembler;
        space u = assembler.getSpace(mb);
        bc.setGeoMap(mp);
        u.setup(bc, dirichlet::interpolation, 0);
        ietiMapper.init( mb, u.mapper(), u.fixedPart() );
    }

    bool cornersAsPrimals = false, edgesAsPrimals = false, facesAsPrimals = false;
    for (size_t i=0; i<primals.length(); ++i)
        switch (primals[i])
        {
            case 'c': cornersAsPrimals = true;   break;
            case 'e': edgesAsPrimals = true;     break;
            case 'f': facesAsPrimals = true;     break;
            default:
                gsInfo << "\nUnkown type of primal constraint: \"" << primals[i] << "\"\n";
                return EXIT_FAILURE;
        }

    if (cornersAsPrimals)
        ietiMapper.cornersAsPrimals();

    if (edgesAsPrimals)
        ietiMapper.interfaceAveragesAsPrimals(mp,1);

    if (facesAsPrimals)
        ietiMapper.interfaceAveragesAsPrimals(mp,2);

    bool fullyRedundant = true,
         noLagrangeMultipliersForCorners = cornersAsPrimals;
    ietiMapper.computeJumpMatrices(fullyRedundant, noLagrangeMultipliersForCorners);

    gsIetiSystem<> ieti;
    ieti.reserve(nPatches+1);

    gsScaledDirichletPrec<> prec;
    prec.reserve(nPatches);

    gsPrimalSystem<> primal(ietiMapper.nPrimalDofs());
    if (eliminateCorners)
        primal.setEliminatePointwiseConstraints(true);

    /******** Neural-network Schur complement setup *********/

    // Load the ONNX model ONCE and share the session across all patches.
    // Each per-patch gsNeuralPrec then only allocates its own small I/O
    // buffers + the bound geometry tensor, so adding patches is O(buffers)
    // not O(reload-model). Critical when patch count grows and even more
    // so with --cuda (no repeated upload of model weights to the GPU).
    gsInfo << "Load ONNX model: " << modelPath << " ("
           << (useCuda ? "CUDA" : "CPU") << ")... " << std::flush;
    gsStopwatch modelTimer;
    gsNeuralModel<real_t>::Ptr nnModel =
        gsNeuralModel<real_t>::Ptr(new gsNeuralModel<real_t>(modelPath, useCuda));
    gsInfo << "done (" << modelTimer.stop() << " s).\n";

    // The model's fixed max shapes (see file header): patches with fewer
    // skeleton/free dofs than these are zero-padded up to them.
    auto modelInputElems = [&nnModel](const std::string & name) -> index_t
    {
        const auto & names  = nnModel->inputNames();
        const auto & shapes = nnModel->inputShapes();
        for (size_t i = 0; i < names.size(); ++i)
            if (names[i] == name)
            {
                int64_t numel = 1;
                for (auto d : shapes[i]) numel *= d;
                return static_cast<index_t>(numel);
            }
        GISMO_ERROR("ieti_nn_example: model input '" << name << "' not found.");
    };
    const index_t maxSkeleton = modelInputElems(nnInputName);
    const index_t maxLocal    = modelInputElems(nnAuxName) / 4;
    gsInfo << "NN model max shapes: skeleton dofs <= " << maxSkeleton
           << ", free local dofs <= " << maxLocal
           << " (patches are zero-padded up to these).\n";

    // Head count H for the Q/K_scaled decomposition: read from the model's
    // own 'Q' output shape [1, Smax, H], not hardcoded.
    auto modelOutputShape = [&nnModel](const std::string & name) -> std::vector<int64_t>
    {
        const auto & names  = nnModel->outputNames();
        const auto & shapes = nnModel->outputShapes();
        for (size_t i = 0; i < names.size(); ++i)
            if (names[i] == name) return shapes[i];
        GISMO_ERROR("ieti_nn_example: model output '" << name << "' not found.");
    };
    const std::vector<int64_t> qShape = modelOutputShape(nnQName);
    GISMO_ENSURE(qShape.size() == 3 && qShape[1] == maxSkeleton,
        "ieti_nn_example: model output '" << nnQName << "' has shape rank "
        << qShape.size() << ", expected [1, " << maxSkeleton << ", H].");
    const index_t nHeads = static_cast<index_t>(qShape[2]);
    gsInfo << "NN model head count H = " << nHeads
           << " (from '" << nnQName << "' output shape).\n";

    const real_t cacheTol = 1e-4;

    gsStopwatch nnTimer;

    for (index_t k=0; k<nPatches; ++k)
    {
        gsBoundaryConditions<> bc_local;
        bc.getConditionsForPatch(k,bc_local);
        gsMultiPatch<> mp_local = mp[k];
        gsMultiBasis<> mb_local = mb[k];

        typedef gsExprAssembler<>::geometryMap geometryMap;
        typedef gsExprAssembler<>::variable    variable;
        typedef gsExprAssembler<>::space       space;

        gsExprAssembler<> assembler(1,1);
        assembler.setIntegrationDomain(mb_local.domain());
        gsExprEvaluator<> ev(assembler);
        geometryMap G = assembler.getMap(mp_local);
        space u = assembler.getSpace(mb_local);
        bc_local.setGeoMap(mp_local);
        u.setup(bc_local, dirichlet::interpolation, 0);
        ietiMapper.initFeSpace(u,k);

        auto ff = assembler.getCoeff(f, G);
        auto aa = assembler.getCoeff(a, G);

        assembler.initSystem();
        assembler.assemble( aa.val() * igrad(u, G) * igrad(u, G).tr() * meas(G), u * ff * meas(G) );

        variable g_N = assembler.getBdrFunction();
        assembler.assembleBdr(bc_local.get("Neumann"),  u * g_N.val() * nv(G).norm() );

        gsSparseMatrix<real_t, RowMajor> jumpMatrix  = ietiMapper.jumpMatrix(k);
        gsSparseMatrix<>                 localMatrix = assembler.matrix();
        gsMatrix<>                       localRhs    = assembler.rhs();

        // --- Cached neural Schur complement for this patch ---------------
        // Runs the model ONCE (computeCachedSchurTensors) to extract and
        // cross-check Q/K_scaled/epsilon for this patch's geometry; the CG
        // solve below then only ever calls gsCachedNeuralSchurOp::apply(),
        // which is pure gsMatrix algebra -- no ONNX Run() per iteration.
        const gsDofMapper & dofMapperLocal = ietiMapper.dofMapperLocal(k);
        gsMatrix<real_t> features = computePatchGeometryFeatures(mp, mb, dofMapperLocal, k);

        std::vector<index_t> skeletonDofs = ietiMapper.skeletonDofs(k);

        GISMO_ENSURE(features.cols() <= maxLocal,
            "ieti_nn_example: patch " << k << " has " << features.cols()
            << " free local dofs, exceeding the model's maximum of "
            << maxLocal << ". Use a coarser discretization or a model "
            "trained with a larger maximum.");

        gsMatrix<real_t> featuresPadded = gsMatrix<real_t>::Zero(4, maxLocal);
        featuresPadded.leftCols(features.cols()) = features;

        const index_t nSkeleton = static_cast<index_t>(skeletonDofs.size());

        gsCachedSchurTensors<real_t> tensors = computeCachedSchurTensors<real_t>(
            nnModel, nnInputName, nnAuxName, nnOutputName,
            nnQName, nnKScaledName, nnEpsilonName,
            featuresPadded, maxSkeleton, nSkeleton, nHeads,
            cacheTol, k
        );

        gsLinearOperator<>::Ptr nnSchurOp = std::make_shared<gsCachedNeuralSchurOp<real_t>>(
            std::move(tensors), maxSkeleton, nSkeleton
        );

        prec.addSubdomain(
            gsScaledDirichletPrec<>::restrictJumpMatrix(jumpMatrix, skeletonDofs).moveToPtr(),
            nnSchurOp
        );
        // ----------------------------------------------------------------

        // primal.handleConstraints rewrites jumpMatrix/localMatrix/localRhs;
        // must run after prec.addSubdomain so the un-modified jump matrix
        // is what we restricted above.
        auto const & pConstraints = ietiMapper.primalConstraints(k);
        auto const & pDofIndices  = ietiMapper.primalDofIndices(k);

        std::vector<gsSparseVector<real_t>> uniqueConstraints;
        std::vector<index_t> uniqueDofIndices;
        std::set<index_t> seen;
        for (size_t i = 0; i < pDofIndices.size(); ++i)
        {
            // Skip constraints whose vector is zero: the corner DOF is entirely
            // on the Dirichlet boundary for this patch, so it has been eliminated
            // from the local free-DOF basis. Adding a zero row/column to the
            // saddle-point system would make it singular and crash the SparseLU
            // solve (see solver_benchmark_example.cpp's IETI-DP block).
            if (pConstraints[i].nonZeros() == 0) continue;
            if (seen.find(pDofIndices[i]) == seen.end())
            {
                uniqueConstraints.push_back(pConstraints[i]);
                uniqueDofIndices.push_back(pDofIndices[i]);
                seen.insert(pDofIndices[i]);
            }
        }

        primal.handleConstraints(
            uniqueConstraints,
            uniqueDofIndices,
            jumpMatrix,
            localMatrix,
            localRhs
        );

        ieti.addSubdomain(
            jumpMatrix.moveToPtr(),
            makeMatrixOp(localMatrix.moveToPtr()),
            give(localRhs)
        );
    } // end for

    const double nnSetupTime = nnTimer.stop();

    if (ietiMapper.nPrimalDofs()>0)
    {
        gsLinearOperator<>::Ptr localSolver
            = makeSparseCholeskySolver(primal.localMatrix());

        ieti.addSubdomain(
            primal.jumpMatrix().moveToPtr(),
            makeMatrixOp(primal.localMatrix().moveToPtr()),
            give(primal.localRhs()),
            localSolver
        );
    }

    gsInfo << "done. " << ietiMapper.nPrimalDofs() << " primal dofs. "
           << "NN+assembly setup time: " << nnSetupTime << " s\n";

    /**************** Setup solver and solve ****************/

    gsInfo << "Setup solver and solve... \n"
        "    Setup multiplicity scaling... " << std::flush;

    prec.setupMultiplicityScaling();

    gsInfo << "done.\n    Setup rhs... " << std::flush;
    gsMatrix<> rhsForSchur = ieti.rhsForSchurComplement();

    gsInfo << "done.\n    Setup cg solver for Lagrange multipliers and solve... " << std::flush;

    gsMatrix<> lambda, errorHistory;
    real_t conditionNumber = 0;

    // Mean solve time over numRun repeats (as in solver_benchmark_example.cpp):
    // the Lagrange multiplier is reset to the same seeded vector each repeat,
    // so the PCG on the Schur complement is deterministic and only the timing
    // varies.
    double solveTime = 0;
    for (index_t run = 0; run < numRun; ++run)
    {
        std::srand(1);
        lambda.setRandom( ieti.nLagrangeMultipliers(), 1 );

        gsStopwatch timer;
        gsConjugateGradient<> PCG( ieti.schurComplement(), prec.preconditioner() );
        PCG.setOptions( cmd.getGroup("Solver") ).solveDetailed( rhsForSchur, lambda, errorHistory );
        solveTime += timer.stop();

        if (calcEigenvalues && run == numRun-1)
            conditionNumber = PCG.getConditionNumber();
    }
    solveTime /= numRun;

    gsInfo << "done. Solve time: " << solveTime << " s"
           << (numRun > 1 ? " (mean over " + std::to_string(numRun) + " repeats)" : "") << "\n"
           << "    Reconstruct solution from Lagrange multipliers... " << std::flush;
    std::vector<gsMatrix<>> uLocal = primal.distributePrimalSolution(
        ieti.constructSolutionFromLagrangeMultipliers(lambda)
    );
    gsMatrix<> uGlobal = ietiMapper.constructGlobalSolutionFromLocalSolutions(uLocal);
    gsInfo << "done.\n\n";

    /******************** Print end Exit ********************/

    const index_t iter = errorHistory.rows()-1;
    const bool success = errorHistory(iter,0) < tolerance;
    if (success)
        gsInfo << "Reached desired tolerance after " << iter << " iterations:\n";
    else
        gsInfo << "Did not reach desired tolerance after " << iter << " iterations:\n";

    if (errorHistory.rows() < 20)
        gsInfo << errorHistory.transpose() << "\n\n";
    else
        gsInfo << errorHistory.topRows(5).transpose() << " ... " << errorHistory.bottomRows(5).transpose()  << "\n\n";

    if (calcEigenvalues)
        gsInfo << "Estimated condition number: " << conditionNumber << "\n";

    if (!out.empty())
    {
        gsFileData<> fd;
        std::time_t time = std::time(NULL);
        fd.add(cmd);
        fd.add(uGlobal);
        fd.addComment(std::string("ieti_nn_example   Timestamp:")+std::ctime(&time));
        fd.save(out);
        gsInfo << "Write solution to file " << out << "\n";
    }

    if (plot)
    {
        gsInfo << "Write Paraview data to file ieti_nn_result.pvd\n";
        gsMultiPatch<> mpsol;
        for (index_t k=0; k<nPatches; ++k)
            mpsol.addPatch( mb[k].makeGeometry( ietiMapper.incorporateFixedPart(k, uLocal[k])  ) );
        gsWriteParaview<>( gsField<>( mp, mpsol ), "ieti_nn_result", 1000);
    }

    if (!plot&&out.empty())
    {
        gsInfo << "Done. No output created, re-run with --plot to get a ParaView "
                  "file containing the solution or --out to write solution to xml file.\n";
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
