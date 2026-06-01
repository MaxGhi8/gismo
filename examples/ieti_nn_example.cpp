/** @file ieti_nn_example.cpp

    @brief IETI solver with a neural-network Schur complement.

    Mirrors ieti_example.cpp (CG on the Schur complement formulation), but
    replaces each patch's local Schur complement operator S_k inside the
    scaled Dirichlet preconditioner with a learned model. One ONNX session
    is loaded per patch and bound to that patch's geometry features, then
    plugged into gsScaledDirichletPrec as the per-patch Schur operator.

    Model contract (the user supplies a compatible ONNX model):

        input  'input' : float[1, N]      <-- residual on local DOFs
        input  'obj.1' : float[1, N, 4]   <-- geometry features per DOF
                                                rows are (x, y, z, w)
                                                x,y,z : physical coords
                                                w     : NURBS weight (=1 for B-splines)
        output 'output': float[1, N]      <-- preconditioned vector

    Here N = mb[k].size() (number of local basis functions per patch).
    Because the scaled Dirichlet preconditioner expects a skeleton->skeleton
    operator, this file wraps the full-DOF NN inside gsNeuralSchurOp: zero-
    extend the skeleton input to the full local DOF vector, run the NN,
    restrict the result back to the skeleton.

    Build: requires ONNX Runtime; configure with
        cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
    Files matching *_nn_example.cpp are skipped automatically otherwise.

    Author(s): M. Ghiotto
*/

#include <ctime>
#include <gismo.h>
#include "gsNeuralPrec.h"

namespace gismo {

/// @brief Wraps a full-local-DOF neural operator as a skeleton->skeleton
/// Schur complement op suitable for gsScaledDirichletPrec::addSubdomain.
///
/// apply(v_skeleton):
///   1) zero-extend v_skeleton to a full local DOF vector
///   2) run the NN (forward Schur action)
///   3) restrict the result to the skeleton entries
template <class T>
class gsNeuralSchurOp : public gsLinearOperator<T>
{
public:
    typedef memory::shared_ptr<gsNeuralSchurOp> Ptr;

    gsNeuralSchurOp(typename gsNeuralPrec<T>::Ptr nn,
                    std::vector<index_t> skeletonDofs,
                    index_t nLocalDofs)
    : m_nn(nn),
      m_skeleton(std::move(skeletonDofs)),
      m_nLocal(nLocalDofs)
    {
        GISMO_ENSURE(m_nn->rows() == m_nLocal,
            "gsNeuralSchurOp: NN expects input size " << m_nn->rows()
            << " but patch has " << m_nLocal << " local DOFs. "
            "Provide a model whose input dimension matches mb[k].size().");
    }

    void apply(const gsMatrix<T> & input, gsMatrix<T> & x) const override
    {
        const index_t nSkel = static_cast<index_t>(m_skeleton.size());
        GISMO_ASSERT(input.rows() == nSkel,
            "gsNeuralSchurOp::apply: expected " << nSkel
            << " skeleton entries, got " << input.rows());

        gsMatrix<T> fullIn = gsMatrix<T>::Zero(m_nLocal, 1);
        for (index_t i = 0; i < nSkel; ++i)
            fullIn(m_skeleton[i], 0) = input(i, 0);

        gsMatrix<T> fullOut;
        m_nn->apply(fullIn, fullOut);

        x.resize(nSkel, 1);
        for (index_t i = 0; i < nSkel; ++i)
            x(i, 0) = fullOut(m_skeleton[i], 0);
    }

    index_t rows() const override { return static_cast<index_t>(m_skeleton.size()); }
    index_t cols() const override { return static_cast<index_t>(m_skeleton.size()); }

private:
    typename gsNeuralPrec<T>::Ptr m_nn;
    std::vector<index_t>          m_skeleton;
    index_t                       m_nLocal;
};

/// @brief Build the (4 x n_dofs) per-DOF geometry feature matrix for patch k.
///
/// Rows are (x, y, z, w):
///   x,y,z : physical coordinates of the basis function's Greville abscissa
///           (z = 0 in 2D);
///   w     : NURBS weight, = 1 for B-spline geometries.
///
/// The (4 x n_dofs) column-major layout matches an ONNX [1, n_dofs, 4]
/// row-major tensor (per-DOF feature vectors contiguous in memory), which
/// is what gsNeuralPrec::setAuxiliaryInput expects.
template <class T>
gsMatrix<T> computePatchGeometryFeatures(const gsMultiPatch<T> & mp,
                                         const gsMultiBasis<T> & mb,
                                         index_t k)
{
    gsMatrix<T> anchors = mb[k].anchors();         // (paramDim, n_dofs)
    gsMatrix<T> phys;
    mp.patch(k).eval_into(anchors, phys);          // (geoDim, n_dofs)

    const index_t n_dofs = mb[k].size();
    gsMatrix<T> features(4, n_dofs);
    features.setZero();
    features.topRows(phys.rows()) = phys;          // x, y, (z)
    features.row(3).setOnes();                     // w = 1 (B-spline)
    return features;
}

} // namespace gismo

using namespace gismo;

int main(int argc, char *argv[])
{
    /************** Define command line options *************/

    std::string geometry("domain2d/yeti_mp2.xml");
    std::string coeff("1.0");
    index_t splitPatches = 1;
    real_t stretchGeometry = 1;
    index_t refinements = 1;
    index_t degree = 2;
    std::string boundaryConditions("d");
    std::string primals("c");
    bool eliminateCorners = false;
    real_t tolerance = 1.e-8;
    index_t maxIterations = 100;
    bool calcEigenvalues = false;
    std::string out;
    bool plot = false;

    // NN-specific options
    std::string modelPath =
        GISMO_DATA_DIR "onnx_models/"
        "best_model_Transformer_homogeneous_neumann_l_0_deg_2_crazygeom_h40_H3_realSPD.onnx";
    std::string nnInputName  = "input";
    std::string nnOutputName = "output";
    std::string nnAuxName    = "obj.1";
    bool useCuda = false;

    gsCmdLine cmd("Solves a PDE with IETI, using a neural-network Schur complement inside the scaled Dirichlet preconditioner.");
    cmd.addString("g", "Geometry",              "Geometry file", geometry);
    cmd.addString("a", "Coeff",                 "Coefficient function a(x)", coeff);
    cmd.addInt   ("",  "SplitPatches",          "Split every patch that many times in 2^d patches", splitPatches);
    cmd.addReal  ("",  "StretchGeometry",       "Stretch geometry in x-direction by the given factor", stretchGeometry);
    cmd.addInt   ("r", "Refinements",           "Number of uniform h-refinement steps to perform before solving", refinements);
    cmd.addInt   ("p", "Degree",                "Degree of the B-spline discretization space", degree);
    cmd.addString("b", "BoundaryConditions",    "Boundary conditions", boundaryConditions);
    cmd.addString("c", "Primals",               "Primal constraints (c=corners, e=edges, f=faces)", primals);
    cmd.addSwitch("e", "EliminateCorners",      "Eliminate corners (if they are primals)", eliminateCorners);
    cmd.addReal  ("t", "Solver.Tolerance",      "Stopping criterion for linear solver", tolerance);
    cmd.addInt   ("",  "Solver.MaxIterations",  "Maximum iterations for linear solver", maxIterations);
    cmd.addSwitch("",  "Solver.CalcEigenvalues","Estimate eigenvalues based on Lanczos", calcEigenvalues);
    cmd.addString("m", "model",                 "Path to the ONNX model file", modelPath);
    cmd.addString("",  "nnInput",               "ONNX primary input tensor name", nnInputName);
    cmd.addString("",  "nnOutput",              "ONNX output tensor name", nnOutputName);
    cmd.addString("",  "nnAux",                 "ONNX auxiliary (geometry) input tensor name", nnAuxName);
    cmd.addSwitch(     "cuda",                  "Use CUDA execution provider", useCuda);
    cmd.addString("",  "out",                   "Write solution and used options to file", out);
    cmd.addSwitch(     "plot",                  "Plot the result with Paraview", plot);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }


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

    gsMultiBasis<> mb(mp);

    gsInfo << "Setup bases and adjust degree... " << std::flush;

    for ( size_t i = 0; i < mb.nBases(); ++ i )
        mb[i].setDegreePreservingMultiplicity(degree);

    for ( index_t i = 0; i < refinements; ++i )
        mb.uniformRefine();

    // Enforce square discretization across all patches (see ieti_example.cpp
    // for rationale). This guarantees mb[k].size() is the same for every k,
    // so a single ONNX model with one fixed input dimension can be used.
    {
        typedef gsTensorBSplineBasis<2,real_t> TBasis;
        const short_t dim = mp.geoDim();

        index_t globalMax = 0;
        for (size_t k = 0; k < mb.nBases(); ++k)
        {
            TBasis& tb = dynamic_cast<TBasis&>(mb[k]);
            for (short_t d = 0; d < dim; ++d)
                globalMax = std::max(globalMax, (index_t)tb.knots(d).numElements());
        }

        for (size_t k = 0; k < mb.nBases(); ++k)
        {
            TBasis& tb = dynamic_cast<TBasis&>(mb[k]);
            for (short_t d = 0; d < dim; ++d)
                while ((index_t)tb.knots(d).numElements() < globalMax)
                    mb[k].uniformRefine(1, 1, d);
        }
    }

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

        // --- Neural Schur complement for this patch ---------------------
        // Cheap: only allocates the per-patch I/O float buffers and the
        // Ort::Value views over them. The ORT session and model weights
        // are shared via nnModel.
        gsNeuralPrec<real_t>::Ptr nn = std::make_shared<gsNeuralPrec<real_t>>(
            nnModel, nnInputName, nnOutputName
        );

        gsMatrix<real_t> features = computePatchGeometryFeatures(mp, mb, k);
        nn->setAuxiliaryInput(nnAuxName, features);

        std::vector<index_t> skeletonDofs = ietiMapper.skeletonDofs(k);

        gsLinearOperator<>::Ptr nnSchurOp = std::make_shared<gsNeuralSchurOp<real_t>>(
            nn, skeletonDofs, mb[k].size()
        );

        prec.addSubdomain(
            gsScaledDirichletPrec<>::restrictJumpMatrix(jumpMatrix, skeletonDofs).moveToPtr(),
            nnSchurOp
        );
        // ----------------------------------------------------------------

        // primal.handleConstraints rewrites jumpMatrix/localMatrix/localRhs;
        // must run after prec.addSubdomain so the un-modified jump matrix
        // is what we restricted above.
        primal.handleConstraints(
            ietiMapper.primalConstraints(k),
            ietiMapper.primalDofIndices(k),
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
    gsMatrix<> lambda;
    lambda.setRandom( ieti.nLagrangeMultipliers(), 1 );

    gsMatrix<> errorHistory;

    gsConjugateGradient<> PCG( ieti.schurComplement(), prec.preconditioner() );
    gsStopwatch timer;
    PCG.setOptions( cmd.getGroup("Solver") ).solveDetailed( rhsForSchur, lambda, errorHistory );
    const double solveTime = timer.stop();

    gsInfo << "done. Solve time: " << solveTime << " s\n"
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
        gsInfo << "Estimated condition number: " << PCG.getConditionNumber() << "\n";

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
