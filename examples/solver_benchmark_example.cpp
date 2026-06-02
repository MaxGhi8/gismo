/** @file solver_benchmark_example.cpp

    @brief Fair benchmark of several linear solvers on one and the same
    isogeometric Poisson discretization.

    The same problem (geometry, basis, boundary conditions, right-hand side)
    is solved with:
      - a direct sparse Cholesky solver (reference),
      - preconditioned conjugate gradient (CG) with several preconditioners
        (none, Jacobi, symmetric Gauss-Seidel, Richardson, incomplete LU),
      - geometric multigrid, both as a standalone solver and as a
        preconditioner for CG,
      - the IETI-DP solver (CG on the Schur complement with the scaled
        Dirichlet preconditioner, as in ieti_example.cpp).

    All iterative methods share the same stopping criteria, so the comparison
    is fair. The manufactured exact solution u = sin(x)*cos(y) gives a
    numbering-independent L2-error column that confirms every method solved
    the very same discretized problem.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): M. Ghiotto
*/

#include <cstdlib>
#include <iomanip>
#include <set>
#include <string>
#include <vector>

#include <gismo.h>

using namespace gismo;

// A single benchmark record. iters == -1 marks the direct solver (no iterations).
struct BenchResult
{
    std::string name;
    index_t     iters;
    double      setupTime;
    double      solveTime;
    real_t      l2err;
    bool        converged;
};

// Numbering-independent accuracy measure shared by every method: the L2 distance
// between the reconstructed solution field and the manufactured exact solution.
// (Idiom from biharmonic_example.cpp.)
real_t l2ErrorVsExact(const gsMultiPatch<>& mp,
                      const gsMultiPatch<>& sol,
                      const gsFunctionExpr<>& uExact)
{
    gsField<> solField(mp, sol);
    return solField.distanceL2(uExact, false);
}

// Shared CG driver for the global system. Builds the result record (including the
// L2 error of the reconstructed field) for the given preconditioner.
BenchResult runGlobalCG(const std::string&            name,
                        const gsSparseMatrix<>&       K,
                        const gsMatrix<>&             F,
                        const gsLinearOperator<>::Ptr& prec,
                        const gsOptionList&           solverOpt,
                        const gsMultiPatch<>&         mp,
                        gsPoissonAssembler<>&         assembler,
                        const gsFunctionExpr<>&       uExact,
                        double                        setupTime)
{
    const real_t tol = solverOpt.getReal("Tolerance");

    // Identical, identically-seeded initial guess for every global solver.
    std::srand(1);
    gsMatrix<> x;
    x.setRandom(K.rows(), 1);

    gsMatrix<> errorHistory;
    gsStopwatch timer;
    gsConjugateGradient<>(K, prec)
        .setOptions(solverOpt)
        .solveDetailed(F, x, errorHistory);
    const double solveTime = timer.stop();

    const index_t iters    = errorHistory.rows() - 1;
    const bool    converged = errorHistory(iters, 0) < tol;

    gsMultiPatch<> sol;
    assembler.constructSolution(x, sol);
    const real_t l2err = l2ErrorVsExact(mp, sol, uExact);

    BenchResult r;
    r.name = name; r.iters = iters; r.setupTime = setupTime;
    r.solveTime = solveTime; r.l2err = l2err; r.converged = converged;
    return r;
}

int main(int argc, char *argv[])
{
    /************** Define command line options *************/

    std::string geometry("domain2d/yeti_mp2.xml");
    index_t splitPatches = 1;
    real_t  stretchGeometry = 1;
    index_t refinements = 1;
    index_t degree = 2;
    std::string boundaryConditions("d");
    std::string primals("c");
    bool    eliminateCorners = false;
    real_t  tolerance = 1.e-8;
    index_t maxIterations = 1000;
    std::string mgSmoother("GaussSeidel");
    index_t mgLevels = -1;

    gsCmdLine cmd("Fair benchmark of several linear solvers on one isogeometric Poisson discretization.");
    cmd.addString("g", "Geometry",              "Geometry file", geometry);
    cmd.addInt   ("",  "SplitPatches",          "Split every patch that many times in 2^d patches", splitPatches);
    cmd.addReal  ("",  "StretchGeometry",       "Stretch geometry in x-direction by the given factor", stretchGeometry);
    cmd.addInt   ("r", "Refinements",           "Number of uniform h-refinement steps to perform before solving", refinements);
    cmd.addInt   ("p", "Degree",                "Degree of the B-spline discretization space", degree);
    cmd.addString("b", "BoundaryConditions",    "Boundary conditions", boundaryConditions);
    cmd.addString("c", "Primals",               "IETI primal constraints (c=corners, e=edges, f=faces)", primals);
    cmd.addSwitch("e", "EliminateCorners",      "IETI: eliminate corners (if they are primals)", eliminateCorners);
    cmd.addReal  ("t", "Solver.Tolerance",      "Stopping criterion for all iterative solvers", tolerance);
    cmd.addInt   ("",  "Solver.MaxIterations",  "Maximum iterations for all iterative solvers", maxIterations);
    cmd.addString("s", "MG.Smoother",           "Multigrid smoother (Richardson, Jacobi, GaussSeidel, IncompleteLU)", mgSmoother);

    // Multigrid sub-options consumed by gsGridHierarchy / gsMultiGridOp.
    cmd.addInt   ("l", "MG.Levels",             "Number of multigrid levels (default: = Refinements)", mgLevels);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

    // Default case is levels := refinements, so replace the invalid default.
    if (mgLevels < 0) { mgLevels = refinements; cmd.setInt("MG.Levels", mgLevels); }

    if ( ! gsFileManager::fileExists(geometry) )
    {
        gsInfo << "Geometry file could not be found.\n";
        gsInfo << "I was searching in the current directory and in: " << gsFileManager::getSearchPaths() << "\n";
        return EXIT_FAILURE;
    }

    gsInfo << "Run solver_benchmark_example with options:\n" << cmd << std::endl;

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

    // Right-hand-side and Dirichlet data chosen so that the exact solution is
    // u = sin(x)*cos(y)  (since -Laplace(sin x cos y) = 2 sin x cos y).
    gsFunctionExpr<> f ( "2*sin(x)*cos(y)", mp.geoDim() );
    gsFunctionExpr<> gD( "sin(x)*cos(y)",   mp.geoDim() );
    gsConstantFunction<> gN( 1.0, mp.geoDim() );
    gsFunctionExpr<> uExact( "sin(x)*cos(y)", mp.geoDim() );

    // The manufactured solution u = sin(x)*cos(y) is consistent with the
    // hardcoded f and gD on ANY domain, so the L2-error column stays meaningful
    // for every geometry/refinement/degree. It is NOT consistent with the
    // constant Neumann data gN = 1.0, however: as soon as any boundary is
    // Neumann, the discrete problem no longer matches uExact and the L2 error is
    // meaningless. We track that here and suppress the column in that case.
    bool hasNeumann = false;

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
            {
                bc.addCondition( *it, condition_type::neumann, &gN );
                hasNeumann = true;
            }
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

    // Enforce a square discretization: some input patches start with a different
    // number of knot spans per direction (e.g. yeti_mp2.xml). Refine every patch
    // in every direction until it matches the global maximum span count. (Carried
    // over verbatim from ieti_example.cpp so the IETI setup is identical.)
    {
        const short_t dim = mp.domainDim();
        index_t globalMax = 0;
        for (size_t k = 0; k < mb.nBases(); ++k)
        {
            const index_t total = mb[k].numElements();
            for (short_t d = 0; d < dim; ++d)
            {
                const index_t side_elements = mb[k].numElements(boxSide(d, 0));
                const index_t n_dir = total / side_elements;
                globalMax = std::max(globalMax, n_dir);
            }
        }

        for (size_t k = 0; k < mb.nBases(); ++k)
        {
            for (short_t d = 0; d < dim; ++d)
            {
                while (true)
                {
                    const index_t total = mb[k].numElements();
                    const index_t side_elements = mb[k].numElements(boxSide(d, 0));
                    const index_t n_dir = total / side_elements;
                    if (n_dir < globalMax)
                        mb[k].uniformRefine(1, 1, d);
                    else
                        break;
                }
            }
        }
    }

    gsInfo << "done.\n";

    for ( size_t i = 0; i < mb.nBases(); ++ i )
    {
        gsInfo << "Patch " << i << ": Degree " << mb[i].degree(0);
        for (short_t d = 1; d < mb[i].domainDim(); ++d) gsInfo << "x" << mb[i].degree(d);
        gsInfo << ", " << mb[i].size() << " basis functions.\n";
    }

    const gsOptionList solverOpt = cmd.getGroup("Solver");

    std::vector<BenchResult> results;

    /******** Assemble the global system (single source of truth) ********/

    gsInfo << "\nAssemble global system (gsPoissonAssembler)... " << std::flush;
    gsStopwatch timer;
    gsPoissonAssembler<> assembler(
        mp, mb, bc, f,
        dirichlet::elimination,
        iFace::glue
    );
    assembler.assemble();
    const double globalAssembleTime = timer.stop();
    const gsSparseMatrix<>& K = assembler.matrix();
    const gsMatrix<>&       F = assembler.rhs();
    gsInfo << "done (" << globalAssembleTime << " s, " << K.rows() << " dofs).\n";

    /**************** Direct solver (reference) ****************/

    gsInfo << "Solve: direct Cholesky... " << std::flush;
    {
        timer.restart();
        gsSparseSolver<>::SimplicialLDLT solver;
        solver.compute(K);
        gsMatrix<> x = solver.solve(F);
        const double solveTime = timer.stop();

        gsMultiPatch<> sol;
        assembler.constructSolution(x, sol);
        BenchResult r;
        r.name = "Direct (Cholesky)"; r.iters = -1; r.setupTime = globalAssembleTime;
        r.solveTime = solveTime; r.l2err = l2ErrorVsExact(mp, sol, uExact);
        r.converged = true;
        results.push_back(r);
    }
    gsInfo << "done.\n";

    /**************** Preconditioned CG variants ****************/

    // Each preconditioner is a gsLinearOperator applied once per CG iteration.
    // For CG validity the preconditioner must be SPD: identity, Jacobi, Richardson
    // and symmetric Gauss-Seidel qualify. ILU is included because the user asked
    // for it; it is the gismo-provided option but is not guaranteed SPD.
    gsInfo << "Solve: CG variants... " << std::flush;

    results.push_back(runGlobalCG("CG (no prec)", K, F,
        gsIdentityOp<>::make(K.rows()), solverOpt, mp, assembler, uExact, globalAssembleTime));

    results.push_back(runGlobalCG("CG + Jacobi", K, F,
        makeJacobiOp(K), solverOpt, mp, assembler, uExact, globalAssembleTime));

    results.push_back(runGlobalCG("CG + sym. Gauss-Seidel (SSOR)", K, F,
        makeSymmetricGaussSeidelOp(K), solverOpt, mp, assembler, uExact, globalAssembleTime));

    results.push_back(runGlobalCG("CG + Richardson", K, F,
        makeRichardsonOp(K), solverOpt, mp, assembler, uExact, globalAssembleTime));

    {
        timer.restart();
        gsLinearOperator<>::Ptr iluPrec = makeIncompleteLUOp(K);
        const double iluSetup = globalAssembleTime + timer.stop();
        results.push_back(runGlobalCG("CG + ILU", K, F,
            iluPrec, solverOpt, mp, assembler, uExact, iluSetup));
    }
    gsInfo << "done.\n";

    /**************** Multigrid (standalone + as CG preconditioner) ****************/

    gsInfo << "Solve: multigrid (smoother: " << mgSmoother << ")... " << std::flush;
    {
        timer.restart();
        const gsOptionList mgOpt = cmd.getGroup("MG");

        // buildByCoarsening consumes its multibasis argument, so feed it a copy.
        std::vector< gsSparseMatrix<real_t,RowMajor> > transferMatrices;
        gsGridHierarchy<>::buildByCoarsening(gsMultiBasis<>(mb), bc, mgOpt)
            .moveTransferMatricesTo(transferMatrices);

        gsMultiGridOp<>::Ptr mg = gsMultiGridOp<>::make(K, transferMatrices);
        mg->setOptions(mgOpt);
        mg->setCoarseSolver( makeSparseCholeskySolver(mg->matrix(0)) );

        for (index_t i = 1; i < mg->numLevels(); ++i)
        {
            gsPreconditionerOp<>::Ptr smootherOp;
            if ( mgSmoother == "Richardson" || mgSmoother == "r" )
                smootherOp = makeRichardsonOp(mg->matrix(i));
            else if ( mgSmoother == "Jacobi" || mgSmoother == "j" )
                smootherOp = makeJacobiOp(mg->matrix(i));
            else if ( mgSmoother == "GaussSeidel" || mgSmoother == "gs" )
                smootherOp = makeGaussSeidelOp(mg->matrix(i));
            else if ( mgSmoother == "IncompleteLU" || mgSmoother == "ilu" )
                smootherOp = makeIncompleteLUOp(mg->matrix(i));
            else
            {
                gsInfo << "\nUnknown multigrid smoother '" << mgSmoother
                       << "'. Known: Richardson, Jacobi, GaussSeidel, IncompleteLU.\n";
                return EXIT_FAILURE;
            }
            mg->setSmoother(i, smootherOp);
        }
        const double mgSetup = globalAssembleTime + timer.stop();

        // (a) Multigrid as a standalone (stationary) solver.
        {
            std::srand(1);
            gsMatrix<> x; x.setRandom(K.rows(), 1);
            gsMatrix<> errorHistory;
            timer.restart();
            gsGradientMethod<>(K, mg)
                .setOptions(solverOpt)
                .solveDetailed(F, x, errorHistory);
            const double solveTime = timer.stop();

            const index_t iters     = errorHistory.rows() - 1;
            const bool    converged = errorHistory(iters, 0) < tolerance;
            gsMultiPatch<> sol; assembler.constructSolution(x, sol);
            BenchResult r;
            r.name = "Multigrid (standalone)"; r.iters = iters; r.setupTime = mgSetup;
            r.solveTime = solveTime; r.l2err = l2ErrorVsExact(mp, sol, uExact);
            r.converged = converged;
            results.push_back(r);
        }

        // (b) The same multigrid operator as a preconditioner for CG.
        results.push_back(runGlobalCG("CG + multigrid", K, F,
            mg, solverOpt, mp, assembler, uExact, mgSetup));
    }
    gsInfo << "done.\n";

    /**************** IETI-DP (ported from ieti_example.cpp) ****************/

    gsInfo << "Solve: IETI-DP... " << std::flush;
    {
        const index_t nPatches = mp.nPatches();
        timer.restart();

        gsIetiMapper<> ietiMapper;
        {
            typedef gsExprAssembler<>::space space;
            gsExprAssembler<> exprAssembler;
            space u = exprAssembler.getSpace(mb);
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
                    gsInfo << "\nUnknown type of primal constraint: \"" << primals[i] << "\"\n";
                    return EXIT_FAILURE;
            }

        if (cornersAsPrimals) ietiMapper.cornersAsPrimals();
        if (edgesAsPrimals)   ietiMapper.interfaceAveragesAsPrimals(mp,1);
        if (facesAsPrimals)   ietiMapper.interfaceAveragesAsPrimals(mp,2);

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

        // NOTE: this assembly loop is intentionally serial. The analogous loop in
        // ieti_example.cpp is OpenMP-parallel, but that contains a data race that
        // corrupts the assembled subdomain data (masked there because the example
        // only reconstructs the solution under --plot and never measures its
        // error). Since this benchmark reports the L2 error of the reconstructed
        // IETI solution, we keep the setup deterministic and correct. This loop is
        // setup, not the measured solve, so its serial cost is reported separately.
        for (index_t k=0; k<nPatches; ++k)
        {
            gsBoundaryConditions<> bc_local;
            bc.getConditionsForPatch(k,bc_local);
            gsMultiPatch<> mp_local = mp[k];
            gsMultiBasis<> mb_local = mb[k];

            typedef gsExprAssembler<>::geometryMap geometryMap;
            typedef gsExprAssembler<>::variable    variable;
            typedef gsExprAssembler<>::space       space;

            gsExprAssembler<> exprAssembler(1,1);
            exprAssembler.setIntegrationDomain(mb_local.domain());
            gsExprEvaluator<> ev(exprAssembler);

            geometryMap G = exprAssembler.getMap(mp_local);
            space u = exprAssembler.getSpace(mb_local);

            bc_local.setGeoMap(mp_local);
            u.setup(bc_local, dirichlet::interpolation, 0);
            ietiMapper.initFeSpace(u,k);

            auto ff = exprAssembler.getCoeff(f, G);
            exprAssembler.initSystem();
            exprAssembler.assemble( igrad(u, G) * igrad(u, G).tr() * meas(G), u * ff * meas(G) );

            variable g_N = exprAssembler.getBdrFunction();
            exprAssembler.assembleBdr(bc_local.get("Neumann"), u * g_N.val() * nv(G).norm() );

            gsSparseMatrix<real_t, RowMajor> jumpMatrix  = ietiMapper.jumpMatrix(k);
            gsSparseMatrix<>                 localMatrix = exprAssembler.matrix();
            gsMatrix<>                       localRhs    = exprAssembler.rhs();

            {
                prec.addSubdomain(
                    gsScaledDirichletPrec<>::restrictToSkeleton(
                        jumpMatrix, localMatrix, ietiMapper.skeletonDofs(k)
                    )
                );

                auto const & pConstraints = ietiMapper.primalConstraints(k);
                auto const & pDofIndices  = ietiMapper.primalDofIndices(k);

                std::vector<gsSparseVector<real_t>> uniqueConstraints;
                std::vector<index_t> uniqueDofIndices;
                std::set<index_t> seen;
                for (size_t i = 0; i < pDofIndices.size(); ++i)
                {
                    if (seen.find(pDofIndices[i]) == seen.end())
                    {
                        uniqueConstraints.push_back(pConstraints[i]);
                        uniqueDofIndices.push_back(pDofIndices[i]);
                        seen.insert(pDofIndices[i]);
                    }
                }

                primal.handleConstraints(
                    uniqueConstraints, uniqueDofIndices, jumpMatrix, localMatrix, localRhs
                );

                ieti.addSubdomain(
                    jumpMatrix.moveToPtr(),
                    makeMatrixOp(localMatrix.moveToPtr()),
                    give(localRhs)
                );
            }
        } // end for

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

        prec.setupMultiplicityScaling();
        gsMatrix<> rhsForSchur = ieti.rhsForSchurComplement();
        const double ietiSetup = timer.stop();

        std::srand(1);
        gsMatrix<> lambda;
        lambda.setRandom( ieti.nLagrangeMultipliers(), 1 );

        gsMatrix<> errorHistory;
        timer.restart();
        gsConjugateGradient<> PCG( ieti.schurComplement(), prec.preconditioner() );
        PCG.setOptions( solverOpt ).solveDetailed( rhsForSchur, lambda, errorHistory );
        const double ietiSolve = timer.stop();

        std::vector<gsMatrix<>> uLocal = primal.distributePrimalSolution(
            ieti.constructSolutionFromLagrangeMultipliers(lambda)
        );

        // Reconstruct the global multipatch field for the L2 error.
        gsMultiPatch<> sol;
        for (index_t k=0; k<nPatches; ++k)
            sol.addPatch( mb[k].makeGeometry( ietiMapper.incorporateFixedPart(k, uLocal[k]) ) );

        const index_t iters     = errorHistory.rows() - 1;
        const bool    converged = errorHistory(iters, 0) < tolerance;
        BenchResult r;
        r.name = "IETI-DP (CG on Schur)"; r.iters = iters; r.setupTime = ietiSetup;
        r.solveTime = ietiSolve; r.l2err = l2ErrorVsExact(mp, sol, uExact);
        r.converged = converged;
        results.push_back(r);
    }
    gsInfo << "done.\n";

    /******************** Print benchmark table ********************/

    // The L2-vs-exact column is only an accuracy measure when the discrete
    // problem actually matches the manufactured solution. With Neumann data
    // gN = 1.0 (inconsistent with uExact) it is not, so we disable it there.
    const bool l2Meaningful = !hasNeumann;

    gsInfo << "\n=================================== Benchmark results ===================================\n";
    gsInfo << std::left << std::setw(32) << "Method"
           << std::right << std::setw(11) << "setup [s]"
           << std::setw(11) << "solve [s]"
           << std::setw(8)  << "iters"
           << std::setw(14) << "L2 error"
           << std::setw(11) << "converged" << "\n";
    gsInfo << "-----------------------------------------------------------------------------------------\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const BenchResult& r = results[i];
        gsInfo << std::left << std::setw(32) << r.name
               << std::right << std::fixed << std::setprecision(4)
               << std::setw(11) << r.setupTime
               << std::setw(11) << r.solveTime;
        if (r.iters < 0)
            gsInfo << std::setw(8) << "-";
        else
            gsInfo << std::setw(8) << r.iters;
        if (l2Meaningful)
            gsInfo << std::scientific << std::setprecision(3) << std::setw(14) << r.l2err;
        else
            gsInfo << std::setw(14) << "n/a";
        gsInfo << std::setw(11) << (r.converged ? "yes" : "NO") << "\n";
    }
    gsInfo << "=========================================================================================\n";
    gsInfo << "Note: IETI-DP iterations count CG steps on the Schur complement (a different operator\n"
              "      and dimension), so iteration counts are directly comparable only among the\n"
              "      global-system solvers.\n";
    if (l2Meaningful)
        gsInfo << "      The matching L2-error column confirms that every method solved the same\n"
                  "      discretized problem.\n";
    else
        gsInfo << "      The L2-error column is disabled (shown as 'n/a'): with Neumann boundary\n"
                  "      conditions the constant Neumann data is inconsistent with the manufactured\n"
                  "      solution u = sin(x)*cos(y), so the L2 distance to it is not meaningful.\n";

    return EXIT_SUCCESS;
}
