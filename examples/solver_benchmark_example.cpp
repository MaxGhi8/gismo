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

// Helper tag to pass types to the benchmark loop lambda.
template<typename T> struct solver_tag { typedef T type; };

// A single benchmark record. iters == -1 marks the direct solver (no iterations).
// The reconstructed solution field is retained so that, when the manufactured
// exact solution is not applicable (curved surface or Neumann data), the methods
// can still be cross-checked against a common reference solution.
struct BenchResult
{
    std::string    name;
    index_t        iters;
    double         setupTime;
    double         solveTime;
    real_t         l2err;
    bool           converged;
    gsMultiPatch<> sol;
};

// Numbering-independent L2 distance between two reconstructed solution fields on
// the same geometry. Used as a method-agreement check when the manufactured
// exact solution is not a valid reference (see l2Meaningful below).
real_t l2FieldDistance(const gsMultiPatch<>& mp,
                       const gsMultiPatch<>& a,
                       const gsMultiPatch<>& b)
{
    gsField<> fa(mp, a);
    gsField<> fb(mp, b);
    return fa.distanceL2(fb);
}

// Subtract the mean of the control values from a scalar solution field. For a
// partition-of-unity basis (B-splines, NURBS) the constant function equals the
// constant control value, so this removes the additive constant from the field
// itself. Used to compare solutions of a pure-Neumann (singular) problem, whose
// solution is only defined up to a constant: different solvers pick different
// constants (a direct solve of the singular system even returns a huge one), but
// the non-constant part is unique, so comparing modulo the constant reveals the
// genuine agreement.
gsMultiPatch<> removeConstant(gsMultiPatch<> sol)
{
    real_t  sum = 0;
    index_t n   = 0;
    for (size_t p = 0; p < sol.nPatches(); ++p)
    {
        sum += sol.patch(p).coefs().sum();
        n   += sol.patch(p).coefs().size();
    }
    const real_t mean = (n > 0) ? sum / n : 0;
    for (size_t p = 0; p < sol.nPatches(); ++p)
        sol.patch(p).coefs().array() -= mean;
    return sol;
}

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

// Reconstruct the global multipatch solution from a solution vector using the
// expression-assembler space (which carries the eliminated Dirichlet values).
// Unlike gsAssembler/gsPoissonAssembler, the expression assembler computes the
// physical gradient with the Moore-Penrose pseudo-inverse of the Jacobian, so it
// is correct for surfaces embedded in 3D (manifolds), not only for square maps.
gsMultiPatch<> reconstructGlobal(const gsExprAssembler<>&        A,
                                 const gsExprAssembler<>::space& u,
                                 gsMatrix<>&                     x)
{
    gsMultiPatch<> sol;
    gsExprAssembler<>::solution u_sol = A.getSolution(u, x);
    u_sol.extract(sol);
    return sol;
}

// Shared iterative solver driver for the global system. Builds the result record
// (including the L2 error of the reconstructed field) for the given solver type
// and preconditioner.
template<typename SolverType>
BenchResult runGlobalSolver(const std::string&             name,
                            const gsSparseMatrix<>&        K,
                            const gsMatrix<>&              F,
                            const gsLinearOperator<>::Ptr& prec,
                            const gsOptionList&            solverOpt,
                            const gsMultiPatch<>&          mp,
                            const gsExprAssembler<>&       A,
                            const gsExprAssembler<>::space& u,
                            const gsFunctionExpr<>&        uExact,
                            double                         setupTime)
{
    const real_t tol = solverOpt.getReal("Tolerance");

    // Identical, identically-seeded initial guess for every global solver.
    std::srand(1);
    gsMatrix<> x;
    x.setRandom(K.rows(), 1);

    gsMatrix<> errorHistory;
    gsStopwatch timer;
    SolverType(K, prec)
        .setOptions(solverOpt)
        .solveDetailed(F, x, errorHistory);
    const double solveTime = timer.stop();

    const index_t iters    = errorHistory.rows() - 1;
    const bool    converged = (iters >= 0) && (errorHistory(iters, 0) < tol);

    gsMultiPatch<> sol = reconstructGlobal(A, u, x);
    const real_t l2err = l2ErrorVsExact(mp, sol, uExact);

    BenchResult r;
    r.name = name; r.iters = iters; r.setupTime = setupTime;
    r.solveTime = solveTime; r.l2err = l2err; r.converged = converged;
    r.sol = give(sol);
    return r;
}

// Elements (knot spans) of patch k of mb in parameter direction d.
static index_t elementsInDir(const gsMultiBasis<>& mb, size_t k, short_t d)
{
    return mb[k].numElements() / mb[k].numElements(boxSide(d, 0));
}

// Build the benchmark discretization basis: extract it from the geometry, set the
// degree, and refine uniformly (-r). The "square-discretization" mode then controls
// the element count per direction:
//   "off"    - keep the native -p/-r knots (lightest). Works only if the input is
//              already interface-conforming.
//   "global" - refine every patch/direction up to the global maximum element count,
//              making every patch identical (heaviest; always conforming). Each level
//              doubles all directions, so the buildByRefinement invariant holds.
gsMultiBasis<> makeBenchBasis(const gsMultiPatch<>& mp, index_t degree,
                              index_t refinements, const std::string& squareMode)
{
    gsMultiBasis<> mb(mp);

    for ( size_t i = 0; i < mb.nBases(); ++ i )
        mb[i].setDegreePreservingMultiplicity(degree);

    for ( index_t i = 0; i < refinements; ++i )
        mb.uniformRefine();

    if (squareMode != "global")
        return mb;

    const short_t dim = mp.domainDim();
    index_t target = 0;
    for (size_t k = 0; k < mb.nBases(); ++k)
        for (short_t d = 0; d < dim; ++d)
            target = std::max(target, elementsInDir(mb, k, d));

    for (size_t k = 0; k < mb.nBases(); ++k)
        for (short_t d = 0; d < dim; ++d)
            while (elementsInDir(mb, k, d) < target)
                mb[k].uniformRefine(1, 1, d);

    return mb;
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
    std::string squareDiscr("global");

    gsCmdLine cmd("Fair benchmark of several linear solvers on one isogeometric Poisson discretization.");
    cmd.addString("g", "Geometry",              "Geometry file", geometry);
    cmd.addInt   ("",  "SplitPatches",          "Split every patch that many times in 2^d patches", splitPatches);
    cmd.addReal  ("",  "StretchGeometry",       "Stretch geometry in x-direction by the given factor", stretchGeometry);
    cmd.addInt   ("r", "Refinements",           "Number of uniform h-refinement steps to perform before solving", refinements);
    cmd.addInt   ("p", "Degree",                "Degree of the B-spline discretization space", degree);
    cmd.addString("",  "SquareDiscretization",  "Element equalisation: off (native, lightest, conforming inputs only) | global (all patches identical, always safe)", squareDiscr);
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

    if (squareDiscr != "off" && squareDiscr != "global")
    {
        gsInfo << "Invalid --SquareDiscretization '" << squareDiscr
               << "'. Use one of: off, global.\n";
        return EXIT_FAILURE;
    }
    if (squareDiscr == "off")
        gsWarn << "SquareDiscretization=off keeps the native (lightest) discretization. "
                  "It is only valid if the input patches are interface-conforming (true for "
                  "well-formed XML multipatches, e.g. yeti). On a non-conforming CAD import "
                  "(e.g. an IGES surface) the IETI-DP solver will fail; use 'repair' or 'global'.\n";

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

    // Domains stored as XML carry their topology (interfaces + boundary) in the
    // file, but CAD formats like IGES/STEP do not: the patches arrive as a bag of
    // surfaces with no connectivity. Without topology, mp has no registered
    // boundary sides, so no Dirichlet conditions get assigned below and the global
    // Poisson system becomes pure-Neumann / singular -- its "solution" is then an
    // arbitrary null-space vector (~1e+14). Recover the topology by geometric
    // matching when the file did not provide it: matching sides become interfaces
    // (gluing the patches), unmatched sides become boundary (and get Dirichlet).
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

    // Ensure a usable topology. Domains stored as XML carry interfaces + boundary
    // in the file; CAD formats like IGES/STEP arrive as a bag of surfaces with no
    // connectivity, and uniformSplit() on such an input drops the boundary sides
    // (it registers the new internal interfaces but leaves nBoundary == 0). With no
    // boundary there are no Dirichlet conditions, so the global Poisson system is
    // pure-Neumann / singular and its "solution" is an arbitrary null-space vector
    // (~1e+14). Recompute the topology by geometric matching whenever no boundary
    // side is registered: matching sides become interfaces (gluing patches),
    // unmatched sides become boundary (and thus receive Dirichlet data).
    if (mp.nBoundary() == 0)
    {
        gsInfo << "(no boundary registered, computing topology) " << std::flush;
        mp.computeTopology();
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
    // If no Dirichlet condition is set anywhere (a closed surface has no boundary,
    // or the user asked for all-Neumann), the global Poisson system is pure Neumann
    // and hence singular: its solution is only defined up to an additive constant.
    bool hasDirichlet = false;

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
            {
                bc.addCondition( *it, condition_type::dirichlet, &gD );
                hasDirichlet = true;
            }
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

    gsInfo << "Setup bases and adjust degree... " << std::flush;

    // mb is the assembled (finest) basis; --SquareDiscretization controls the
    // per-direction element equalisation (see makeBenchBasis).
    gsMultiBasis<> mb = makeBenchBasis(mp, degree, refinements, squareDiscr);

    gsInfo << "done.\n";

    // Summarise the discretization instead of printing one line per patch (a
    // CAD import can have thousands of patches). Report the per-direction degree,
    // number of knots and basis-function count for a representative patch, and the
    // total over all patches; flag whether every patch shares the same structure.
    {
        const short_t dim = mb[0].domainDim();
        bool uniform = true;
        for (size_t k = 1; k < mb.nBases() && uniform; ++k)
        {
            if (mb[k].domainDim() != dim || mb[k].size() != mb[0].size())
                uniform = false;
            for (short_t d = 0; d < dim && uniform; ++d)
                if (mb[k].degree(d) != mb[0].degree(d) ||
                    mb[k].component(d).size() != mb[0].component(d).size())
                    uniform = false;
        }

        index_t totalDofs = 0;
        for (size_t k = 0; k < mb.nBases(); ++k) totalDofs += mb[k].size();

        gsInfo << mb.nBases() << " patches"
               << (uniform ? " (all identical)" : " (patches differ; showing patch 0)") << ".\n";
        gsInfo << "  Patch 0: degree ";
        for (short_t d = 0; d < dim; ++d) gsInfo << (d ? "x" : "") << mb[0].degree(d);
        gsInfo << ", knots ";
        for (short_t d = 0; d < dim; ++d)
            gsInfo << (d ? "x" : "") << (mb[0].component(d).size() + mb[0].degree(d) + 1);
        gsInfo << ", " << mb[0].size() << " basis functions.\n";
        // NB: this sum counts each shared interface basis function once per patch and
        // does NOT eliminate Dirichlet dofs, so it is the un-glued (IETI-style local)
        // total -- larger than the coupled global system that direct/CG/multigrid
        // solve. The coupled size is reported on the assembler line below.
        gsInfo << "  Sum over all patches (un-glued, before interface gluing/BC "
                  "elimination): " << totalDofs << ".\n";
    }

    const gsOptionList solverOpt = cmd.getGroup("Solver");

    std::vector<BenchResult> results;

    /******** Assemble the global system (single source of truth) ********/

    // Assembled with the expression-template engine (gsExprAssembler). Its
    // physical-gradient transform uses the Moore-Penrose pseudo-inverse of the
    // Jacobian and is therefore correct on surfaces embedded in 3D (manifolds).
    // The classical gsPoissonAssembler/gsAssembler path instead uses
    // jacobian.cramerInverse(), which is valid only for a square Jacobian
    // (domainDim == geoDim) and silently builds a garbage stiffness matrix on a
    // manifold. The weak form, Dirichlet strategy and Neumann term below are
    // identical to the per-patch IETI assembly, so every method discretizes one
    // and the same problem on every geometry.
    typedef gsExprAssembler<>::geometryMap geometryMap;
    typedef gsExprAssembler<>::space       space;
    typedef gsExprAssembler<>::variable    variable;

    gsInfo << "\nAssemble global system (gsExprAssembler)... " << std::flush;
    gsStopwatch timer;

    const bool singular = !hasDirichlet;

    gsExprAssembler<> A(1,1);
    A.setIntegrationElements(mb);
    geometryMap G = A.getMap(mp);
    space       u = A.getSpace(mb);

    bc.setGeoMap(mp);
    u.setup(bc, dirichlet::interpolation, 0);

    // Pure-Neumann (closed surface, no Dirichlet) Poisson is solvable only if the
    // source is compatible, integral_S f dS = 0. The manufactured f does not satisfy
    // that, so subtract its surface average. We modify f itself, so the global
    // assembler AND the per-patch IETI assembly below both become compatible and
    // hence solve the same well-posed problem (unique up to a constant, which the
    // zero-mean comparison fixes).
    if (singular)
    {
        gsExprEvaluator<> ev(A);
        auto fv = ev.getVariable(f, G);
        const real_t cf = ev.integral(fv * meas(G)) / ev.integral(meas(G));
        f = gsFunctionExpr<>("2*sin(x)*cos(y) - (" + std::to_string(cf) + ")", mp.geoDim());
        gsInfo << "[pure Neumann] subtracting mean(f)=" << cf
               << " to make the RHS compatible... " << std::flush;
    }

    A.initSystem();
    auto ff = A.getCoeff(f, G);
    A.assemble( igrad(u, G) * igrad(u, G).tr() * meas(G), u * ff * meas(G) );

    variable g_N = A.getBdrFunction();
    A.assembleBdr( bc.get("Neumann"), u * g_N.val() * nv(G).norm() );

    // For the singular (pure-Neumann) case, pin one dof so the assembled system is
    // SPD and every solver works (the multigrid coarse Cholesky in particular needs
    // a non-singular matrix). The fixed gauge (u_0 = 0) is washed out by the
    // zero-mean comparison used for the L2 column. The system is already compatible
    // (above), so a solution with u_0 = 0 exists and the pin selects it.
    gsSparseMatrix<> Kpinned;
    gsMatrix<>       Fpinned;
    const gsSparseMatrix<>* Kptr = &A.matrix();
    const gsMatrix<>*       Fptr = &A.rhs();
    if (singular)
    {
        Kpinned = A.matrix();
        Fpinned = A.rhs();
        const index_t j = 0;
        for (index_t k = 0; k < Kpinned.outerSize(); ++k)
            for (gsSparseMatrix<>::InnerIterator it(Kpinned, k); it; ++it)
                if (it.row() == j || it.col() == j)
                    it.valueRef() = (it.row() == j && it.col() == j) ? 1.0 : 0.0;
        Fpinned(j, 0) = 0;
        Kptr = &Kpinned;
        Fptr = &Fpinned;
    }

    const double globalAssembleTime = timer.stop();
    const gsSparseMatrix<>& K = *Kptr;
    const gsMatrix<>&       F = *Fptr;
    gsInfo << "done (" << globalAssembleTime << " s, " << K.rows()
           << " coupled dofs = global system size solved by direct/CG/multigrid).\n";

    /**************** Direct solver (reference) ****************/

    gsInfo << "Solve: direct Cholesky... " << std::flush;
    {
        timer.restart();
        gsSparseSolver<>::SimplicialLDLT solver;
        solver.compute(K);
        gsMatrix<> x = solver.solve(F);
        const double solveTime = timer.stop();

        gsMultiPatch<> sol = reconstructGlobal(A, u, x);
        BenchResult r;
        r.name = "Direct (Cholesky)"; r.iters = -1; r.setupTime = globalAssembleTime;
        r.solveTime = solveTime; r.l2err = l2ErrorVsExact(mp, sol, uExact);
        r.converged = true;
        r.sol = give(sol);
        results.push_back(r);
    }
    gsInfo << "done.\n";

    /**************** Preconditioned benchmark loop ****************/

    struct PrecRecord {
        std::string name;
        gsLinearOperator<>::Ptr op;
        double setupTime;
    };
    std::vector<PrecRecord> preconditioners;

    preconditioners.push_back({"no prec", gsIdentityOp<>::make(K.rows()), globalAssembleTime});
    preconditioners.push_back({"Jacobi", makeJacobiOp(K), globalAssembleTime});
    preconditioners.push_back({"Gauss-Seidel", makeGaussSeidelOp(K), globalAssembleTime});
    preconditioners.push_back({"rev. Gauss-Seidel", makeReverseGaussSeidelOp(K), globalAssembleTime});
    preconditioners.push_back({"symm. Gauss-Seidel", makeSymmetricGaussSeidelOp(K), globalAssembleTime});
    preconditioners.push_back({"Richardson", makeRichardsonOp(K), globalAssembleTime});

    {
        timer.restart();
        gsLinearOperator<>::Ptr iluPrec = makeIncompleteLUOp(K);
        preconditioners.push_back({"ILU", iluPrec, globalAssembleTime + timer.stop()});
    }

    // MG setup logic
    bool basisIsRational = false;
    for (size_t k = 0; k < mb.nBases(); ++k)
        if (mb.basis(k).isRational()) { basisIsRational = true; break; }

    gsInfo << "Setup multigrid (smoother: " << mgSmoother
           << (basisIsRational ? ", NURBS hierarchy built by refinement" : "")
           << ")... " << std::flush;
    {
        timer.restart();
        const gsOptionList mgOpt = cmd.getGroup("MG");

        std::vector< gsSparseMatrix<real_t,RowMajor> > transferMatrices;
        if (basisIsRational)
        {
            const index_t levels     = std::min(mgLevels, refinements) + 1;
            const index_t coarseRefs = refinements - (levels - 1);
            gsMultiBasis<> coarseBasis = makeBenchBasis(mp, degree, coarseRefs, squareDiscr);
            gsGridHierarchy<>::buildByRefinement(give(coarseBasis), bc, mgOpt, levels)
                .moveTransferMatricesTo(transferMatrices);
        }
        else
        {
            gsGridHierarchy<>::buildByCoarsening(gsMultiBasis<>(mb), bc, mgOpt)
                .moveTransferMatricesTo(transferMatrices);
        }

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
        preconditioners.push_back({"multigrid", mg, globalAssembleTime + timer.stop()});
    }
    gsInfo << "done.\n";

    auto runAllPrec = [&](const std::string& solverName, auto solverTag) {
        typedef typename decltype(solverTag)::type SType;
        for (auto const& p : preconditioners) {
            results.push_back(runGlobalSolver<SType>(
                solverName + " + " + p.name, K, F, p.op, solverOpt, mp, A, u, uExact, p.setupTime));
        }
    };

    gsInfo << "Solve: iterative benchmark loop... " << std::flush;
    runAllPrec("CG",          solver_tag<gsConjugateGradient<>>{});
    runAllPrec("MinRes",      solver_tag<gsMinimalResidual<>>{});
    runAllPrec("MinRes-QLP",  solver_tag<gsMinResQLP<>>{});
    runAllPrec("GMRES",       solver_tag<gsGMRes<>>{});
    runAllPrec("BiCGStab",    solver_tag<gsBiCgStab<>>{});
    runAllPrec("Gradient",    solver_tag<gsGradientMethod<>>{});
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
        r.sol = give(sol);
        results.push_back(r);

        // IETI does NOT solve the coupled global system; it solves a CG on the Schur
        // complement of the Lagrange multipliers plus a primal coarse problem and one
        // local solve per patch. Report those sizes so the 'iters' column is read in
        // context (they count CG steps on this much smaller operator).
        gsInfo << "\n  IETI sizes: " << ieti.nLagrangeMultipliers()
               << " Lagrange multipliers (the CG/Schur system size), "
               << ietiMapper.nPrimalDofs() << " primal dofs, " << nPatches
               << " local patch solves. ";
    }
    gsInfo << "done.\n";

    /******************** Print benchmark table ********************/

    // The manufactured exact solution u = sin(x)*cos(y) is a valid reference only
    // when the assembled PDE is the flat-space Poisson equation -Lap(u) = f, for
    // which it is the exact solution. That holds when domainDim == geoDim (planar
    // domains and volume domains), but NOT for:
    //   - a surface embedded in 3D (domainDim < geoDim): the assembler discretizes
    //     the Laplace-Beltrami operator on a curved manifold, whose solution is not
    //     sin(x)*cos(y);
    //   - Neumann data gN = 1.0, inconsistent with that exact solution.
    // In those cases we cannot report an accuracy error. Instead we still give a
    // numbering-independent agreement check: the L2 distance of each method's
    // reconstructed solution to the IETI-DP solution, taken as the reference. This
    // shows that the methods that converge all reach the same discrete solution.
    const bool l2Meaningful = (mp.domainDim() == mp.geoDim()) && !hasNeumann;

    // For the singular (pure-Neumann) case the system was made compatible and a dof
    // was pinned, so the converging methods now reach the same solution up to a
    // constant. We compare modulo that constant (the pin/IETI gauges differ).
    std::string l2Header = "L2 error";
    bool refConverged = true;
    if (!l2Meaningful)
    {
        const gsMultiPatch<>* refSol = 0;
        for (size_t i = 0; i < results.size(); ++i)
            if (results[i].name.compare(0, 4, "IETI") == 0)
            { refSol = &results[i].sol; refConverged = results[i].converged; }

        if (refSol)
        {
            l2Header = singular ? "L2 vs IETI*" : "L2 vs IETI";
            const gsMultiPatch<> ref = singular ? removeConstant(*refSol) : *refSol;
            for (size_t i = 0; i < results.size(); ++i)
            {
                const gsMultiPatch<> cur =
                    singular ? removeConstant(results[i].sol) : results[i].sol;
                results[i].l2err = l2FieldDistance(mp, cur, ref);
            }
        }
    }

    gsInfo << "\n=================================== Benchmark results ===================================\n";
    gsInfo << std::left << std::setw(32) << "Method"
           << std::right << std::setw(11) << "setup [s]"
           << std::setw(11) << "solve [s]"
           << std::setw(8)  << "iters"
           << std::setw(14) << l2Header
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
        gsInfo << std::scientific << std::setprecision(3) << std::setw(14) << r.l2err
               << std::setw(11) << (r.converged ? "yes" : "NO") << "\n";
    }
    gsInfo << "=========================================================================================\n";
    gsInfo << "Note: IETI-DP iterations count CG steps on the Schur complement (a different operator\n"
              "      and dimension), so iteration counts are directly comparable only among the\n"
              "      global-system solvers.\n";
    if (l2Meaningful)
        gsInfo << "      The 'L2 error' column is the distance to the exact solution u = sin(x)*cos(y);\n"
                  "      its agreement across methods confirms they solved the same discretized problem.\n";
    else
    {
        gsInfo << "      The exact solution u = sin(x)*cos(y) is not valid here (curved surface and/or\n"
                  "      Neumann data), so no accuracy error is reported. The '" << l2Header << "' column is the\n"
                  "      L2 distance of each solution to the IETI-DP solution (the reference, hence 0).\n";
        if (!singular)
            gsInfo << "      The converged solvers sharing one common value confirms they reached the\n"
                      "      same discrete solution.\n";
        else
            gsInfo << "      (*) No Dirichlet data anywhere: the global Poisson system is pure-Neumann. It\n"
                      "      has been made well-posed -- the source was made compatible (mean(f) subtracted,\n"
                      "      so integral(f)=0) and one dof was pinned to remove the constant null space. The\n"
                      "      solution is then unique up to a constant, so the column is computed modulo that\n"
                      "      constant; the converged methods agreeing (small values) confirms they reach the\n"
                      "      same solution.\n";
        if (!refConverged)
            gsWarn << "      WARNING: the IETI-DP reference solver did not converge; the comparison\n"
                      "      column may be unreliable.\n";
    }

    return EXIT_SUCCESS;
}
