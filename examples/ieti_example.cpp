/** @file ieti_example.cpp

    @brief Provides examples for the ieti solver.

    Here, CG solves the Schur complement formulation. For solving
    the saddle point formulation with MINRES, see ieti2_example.cpp.

    This class uses the expression assembler, for a use of the
    gsPoisson Assembler, see ieti2_example.cpp.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): S. Takacs
*/

#include <ctime>
#include <set>
#include <gismo.h>

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

    gsCmdLine cmd("Solves a PDE with an isogeometric discretization using an isogeometric tearing and interconnecting (IETI) solver.");
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
    cmd.addString("",  "out",                   "Write solution and used options to file", out);
    cmd.addSwitch(     "plot",                  "Plot the result with Paraview", plot);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }


    if ( ! gsFileManager::fileExists(geometry) )
    {
        gsInfo << "Geometry file could not be found.\n";
        gsInfo << "I was searching in the current directory and in: " << gsFileManager::getSearchPaths() << "\n";
        return EXIT_FAILURE;
    }

    gsInfo << "Run ieti_example with options:\n" << cmd << std::endl;

    /******************* Define geometry ********************/

    gsInfo << "Define geometry... " << std::flush;

    //! [Define Geometry]
    gsMultiPatch<>::uPtr mpPtr = gsReadFile<>(geometry);
    //! [Define Geometry]
    if (!mpPtr)
    {
        gsInfo << "No geometry found in file " << geometry << ".\n";
        return EXIT_FAILURE;
    }
    //! [Define Geometry2]
    gsMultiPatch<>& mp = *mpPtr;
    mp.computeTopology();

    for (index_t i=0; i<splitPatches; ++i)
    {
        gsInfo << "split patches uniformly... " << std::flush;
        mp = mp.uniformSplit();
    }
    //! [Define Geometry2]

    if (stretchGeometry!=1)
    {
       gsInfo << "and stretch it... " << std::flush;
       for (size_t i=0; i!=mp.nPatches(); ++i)
           const_cast<gsGeometry<>&>(mp[i]).scale(stretchGeometry,0);
       // Const cast is allowed since the object itself is not const. Stretching the
       // overall domain keeps its topology.
    }

    gsInfo << "done.\n";

    /************** Define boundary conditions **************/

    gsInfo << "Define right-hand-side and boundary conditions... " << std::flush;

    //! [Define Source]
    // Coefficient function
    gsFunctionExpr<> a( coeff, mp.geoDim() );

    // Right-hand-side
    gsFunctionExpr<> f( "2*sin(x)*cos(y)", mp.geoDim() );

    // Dirichlet function
    gsFunctionExpr<> gD( "sin(x)*cos(y)", mp.geoDim() );

    // Neumann
    gsConstantFunction<> gN( 1.0, mp.geoDim() );

    gsBoundaryConditions<> bc;
    //! [Define Source]
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

    //! [Define Basis]
    gsMultiBasis<> mb(mp);
    //! [Define Basis]

    gsInfo << "Setup bases and adjust degree... " << std::flush;

    //! [Set degree and refine]
    for ( size_t i = 0; i < mb.nBases(); ++ i )
        mb[i].setDegreePreservingMultiplicity(degree);

    for ( index_t i = 0; i < refinements; ++i )
        mb.uniformRefine();
    //! [Set degree and refine]

    // Some patches in the input geometry may have more knot spans in one
    // parametric direction than others (e.g. yeti_mp2.xml has patches with
    // 1 and 2 initial spans). After uniform refinement this leads to
    // different numbers of basis functions per patch. We enforce a square discretization:
    // find the global maximum span count across all patches and all directions,
    // then refine every patch in every direction until it matches that value.
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

    // For IGES/CAD geometries (e.g. hummingbird) patches may have different
    // knot structures along shared interfaces and different parametric domains.
    // We normalize all interior knots to [0,1] before taking the union so that
    // the correct parametric correspondence is respected.
    if (mp.domainDim() == 2)
    {
        auto getKV = [&](index_t k, short_t d) -> gsKnotVector<real_t>&
        {
            if (auto* tb = dynamic_cast<gsTensorBSplineBasis<2,real_t>*>(&mb[k]))
                return tb->knots(d);
            if (auto* tn = dynamic_cast<gsTensorNurbsBasis<2,real_t>*>(&mb[k]))
                return tn->knots(d);
            GISMO_ERROR("makeInterfacesConforming: unsupported basis type for patch " << k);
        };

        // Normalize a knot t ∈ (lo,hi) to (0,1), optionally reflecting
        auto toRef = [](real_t t, real_t lo, real_t hi, bool isFlip) -> real_t
        {
            real_t s = (t - lo) / (hi - lo);
            return isFlip ? 1.0 - s : s;
        };
        // De-normalize from reference [0,1] back to [lo,hi], reflecting if needed
        auto fromRef = [](real_t r, real_t lo, real_t hi, bool isFlip) -> real_t
        {
            real_t s = isFlip ? 1.0 - r : r;
            return lo + s * (hi - lo);
        };

        typedef std::pair<index_t,short_t> PD;

        // Build adjacency list: each interface gives an undirected edge (pd0, pd1, orient)
        std::map<PD, std::vector<std::pair<PD,bool>>> adj;
        for (const boundaryInterface& bi : mp.topology().interfaces())
        {
            const index_t p0 = bi.first().patch;
            const index_t p1 = bi.second().patch;
            const short_t d0 = 1 - bi.first().direction();
            const short_t d1 = bi.dirMap(bi.first(), d0);
            const bool orient = bi.dirOrientation(bi.first(), d0);
            adj[{p0,d0}].emplace_back(PD{p1,d1}, orient);
            adj[{p1,d1}].emplace_back(PD{p0,d0}, orient);
        }

        std::set<PD> visited;

        auto processComp = [&](PD root)
        {
            if (visited.count(root)) return;
            std::vector<PD> comp;
            std::map<PD, bool> flipped; // orientation of this node relative to root
            std::queue<PD> bfsq;
            bfsq.push(root);
            visited.insert(root);
            flipped[root] = false;
            comp.push_back(root);

            while (!bfsq.empty())
            {
                PD cur = bfsq.front(); bfsq.pop();
                auto it = adj.find(cur);
                if (it == adj.end()) continue;
                for (auto& [nbr, orient] : it->second)
                {
                    if (visited.count(nbr)) continue;
                    // flipped[nbr] = flipped[cur] XOR (orientation reversed)
                    flipped[nbr] = flipped[cur] ^ !orient;
                    visited.insert(nbr);
                    comp.push_back(nbr);
                    bfsq.push(nbr);
                }
            }

            // Handle orientation-cycle inconsistency (odd number of flips in a cycle):
            // if a visited neighbour contradicts its flip, the union must be symmetric.
            bool needSym = false;
            for (auto& pd : comp)
            {
                auto it = adj.find(pd);
                if (it == adj.end()) continue;
                for (auto& [nbr, orient] : it->second)
                    if (flipped[nbr] != (flipped[pd] ^ !orient))
                    { needSym = true; break; }
                if (needSym) break;
            }

            // Union of all interior knots in the normalized [0,1] reference frame
            std::set<real_t> unionSet;
            for (auto& pd : comp)
            {
                gsKnotVector<real_t>& kv = getKV(pd.first, pd.second);
                real_t lo = kv.first(), hi = kv.last();
                bool isFlip = flipped[pd];
                for (auto it = kv.ubegin(); it != kv.uend(); ++it)
                    if (*it > lo + 1e-14 && *it < hi - 1e-14)
                        unionSet.insert(toRef(*it, lo, hi, isFlip));
            }
            if (needSym)
            {
                std::vector<real_t> extra;
                for (real_t r : unionSet) extra.push_back(1.0 - r);
                for (real_t r : extra) unionSet.insert(r);
            }

            // Insert missing knots into each patch in the component
            for (auto& pd : comp)
            {
                gsKnotVector<real_t>& kv = getKV(pd.first, pd.second);
                real_t lo = kv.first(), hi = kv.last();
                bool isFlip = flipped[pd];

                // Current interior knots in reference frame
                std::set<real_t> current;
                for (auto it = kv.ubegin(); it != kv.uend(); ++it)
                    if (*it > lo + 1e-14 && *it < hi - 1e-14)
                        current.insert(toRef(*it, lo, hi, isFlip));

                for (real_t r : unionSet)
                {
                    auto it = current.lower_bound(r - 1e-10);
                    if (it == current.end() || std::abs(*it - r) >= 1e-10)
                        kv.insert(fromRef(r, lo, hi, isFlip));
                }
            }
        };

        for (auto& [pd, _] : adj) processComp(pd);
        for (index_t k = 0; k < (index_t)mb.nBases(); ++k)
            for (short_t d = 0; d < mp.domainDim(); ++d)
                processComp({k, d});

        // Verify and report remaining non-conforming interfaces
        index_t nBad = 0;
        for (const boundaryInterface& bi : mp.topology().interfaces())
        {
            const index_t p0 = bi.first().patch;
            const index_t p1 = bi.second().patch;
            const short_t d0 = 1 - bi.first().direction();
            const short_t d1 = bi.dirMap(bi.first(), d0);
            const index_t n0 = getKV(p0,d0).uSize() - 2;
            const index_t n1 = getKV(p1,d1).uSize() - 2;
            if (n0 != n1)
            {
                if (nBad < 3)
                    gsWarn << "Non-conforming: patch " << p0 << " dir " << d0
                           << " (" << n0 << ") vs patch " << p1 << " dir " << d1
                           << " (" << n1 << ")\n";
                ++nBad;
            }
        }
        if (nBad)
            gsWarn << nBad << " non-conforming interface(s) remain.\n";
    }

    gsInfo << "done.\n";

    {
        const short_t dim = mb[0].domainDim();
        bool uniform = true;
        for (size_t k = 1; k < mb.nBases() && uniform; ++k)
        {
            if (mb[k].size() != mb[0].size())
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
        gsInfo << ", " << mb[0].size() << " basis functions.\n";
        gsInfo << "  Total (un-glued): " << totalDofs << " dofs.\n";
    }

    /********* Setup assembler and assemble matrix **********/

    gsInfo << "Setup assembler and assemble matrix... " << std::flush;

    const index_t nPatches = mp.nPatches();

    //! [Define Ieti Mapper]
    gsIetiMapper<> ietiMapper;
    //! [Define Ieti Mapper]

    // We start by setting up a global FeSpace that allows us to
    // obtain a dof mapper and the Dirichlet data
    //! [Define global mapper]
    {
        typedef gsExprAssembler<>::space  space;
        gsExprAssembler<> assembler;
        space u = assembler.getSpace(mb);
        bc.setGeoMap(mp);
        u.setup(bc, dirichlet::interpolation, 0);
        ietiMapper.init( mb, u.mapper(), u.fixedPart() );
    }
    //! [Define global mapper]

    // Which primal dofs should we choose?
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

    // We tell the ieti mapper which primal constraints we want; calling
    // more than one such function is possible.
    //! [Define primals]
    if (cornersAsPrimals)
        ietiMapper.cornersAsPrimals();

    if (edgesAsPrimals)
        ietiMapper.interfaceAveragesAsPrimals(mp,1);

    if (facesAsPrimals)
        ietiMapper.interfaceAveragesAsPrimals(mp,2);
    //! [Define primals]

    // Compute the jump matrices
    bool fullyRedundant = true,
         noLagrangeMultipliersForCorners = cornersAsPrimals;
    //! [Define jumps]
    ietiMapper.computeJumpMatrices(fullyRedundant, noLagrangeMultipliersForCorners);
    //! [Define jumps]

    //! [Setup]
    // The ieti system does not have a special treatment for the
    // primal dofs. They are just one more subdomain
    gsIetiSystem<> ieti;
    ieti.reserve(nPatches+1);

    // The scaled Dirichlet preconditioner is independent of the
    // primal dofs.
    gsScaledDirichletPrec<> prec;
    prec.reserve(nPatches);

    // Setup the primal system, which needs to know the number of primal dofs.
    gsPrimalSystem<> primal(ietiMapper.nPrimalDofs());
    if (eliminateCorners)
        primal.setEliminatePointwiseConstraints(true);
    //! [Setup]

    // Per-patch assembled data (filled in parallel, consumed in serial order)
    std::vector<gsSparseMatrix<real_t, RowMajor>> jumpMatrices(nPatches);
    std::vector<gsSparseMatrix<>>                 localMatrices(nPatches);
    std::vector<gsMatrix<>>                       localRhss(nPatches);

    //! [Assemble]
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (index_t k=0; k<nPatches; ++k)
    {
        // We use the local variants of everything
        gsBoundaryConditions<> bc_local;
        bc.getConditionsForPatch(k,bc_local);
        gsMultiPatch<> mp_local = mp[k];
        gsMultiBasis<> mb_local = mb[k];

        // The usual stuff for the expression assembler
        typedef gsExprAssembler<>::geometryMap geometryMap;
        typedef gsExprAssembler<>::variable    variable;
        typedef gsExprAssembler<>::space       space;
        typedef gsExprAssembler<>::solution    solution;

        // We set up the assembler
        gsExprAssembler<> assembler(1,1);

        // Elements used for numerical integration
        assembler.setIntegrationDomain(mb_local.domain());
        gsExprEvaluator<> ev(assembler);

        // Set the geometry map
        geometryMap G = assembler.getMap(mp_local);

        // Set the discretization space
        space u = assembler.getSpace(mb_local);

        // Incorporate Dirichlet BC
        bc_local.setGeoMap(mp_local);
        u.setup(bc_local, dirichlet::interpolation, 0);

        // This function provides a new dof mapper and the Dirichlet data
        // This is necessary since it might happen that a 2d-patch touches the
        // Dirichlet boundary just with a corner or that a 3d-patch touches the
        // Dirichlet boundary with a corner or an edge. These cases are not
        // covered by bc.getConditionsForPatch
        ietiMapper.initFeSpace(u,k);

        // Set the source term
        auto ff = assembler.getCoeff(f, G);

        // Set the coefficient
        auto aa = assembler.getCoeff(a, G);

        // Initialize the system
        assembler.initSystem();

        // Compute the system matrix and right-hand side
        assembler.assemble( aa.val() * igrad(u, G) * igrad(u, G).tr() * meas(G), u * ff * meas(G) );

        // Add contributions from Neumann conditions to right-hand side
        variable g_N = assembler.getBdrFunction();
        assembler.assembleBdr(bc_local.get("Neumann"),  u * g_N.val() * nv(G).norm() );

        // Store assembled data indexed by physical patch k (not thread order)
        jumpMatrices[k]  = ietiMapper.jumpMatrix(k);
        localMatrices[k] = assembler.matrix();
        localRhss[k]     = assembler.rhs();
    }
    //! [Assemble]

    // Add patches to prec/primal/ieti in sequential order k=0..nPatches-1 so
    // that constructSolutionFromLagrangeMultipliers returns results indexed by
    // physical patch (required by constructGlobalSolutionFromLocalSolutions).
    for (index_t k=0; k<nPatches; ++k)
    {
        gsSparseMatrix<real_t, RowMajor>& jumpMatrix  = jumpMatrices[k];
        gsSparseMatrix<>&                 localMatrix = localMatrices[k];
        gsMatrix<>&                       localRhs    = localRhss[k];

        // Add the patch to the scaled Dirichlet preconditioner
        //! [Patch to preconditioner]
        prec.addSubdomain(
            gsScaledDirichletPrec<>::restrictToSkeleton(
                jumpMatrix,
                localMatrix,
                ietiMapper.skeletonDofs(k)
            )
        );
        //! [Patch to preconditioner]

        // This function writes back to jumpMatrix, localMatrix, and localRhs,
        // so it must be called after prec.addSubdomain().
        //! [Patch to primals]
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
            uniqueConstraints,
            uniqueDofIndices,
            jumpMatrix,
            localMatrix,
            localRhs
        );
        //! [Patch to primals]

        // Add the patch to the Ieti system
        //! [Patch to system]
        ieti.addSubdomain(
            jumpMatrix.moveToPtr(),
            makeMatrixOp(localMatrix.moveToPtr()),
            give(localRhs)
        );
        //! [Patch to system]
    //! [End of assembling loop]
    } // end for
    //! [End of assembling loop]

    // Add the primal problem if there are primal constraints
    //! [Primal to system]
    if (ietiMapper.nPrimalDofs()>0)
    {
        // It is not required to provide a local solver to .addSubdomain,
        // since a sparse LU solver would be set up on the fly if required.
        // Here, we make use of the fact that we can use a Cholesky solver
        // because the primal problem is symmetric and positive definite:
        gsLinearOperator<>::Ptr localSolver
            = makeSparseCholeskySolver(primal.localMatrix());

        ieti.addSubdomain(
            primal.jumpMatrix().moveToPtr(),
            makeMatrixOp(primal.localMatrix().moveToPtr()),
            give(primal.localRhs()),
            localSolver
        );
    }
    //! [Primal to system]

    gsInfo << "done. " << ietiMapper.nPrimalDofs() << " primal dofs, "
           << ieti.nLagrangeMultipliers() << " Lagrange multipliers.\n";

    /**************** Setup solver and solve ****************/

    gsInfo << "Solve (CG on Schur complement)... " << std::flush;

    //! [Setup scaling]
    prec.setupMultiplicityScaling();
    //! [Setup scaling]

    //! [Setup rhs]
    gsMatrix<> rhsForSchur = ieti.rhsForSchurComplement();
    //! [Setup rhs]

    //! [Define initial guess]
    gsMatrix<> lambda;
    lambda.setRandom( ieti.nLagrangeMultipliers(), 1 );
    //! [Define initial guess]

    gsMatrix<> errorHistory;

    //! [Solve]
    gsConjugateGradient<> PCG( ieti.schurComplement(), prec.preconditioner() );
    gsStopwatch timer;
    PCG.setOptions( cmd.getGroup("Solver") ).solveDetailed( rhsForSchur, lambda, errorHistory );
    const double solveTime = timer.stop();
    //! [Solve]

    //! [Recover]
    std::vector<gsMatrix<>> uLocal = primal.distributePrimalSolution(
        ieti.constructSolutionFromLagrangeMultipliers(lambda)
    );
    gsMatrix<> uGlobal = ietiMapper.constructGlobalSolutionFromLocalSolutions(uLocal);
    //! [Recover]

    gsInfo << "done (" << solveTime << " s).\n\n";

    /******************** Print end Exit ********************/

    const index_t iter = errorHistory.rows()-1;
    const bool success = errorHistory(iter,0) < tolerance;
    gsInfo << (success ? "Converged" : "NOT converged") << " after " << iter
           << " iterations (final residual: " << errorHistory(iter,0) << ").\n";
    if (errorHistory.rows() <= 10)
        gsInfo << "Residuals: " << errorHistory.transpose() << "\n\n";
    else
        gsInfo << "Residuals: " << errorHistory.topRows(3).transpose()
               << " ... " << errorHistory.bottomRows(3).transpose() << "\n\n";

    if (calcEigenvalues)
        gsInfo << "Estimated condition number: " << PCG.getConditionNumber() << "\n";

    if (!out.empty())
    {
        gsFileData<> fd;
        std::time_t time = std::time(NULL);
        fd.add(cmd);
        fd.add(uGlobal);
        fd.addComment(std::string("ieti_example   Timestamp:")+std::ctime(&time));
        fd.save(out);
        gsInfo << "Write solution to file " << out << "\n";
    }

    if (plot)
    {
        gsMultiPatch<> mpsol;
        for (index_t k=0; k<nPatches; ++k)
            mpsol.addPatch( mb[k].makeGeometry( ietiMapper.incorporateFixedPart(k, uLocal[k]) ) );

        gsInfo << "Write Paraview data to files ieti_geometry.pvd, ieti_source.pvd, ieti_result.pvd\n";
        gsWriteParaview(mp, "ieti_geometry", 1000);
        gsWriteParaview<>( gsField<>(mp, f), "ieti_source", 1000);
        gsWriteParaview<>( gsField<>(mp, mpsol), "ieti_result", 1000);
    }

    if (!plot&&out.empty())
    {
        gsInfo << "Done. No output created, re-run with --plot to get a ParaView "
                  "file containing the solution or --out to write solution to xml file.\n";
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
