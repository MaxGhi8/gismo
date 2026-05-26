/** @file ieti_AF_example.cpp

    @brief All-Floating (Total FETI) IETI example.

    Solves the same Poisson problem as ieti_example.cpp with two methods:
      (1) IETI-DP: Dirichlet BCs are eliminated from local matrices (as in
          ieti_example.cpp).
      (2) IETI-AF: Dirichlet BCs are NOT eliminated; they are imposed weakly
          through additional Lagrange multipliers, so every patch is a
          "floating" subdomain treated uniformly (Total FETI).

    Both solutions are reconstructed as multipatch functions and compared
    coefficient-wise (max norm) and via L2 difference, to verify that they
    coincide.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): based on ieti_example.cpp by S. Takacs
*/

#include <ctime>
#include <gismo.h>

using namespace gismo;

// ------------------------------------------------------------------
// Helper: parse primals string ("c"/"e"/"f") into flags
// ------------------------------------------------------------------
struct PrimalsConfig
{
    bool corners = false;
    bool edges   = false;
    bool faces   = false;
};

static bool parsePrimals(const std::string& primals, PrimalsConfig& cfg)
{
    for (size_t i = 0; i < primals.length(); ++i)
        switch (primals[i])
        {
            case 'c': cfg.corners = true; break;
            case 'e': cfg.edges   = true; break;
            case 'f': cfg.faces   = true; break;
            default:
                gsInfo << "\nUnknown primal constraint type: \"" << primals[i] << "\"\n";
                return false;
        }
    return true;
}

// ------------------------------------------------------------------
// IETI-DP solve (mirrors ieti_example.cpp)
// Returns the per-patch coefficient vectors of the global solution
// (size mb[k].size() per patch, Dirichlet values included).
// ------------------------------------------------------------------
static std::vector< gsMatrix<> > solveIetiDP(
    const gsMultiPatch<>&         mp,
    const gsMultiBasis<>&         mb,
    const gsBoundaryConditions<>& bc,
    const gsFunction<>&           f,
    const PrimalsConfig&          primalsCfg,
    bool                          eliminateCorners,
    const gsOptionList&           solverOpts,
    index_t&                      iterOut,
    real_t&                       finalResOut)
{
    const index_t nPatches = mp.nPatches();

    gsIetiMapper<> ietiMapper;
    {
        typedef gsExprAssembler<>::space space;
        gsExprAssembler<> assembler;
        space u = assembler.getSpace(mb);
        // Note: setGeoMap is a non-const op on bc; safe because we own a non-const view here only
        gsBoundaryConditions<> bcCopy = bc;
        bcCopy.setGeoMap(mp);
        u.setup(bcCopy, dirichlet::interpolation, 0);
        ietiMapper.init(mb, u.mapper(), u.fixedPart());
    }

    if (primalsCfg.corners) ietiMapper.cornersAsPrimals();
    if (primalsCfg.edges)   ietiMapper.interfaceAveragesAsPrimals(mp, 1);
    if (primalsCfg.faces)   ietiMapper.interfaceAveragesAsPrimals(mp, 2);

    const bool fullyRedundant            = true;
    const bool noLagrangeMultsForCorners = primalsCfg.corners;
    ietiMapper.computeJumpMatrices(fullyRedundant, noLagrangeMultsForCorners);

    gsIetiSystem<>          ieti;     ieti.reserve(nPatches + 1);
    gsScaledDirichletPrec<> prec;     prec.reserve(nPatches);
    gsPrimalSystem<>        primal(ietiMapper.nPrimalDofs());
    if (eliminateCorners) primal.setEliminatePointwiseConstraints(true);

    for (index_t k = 0; k < nPatches; ++k)
    {
        gsBoundaryConditions<> bc_local;
        bc.getConditionsForPatch(k, bc_local);
        gsMultiPatch<> mp_local = mp[k];
        gsMultiBasis<> mb_local = mb[k];

        typedef gsExprAssembler<>::geometryMap geometryMap;
        typedef gsExprAssembler<>::variable    variable;
        typedef gsExprAssembler<>::space       space;

        gsExprAssembler<> assembler(1, 1);
        assembler.setIntegrationDomain(mb_local.domain());

        geometryMap G = assembler.getMap(mp_local);
        space       u = assembler.getSpace(mb_local);

        bc_local.setGeoMap(mp_local);
        u.setup(bc_local, dirichlet::interpolation, 0);
        ietiMapper.initFeSpace(u, k);

        auto ff = assembler.getCoeff(f, G);
        assembler.initSystem();
        assembler.assemble( igrad(u, G) * igrad(u, G).tr() * meas(G),
                            u * ff * meas(G) );

        variable g_N = assembler.getBdrFunction();
        assembler.assembleBdr( bc_local.get("Neumann"), u * g_N.val() * nv(G).norm() );

        gsSparseMatrix<real_t, RowMajor> jumpMatrix  = ietiMapper.jumpMatrix(k);
        gsSparseMatrix<>                 localMatrix = assembler.matrix();
        gsMatrix<>                       localRhs    = assembler.rhs();

        prec.addSubdomain(
            gsScaledDirichletPrec<>::restrictToSkeleton(
                jumpMatrix, localMatrix, ietiMapper.skeletonDofs(k)
            )
        );

        primal.handleConstraints(
            ietiMapper.primalConstraints(k),
            ietiMapper.primalDofIndices(k),
            jumpMatrix, localMatrix, localRhs
        );

        ieti.addSubdomain(
            jumpMatrix.moveToPtr(),
            makeMatrixOp(localMatrix.moveToPtr()),
            give(localRhs)
        );
    }

    if (ietiMapper.nPrimalDofs() > 0)
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
    gsMatrix<> lambda;
    lambda.setRandom(ieti.nLagrangeMultipliers(), 1);

    gsMatrix<> errorHistory;
    gsConjugateGradient<> PCG(ieti.schurComplement(), prec.preconditioner());
    PCG.setOptions(solverOpts).solveDetailed(rhsForSchur, lambda, errorHistory);

    iterOut     = errorHistory.rows() - 1;
    finalResOut = errorHistory(iterOut, 0);

    std::vector<gsMatrix<>> uLocal = primal.distributePrimalSolution(
        ieti.constructSolutionFromLagrangeMultipliers(lambda)
    );

    // Assemble per-patch full coefficient vectors (Dirichlet values added back)
    std::vector< gsMatrix<> > patchCoefs(nPatches);
    for (index_t k = 0; k < nPatches; ++k)
        patchCoefs[k] = ietiMapper.incorporateFixedPart(k, uLocal[k]);

    return patchCoefs;
}

// ------------------------------------------------------------------
// IETI-AF (All-Floating / Total FETI) solve.
// Dirichlet BCs are NOT eliminated; they are imposed via additional
// Lagrange multipliers. Returns per-patch full coefficient vectors.
// ------------------------------------------------------------------
static std::vector< gsMatrix<> > solveIetiAF(
    const gsMultiPatch<>&         mp,
    const gsMultiBasis<>&         mb,
    const gsBoundaryConditions<>& bc,
    const gsFunction<>&           f,
    const PrimalsConfig&          primalsCfg,
    bool                          eliminateCorners,
    const gsOptionList&           solverOpts,
    index_t&                      iterOut,
    real_t&                       finalResOut)
{
    const index_t nPatches = mp.nPatches();

    // ----------------------------------------------------------------
    // (A) Reference mapper WITH Dirichlet elimination (only used to
    //     identify which patch-local basis functions are Dirichlet,
    //     and to retrieve their prescribed values via interpolation).
    // ----------------------------------------------------------------
    gsDofMapper mapperDir;
    gsMatrix<>  fixedPartDir;
    {
        typedef gsExprAssembler<>::space space;
        gsExprAssembler<> assembler;
        space u = assembler.getSpace(mb);
        gsBoundaryConditions<> bcCopy = bc;
        bcCopy.setGeoMap(mp);
        u.setup(bcCopy, dirichlet::interpolation, 0);
        mapperDir    = u.mapper();
        fixedPartDir = u.fixedPart();
    }

    // ----------------------------------------------------------------
    // (B) IETI mapper WITHOUT Dirichlet elimination.
    //     We pass an empty boundary-condition object so the dof mapper
    //     keeps every basis function as a free dof.
    // ----------------------------------------------------------------
    gsIetiMapper<> ietiMapper;
    {
        typedef gsExprAssembler<>::space space;
        gsExprAssembler<> assembler;
        space u = assembler.getSpace(mb);
        gsBoundaryConditions<> bcEmpty;     // no conditions => no elimination
        u.setup(bcEmpty, dirichlet::interpolation, 0);
        ietiMapper.init(mb, u.mapper(), u.fixedPart());
    }

    // For AF every patch is floating. We make every corner a primal dof
    // (incl. Dirichlet corners) to ensure every patch has at least one
    // primal constraint that kills its local constant null space.
    //
    // The corresponding coarse primal matrix P = sum Psi_k^T A_k Psi_k
    // inherits the global constants kernel because the primal subdomain
    // does not see the Dirichlet Lagrange multipliers. We will repair
    // this *after* primal.handleConstraints by pinning the primal dofs
    // that correspond to Dirichlet corners to their Dirichlet values
    // (i.e. applying Dirichlet at the coarse level too). For that we
    // need to remember (primal-dof-index -> Dirichlet value).
    std::vector< std::pair<index_t, real_t> > dirichletPrimals; // (primalIdx, value)
    if (primalsCfg.corners)
    {
        const index_t dim = mb.dim();
        std::map< index_t,
                  std::pair< std::vector< std::pair<index_t, gsSparseVector<real_t> > >,
                             std::pair<bool, real_t> > >  groups;  // bool = is Dirichlet, real_t = value
        for (index_t k = 0; k < nPatches; ++k)
        {
            for (boxCorner it = boxCorner::getFirst(dim); it != boxCorner::getEnd(dim); ++it)
            {
                const index_t idx       = mb[k].functionAtCorner(it);
                const index_t globalDir = mapperDir.index(idx, k);
                const index_t globalIdx = ietiMapper.dofMapperGlobal().index(idx, k);
                const index_t localIdx  = ietiMapper.dofMapperLocal(k).index(idx, 0);
                gsSparseVector<real_t> constr(ietiMapper.dofMapperLocal(k).freeSize());
                constr[localIdx] = 1;

                auto& entry = groups[globalIdx];
                entry.first.push_back({k, give(constr)});
                if ( mapperDir.is_boundary_index(globalDir) )
                {
                    const index_t bidx = mapperDir.global_to_bindex(globalDir);
                    entry.second = {true, fixedPartDir(bidx, 0)};
                }
                // else: leave entry.second.first as false (default)
            }
        }
        for (auto& kv : groups)
        {
            const index_t primalIdx = ietiMapper.nPrimalDofs();
            ietiMapper.customPrimalConstraints(kv.second.first);
            if (kv.second.second.first)
                dirichletPrimals.push_back({primalIdx, kv.second.second.second});
        }
    }
    if (primalsCfg.edges)   ietiMapper.interfaceAveragesAsPrimals(mp, 1);
    if (primalsCfg.faces)   ietiMapper.interfaceAveragesAsPrimals(mp, 2);

    // Interface jump matrices (no Dirichlet enforcement yet)
    const bool fullyRedundant            = true;
    const bool noLagrangeMultsForCorners = primalsCfg.corners;
    ietiMapper.computeJumpMatrices(fullyRedundant, noLagrangeMultsForCorners);

    const index_t nInterfaceLM = ietiMapper.nLagrangeMultipliers();

    // ----------------------------------------------------------------
    // (C) Identify Dirichlet dofs per patch (from the reference mapper),
    //     get their target values, and record their position in the
    //     IETI-AF local dof mapper.
    //     For every Dirichlet dof we will create ONE extra Lagrange
    //     multiplier (a row with +1 at that local dof in patch k).
    // ----------------------------------------------------------------
    // dirichletRows[k] : list of (extra-LM-row-index , local-dof-index)
    // dirichletVals    : 1-column matrix of prescribed values, ordered
    //                    by the same extra-LM-row-index
    //
    // For each *unique* global Dirichlet dof we create exactly ONE
    // Lagrange multiplier and place the +1 entry only on the FIRST
    // patch that carries that dof; the constraint is propagated to
    // every other patch through the interface jumps. Duplicating the
    // multiplier per patch would make the augmented system rank-
    // deficient.
    std::vector< std::vector< std::pair<index_t, index_t> > > dirichletRows(nPatches);
    std::vector<real_t> dirichletValsVec;
    {
        index_t extraLM = 0;
        std::set<index_t> seenGlobalDir;
        for (index_t k = 0; k < nPatches; ++k)
        {
            const index_t patchSize = mapperDir.patchSize(k);
            for (index_t i = 0; i < patchSize; ++i)
            {
                const index_t globalDir = mapperDir.index(i, k);
                if ( !mapperDir.is_boundary_index(globalDir) ) continue;
                if ( !seenGlobalDir.insert(globalDir).second ) continue;

                const index_t bidx = mapperDir.global_to_bindex(globalDir);
                const real_t  val  = fixedPartDir(bidx, 0);

                // Local dof index in the IETI-AF mapper (all dofs free)
                const index_t localIdx = ietiMapper.dofMapperLocal(k).index(i, 0);

                dirichletRows[k].push_back({extraLM, localIdx});
                dirichletValsVec.push_back(val);
                ++extraLM;
            }
        }
    }
    const index_t nDirichletLM = static_cast<index_t>(dirichletValsVec.size());
    const index_t nTotalLM     = nInterfaceLM + nDirichletLM;

    // Pack Dirichlet values into a column vector for assembly below
    gsMatrix<> dirichletVals(nDirichletLM, 1);
    for (index_t i = 0; i < nDirichletLM; ++i)
        dirichletVals(i, 0) = dirichletValsVec[i];

    // ----------------------------------------------------------------
    // (D) Per-patch assembly: local stiffness with NO Dirichlet
    //     elimination + augmented jump matrix (interface rows from
    //     ietiMapper + Dirichlet rows from (C)).
    // ----------------------------------------------------------------
    gsIetiSystem<>          ieti;     ieti.reserve(nPatches + 1);
    gsScaledDirichletPrec<> prec;     prec.reserve(nPatches);
    gsPrimalSystem<>        primal(ietiMapper.nPrimalDofs());
    if (eliminateCorners) primal.setEliminatePointwiseConstraints(true);

    for (index_t k = 0; k < nPatches; ++k)
    {
        // Local Neumann conditions only (Dirichlet handled by LMs)
        gsBoundaryConditions<> bc_local_all, bc_local_neu;
        bc.getConditionsForPatch(k, bc_local_all);
        // copy only Neumann
        for (auto it = bc_local_all.neumannBegin();
                  it != bc_local_all.neumannEnd(); ++it)
            bc_local_neu.addCondition(it->ps, it->type(), it->function());

        gsMultiPatch<> mp_local = mp[k];
        gsMultiBasis<> mb_local = mb[k];

        typedef gsExprAssembler<>::geometryMap geometryMap;
        typedef gsExprAssembler<>::variable    variable;
        typedef gsExprAssembler<>::space       space;

        gsExprAssembler<> assembler(1, 1);
        assembler.setIntegrationDomain(mb_local.domain());

        geometryMap G = assembler.getMap(mp_local);
        space       u = assembler.getSpace(mb_local);

        // Empty BC for the local space too: no Dirichlet elimination
        bc_local_neu.setGeoMap(mp_local);
        u.setup(bc_local_neu, dirichlet::interpolation, 0);
        ietiMapper.initFeSpace(u, k);

        auto ff = assembler.getCoeff(f, G);
        assembler.initSystem();
        assembler.assemble( igrad(u, G) * igrad(u, G).tr() * meas(G),
                            u * ff * meas(G) );

        variable g_N = assembler.getBdrFunction();
        assembler.assembleBdr( bc_local_neu.get("Neumann"), u * g_N.val() * nv(G).norm() );

        // The original (interface-only) jump matrix from ietiMapper.
        gsSparseMatrix<real_t, RowMajor> jumpInterface = ietiMapper.jumpMatrix(k);
        const index_t nLocal = jumpInterface.cols();

        gsSparseMatrix<>                 localMatrix = assembler.matrix();
        gsMatrix<>                       localRhs    = assembler.rhs();

        // ---- Build the augmented jump matrix B_k^aug (rows = nTotalLM) ----
        gsSparseEntries<> entries;
        // copy interface entries
        for (index_t row = 0; row < jumpInterface.outerSize(); ++row)
            for (gsSparseMatrix<real_t, RowMajor>::InnerIterator it(jumpInterface, row); it; ++it)
                entries.add(it.row(), it.col(), it.value());
        // add Dirichlet rows: +1 at the local dof; row index = nInterfaceLM + extraIdx
        for (auto& pr : dirichletRows[k])
            entries.add(nInterfaceLM + pr.first, pr.second, (real_t)1);

        gsSparseMatrix<real_t, RowMajor> jumpAug(nTotalLM, nLocal);
        jumpAug.setFrom(entries);

        // ---- Scaled Dirichlet preconditioner: built on the interface
        // sub-block only (Dirichlet rows are handled by a separate
        // identity block of the global preconditioner below).
        prec.addSubdomain(
            gsScaledDirichletPrec<>::restrictToSkeleton(
                jumpInterface, localMatrix, ietiMapper.skeletonDofs(k)
            )
        );

        // ---- Primal handling (uses the augmented jump matrix). ----
        primal.handleConstraints(
            ietiMapper.primalConstraints(k),
            ietiMapper.primalDofIndices(k),
            jumpAug, localMatrix, localRhs
        );

        ieti.addSubdomain(
            jumpAug.moveToPtr(),
            makeMatrixOp(localMatrix.moveToPtr()),
            give(localRhs)
        );
    }

    // Primal subdomain. The primal jump matrix lives in the augmented
    // LM space, so its row count matches nTotalLM automatically because
    // we passed augmented matrices to primal.handleConstraints.
    //
    // Adjustment to the global constraint RHS introduced by pinning
    // Dirichlet primal dofs at the coarse level. We accumulate it here
    // and apply it to rhsForSchur below.
    gsMatrix<> coarseSchurAdj = gsMatrix<>::Zero(nTotalLM, 1);

    if (ietiMapper.nPrimalDofs() > 0)
    {
        // ---- Apply Dirichlet at the coarse level. ----
        // The primal coarse matrix P has the constants kernel because
        // it is a discrete Laplacian on the corner dofs and does not
        // see the Dirichlet LMs. For every primal dof i_d that lies on
        // the Dirichlet boundary we pin it via classical Dirichlet
        // elimination of row/col i_d of P, column i_d of B_P, and the
        // associated rhs adjustment.
        gsSparseMatrix<>                 P    = primal.localMatrix();
        gsMatrix<>                       rhsP = primal.localRhs();
        gsSparseMatrix<real_t, RowMajor> Bp   = primal.jumpMatrix();

        if (!dirichletPrimals.empty())
        {
            std::vector<bool> isDir(P.rows(), false);
            gsMatrix<>        dirVal(P.rows(), 1);    dirVal.setZero();
            for (auto& pr : dirichletPrimals)
            {
                isDir[pr.first]    = true;
                dirVal(pr.first,0) = pr.second;
            }

            // 1) rhsP -= P[:, dir] * dirVal[dir]
            for (index_t j = 0; j < P.outerSize(); ++j)
            {
                if (!isDir[j]) continue;
                for (gsSparseMatrix<>::InnerIterator it(P, j); it; ++it)
                    if (!isDir[it.row()])
                        rhsP(it.row(), 0) -= it.value() * dirVal(j, 0);
            }

            // 2) Build modified P: keep only non-dir rows/cols, put 1 on
            //    diagonal for dir rows.
            gsSparseEntries<> Pse;
            for (index_t j = 0; j < P.outerSize(); ++j)
            {
                if (isDir[j]) { Pse.add(j, j, (real_t)1); continue; }
                for (gsSparseMatrix<>::InnerIterator it(P, j); it; ++it)
                    if (!isDir[it.row()])
                        Pse.add(it.row(), it.col(), it.value());
            }
            P.setZero();  P.setFrom(Pse);  P.makeCompressed();

            // 3) Set Dirichlet rhs values
            for (auto& pr : dirichletPrimals)
                rhsP(pr.first, 0) = pr.second;

            // 4) Record B_P[:, dir] * dirVal contribution to constraint RHS,
            //    then zero those columns of B_P.
            coarseSchurAdj += Bp * dirVal;

            gsSparseEntries<> Bpse;
            for (index_t row = 0; row < Bp.outerSize(); ++row)
                for (gsSparseMatrix<real_t, RowMajor>::InnerIterator it(Bp, row); it; ++it)
                    if (!isDir[it.col()])
                        Bpse.add(it.row(), it.col(), it.value());
            Bp.setZero();  Bp.setFrom(Bpse);
        }

        auto Pptr  = memory::make_shared(new gsSparseMatrix<>(give(P)));
        auto Bpptr = memory::make_shared(new gsSparseMatrix<real_t, RowMajor>(give(Bp)));
        gsLinearOperator<>::Ptr localSolver = makeSparseLUSolver(*Pptr);
        ieti.addSubdomain(
            Bpptr,
            makeMatrixOp(Pptr),
            give(rhsP),
            localSolver
        );
    }

    prec.setupMultiplicityScaling();

    // ---- Build the block-diagonal preconditioner ----
    // P = diag( SDP on interface rows, I on Dirichlet rows ).
    gsLinearOperator<>::Ptr sdPrec = prec.preconditioner();
    gsLinearOperator<>::Ptr fullPrec;
    if (nDirichletLM == 0)
        fullPrec = sdPrec;
    else
    {
        gsBlockOp<>::Ptr bp = gsBlockOp<>::make(2, 2);
        bp->addOperator(0, 0, sdPrec);
        bp->addOperator(1, 1, gsIdentityOp<>::make(nDirichletLM));
        fullPrec = bp;
    }

    // ---- Modify the Schur RHS: subtract d (constraint RHS) ----
    // Original constraint is sum B^aug u = 0; in AF we want sum B^aug u = d
    // where d is zero on interface rows and = Dirichlet values on Dir rows.
    // Schur form:  S lambda = g - d  where g = sum B A^-1 f.
    gsMatrix<> rhsForSchur = ieti.rhsForSchurComplement();
    // Account for the original Dirichlet LM constraint values AND for
    // the contribution of the pinned coarse-level primals (B_P column
    // sweep into the constraint RHS).
    rhsForSchur.bottomRows(nDirichletLM) -= dirichletVals;
    rhsForSchur                          += coarseSchurAdj;

    gsMatrix<> lambda;
    lambda.setZero(ieti.nLagrangeMultipliers(), 1);

    gsMatrix<> errorHistory;
    gsConjugateGradient<> PCG(ieti.schurComplement(), fullPrec);
    PCG.setOptions(solverOpts).solveDetailed(rhsForSchur, lambda, errorHistory);

    iterOut     = errorHistory.rows() - 1;
    finalResOut = errorHistory(iterOut, 0);

    std::vector<gsMatrix<>> uLocal = primal.distributePrimalSolution(
        ieti.constructSolutionFromLagrangeMultipliers(lambda)
    );

    std::vector< gsMatrix<> > patchCoefs(nPatches);
    for (index_t k = 0; k < nPatches; ++k)
        patchCoefs[k] = ietiMapper.incorporateFixedPart(k, uLocal[k]);

    return patchCoefs;
}

// ------------------------------------------------------------------
// Build a gsMultiPatch solution field from per-patch coefficient
// vectors and a multibasis (one geometry per patch).
// ------------------------------------------------------------------
static gsMultiPatch<> makeSolutionMP(const gsMultiBasis<>& mb,
                                     const std::vector< gsMatrix<> >& patchCoefs)
{
    gsMultiPatch<> sol;
    for (size_t k = 0; k < mb.nBases(); ++k)
        sol.addPatch( mb[k].makeGeometry(patchCoefs[k]) );
    return sol;
}

// ==================================================================
//                                MAIN
// ==================================================================
int main(int argc, char *argv[])
{
    // ---------------- command line (same as ieti_example) ----------------
    std::string geometry("domain2d/yeti_mp2.xml");
    index_t splitPatches      = 1;
    real_t  stretchGeometry   = 1;
    index_t refinements       = 1;
    index_t degree            = 2;
    std::string boundaryConditions("d");
    std::string primals("c");
    bool   eliminateCorners   = false;
    real_t tolerance          = 1.e-8;
    index_t maxIterations     = 100;
    bool   plot               = false;

    gsCmdLine cmd("IETI-AF (Total FETI) example. Solves a Poisson problem with "
                  "both IETI-DP and IETI-AF and compares the two solutions.");
    cmd.addString("g", "Geometry",            "Geometry file", geometry);
    cmd.addInt   ("",  "SplitPatches",        "Split every patch that many times in 2^d patches", splitPatches);
    cmd.addReal  ("",  "StretchGeometry",     "Stretch geometry in x-direction by the given factor", stretchGeometry);
    cmd.addInt   ("r", "Refinements",         "Number of uniform h-refinement steps", refinements);
    cmd.addInt   ("p", "Degree",              "Degree of the B-spline discretization space", degree);
    cmd.addString("b", "BoundaryConditions",  "Boundary conditions", boundaryConditions);
    cmd.addString("c", "Primals",             "Primal constraints (c=corners, e=edges, f=faces)", primals);
    cmd.addSwitch("e", "EliminateCorners",    "Eliminate corners (if they are primals)", eliminateCorners);
    cmd.addReal  ("t", "Solver.Tolerance",    "Stopping criterion for linear solver", tolerance);
    cmd.addInt   ("",  "Solver.MaxIterations","Maximum iterations for linear solver", maxIterations);
    cmd.addSwitch(     "plot",                "Plot the AF result with Paraview", plot);
    try { cmd.getValues(argc, argv); } catch (int rv) { return rv; }

    if (!gsFileManager::fileExists(geometry))
    {
        gsInfo << "Geometry file not found.\n";
        return EXIT_FAILURE;
    }

    // ---------------- geometry & refinement ----------------
    gsMultiPatch<>::uPtr mpPtr = gsReadFile<>(geometry);
    if (!mpPtr) { gsInfo << "No geometry in " << geometry << ".\n"; return EXIT_FAILURE; }
    gsMultiPatch<>& mp = *mpPtr;

    for (index_t i = 0; i < splitPatches; ++i) mp = mp.uniformSplit();

    if (stretchGeometry != 1)
        for (size_t i = 0; i != mp.nPatches(); ++i)
            const_cast<gsGeometry<>&>(mp[i]).scale(stretchGeometry, 0);

    // ---------------- boundary conditions ----------------
    gsFunctionExpr<>    f ("2*sin(x)*cos(y)",  mp.geoDim());
    gsFunctionExpr<>    gD("sin(x)*cos(y)",    mp.geoDim());
    gsConstantFunction<> gN(1.0,               mp.geoDim());

    gsBoundaryConditions<> bc;
    {
        const index_t len = boundaryConditions.length();
        index_t i = 0;
        for (auto it = mp.bBegin(); it < mp.bEnd(); ++it, ++i)
        {
            char b_local = (len == 1) ? boundaryConditions[0]
                            : (i < len ? boundaryConditions[i] : '?');
            if      (b_local == 'd') bc.addCondition(*it, condition_type::dirichlet, &gD);
            else if (b_local == 'n') bc.addCondition(*it, condition_type::neumann,   &gN);
            else { gsInfo << "Invalid BC '" << b_local << "'\n"; return EXIT_FAILURE; }
        }
    }

    gsMultiBasis<> mb(mp);
    for (size_t i = 0; i < mb.nBases(); ++i) mb[i].setDegreePreservingMultiplicity(degree);
    for (index_t i = 0; i < refinements; ++i) mb.uniformRefine();

    // Enforce square discretization: find the global maximum knot span count
    // across all patches and all directions, then refine every patch in every
    // direction until it matches, so all patches have the same number of BFs.
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

    // ---------------- primals config ----------------
    PrimalsConfig primalsCfg;
    if (!parsePrimals(primals, primalsCfg)) return EXIT_FAILURE;

    gsOptionList solverOpts = cmd.getGroup("Solver");

    gsStopwatch timer;

    gsInfo << "\n*** Running IETI-DP ***\n";
    index_t iterDP; real_t resDP;
    timer.restart();
    std::vector< gsMatrix<> > coefDP =
        solveIetiDP(mp, mb, bc, f, primalsCfg, eliminateCorners, solverOpts, iterDP, resDP);
    const double timeDP = timer.stop();
    gsInfo << "IETI-DP: " << iterDP << " iterations, final residual " << resDP
           << ", solve time: " << timeDP << " s\n";

    gsInfo << "\n*** Running IETI-AF (Total FETI) ***\n";
    index_t iterAF; real_t resAF;
    timer.restart();
    std::vector< gsMatrix<> > coefAF =
        solveIetiAF(mp, mb, bc, f, primalsCfg, eliminateCorners, solverOpts, iterAF, resAF);
    const double timeAF = timer.stop();
    gsInfo << "IETI-AF: " << iterAF << " iterations, final residual " << resAF
           << ", solve time: " << timeAF << " s\n";

    // ---------------- compare ----------------
    GISMO_ASSERT(coefDP.size() == coefAF.size(), "patch count mismatch");

    real_t maxAbsDiff = 0;
    real_t maxRefAbs  = 0;
    for (size_t k = 0; k < coefDP.size(); ++k)
    {
        GISMO_ASSERT(coefDP[k].rows() == coefAF[k].rows()
                     && coefDP[k].cols() == coefAF[k].cols(),
                     "coef size mismatch on patch " << k);
        maxAbsDiff = std::max(maxAbsDiff, (coefDP[k] - coefAF[k]).cwiseAbs().maxCoeff());
        maxRefAbs  = std::max(maxRefAbs , coefDP[k].cwiseAbs().maxCoeff());
    }
    const real_t relErr = maxRefAbs > 0 ? maxAbsDiff / maxRefAbs : maxAbsDiff;

    gsInfo << "\n*** Comparison ***\n";
    gsInfo << "max |coef_DP - coef_AF| over all patches : " << maxAbsDiff << "\n";
    gsInfo << "max |coef_DP|                            : " << maxRefAbs  << "\n";
    gsInfo << "relative max-coefficient difference      : " << relErr     << "\n";

    // L2 difference of the two reconstructed multipatch solutions
    gsMultiPatch<> solDP = makeSolutionMP(mb, coefDP);
    gsMultiPatch<> solAF = makeSolutionMP(mb, coefAF);

    real_t sqDiff = 0, sqRef = 0;
    {
        gsExprAssembler<> ev_as(1, 1);
        ev_as.setIntegrationElements(mb);
        gsExprEvaluator<> ev(ev_as);
        auto G  = ev.getMap(mp);
        auto uA = ev.getVariable(solDP);
        auto uB = ev.getVariable(solAF);
        sqDiff = ev.integral( (uA - uB) * (uA - uB) * meas(G) );
        sqRef  = ev.integral( uA * uA * meas(G) );
    }
    const real_t l2Diff = math::sqrt(std::max<real_t>(sqDiff, 0));
    const real_t l2Ref  = math::sqrt(std::max<real_t>(sqRef , 0));
    gsInfo << "||u_DP - u_AF||_L2  : " << l2Diff << "\n";
    gsInfo << "||u_DP||_L2         : " << l2Ref  << "\n";
    gsInfo << "relative L2 diff    : " << (l2Ref > 0 ? l2Diff / l2Ref : l2Diff) << "\n";

    if (plot)
    {
        gsInfo << "Writing Paraview output: ieti_AF_result_DP.pvd, ieti_AF_result_AF.pvd\n";
        gsWriteParaview<>(gsField<>(mp, solDP), "ieti_AF_result_DP", 1000);
        gsWriteParaview<>(gsField<>(mp, solAF), "ieti_AF_result_AF", 1000);
    }

    // success criterion: AF matches DP up to solver tolerance
    const real_t okTol = std::max<real_t>(1.e-6, 100 * tolerance);
    const bool   ok    = (l2Ref > 0 ? l2Diff / l2Ref : l2Diff) < okTol;
    if (ok) gsInfo << "\nOK: IETI-AF matches IETI-DP within " << okTol << "\n";
    else    gsInfo << "\nFAIL: IETI-AF differs from IETI-DP above " << okTol << "\n";

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
