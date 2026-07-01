/** @file ieti_dataset_generation_example.cpp

    @brief Generates a training dataset for the neural Schur operator.

    For each of the yeti domain's 21 fixed patches, this computes the exact
    local Dirichlet Schur complement S_k (via gsScaledDirichletPrec, reusing
    gsIetiMapper only to find each patch's skeleton dofs -- no global IETI
    solve, no primal system), draws Gaussian-random-field probe vectors v on
    the skeleton, and writes (geometry, v, S_k*v) triples to a single CSV.

    Author(s): M. Ghiotto
*/

#include <random>
#include <fstream>
#include <iomanip>
#include <limits>
#include <gismo.h>

using namespace gismo;

// Dense Schur complement of A onto the given skeleton dofs, computed
// independently of gsScaledDirichletPrec (dense inverse instead of sparse
// Cholesky), for use as a cross-check.
gsMatrix<> denseSchurComplement(const gsSparseMatrix<> & A, const std::vector<index_t> & skel)
{
    const index_t nFree = A.rows();
    std::vector<bool> isSkeleton(nFree, false);
    for (index_t idx : skel) isSkeleton[idx] = true;

    std::vector<index_t> interior;
    for (index_t i = 0; i < nFree; ++i)
        if (!isSkeleton[i]) interior.push_back(i);

    gsMatrix<> dense(A);
    const index_t ns = skel.size();
    const index_t ni = interior.size();

    gsMatrix<> A00(ns,ns), A01(ns,ni), A10(ni,ns), A11(ni,ni);
    for (index_t a=0; a<ns; ++a)
        for (index_t b=0; b<ns; ++b)
            A00(a,b) = dense(skel[a], skel[b]);
    for (index_t a=0; a<ns; ++a)
        for (index_t b=0; b<ni; ++b)
            A01(a,b) = dense(skel[a], interior[b]);
    for (index_t a=0; a<ni; ++a)
        for (index_t b=0; b<ns; ++b)
            A10(a,b) = dense(interior[a], skel[b]);
    for (index_t a=0; a<ni; ++a)
        for (index_t b=0; b<ni; ++b)
            A11(a,b) = dense(interior[a], interior[b]);

    return A00 - A01 * A11.inverse() * A10;
}

int main(int argc, char *argv[])
{
    /************** Define command line options *************/

    std::string geometry("domain2d/yeti_mp2.xml");
    index_t refinements = 2;
    index_t degree = 2;
    std::string boundaryConditions("d");
    index_t numSamples = 20000;
    index_t seed = 0;
    std::string out("yeti_dataset.csv");
    bool plot = false;
    index_t plotPatch = 0;
    index_t plotSample = 0;

    gsCmdLine cmd("Generates a training dataset for the neural Schur operator "
                  "from the yeti domain's local Dirichlet Schur complements.");
    cmd.addString("g", "Geometry",           "Geometry file", geometry);
    cmd.addInt   ("r", "Refinements",        "Number of uniform h-refinement steps", refinements);
    cmd.addInt   ("p", "Degree",             "Degree of the B-spline discretization space", degree);
    cmd.addString("b", "BoundaryConditions", "Boundary conditions (only 'd' supported)", boundaryConditions);
    cmd.addInt   ("n", "NumSamples",         "Total number of probe/target samples across all patches", numSamples);
    cmd.addInt   ("",  "seed",               "RNG seed", seed);
    cmd.addString("o", "out",                "Output CSV path", out);
    cmd.addSwitch(     "plot",               "Write one (patch, sample) probe/target pair to ParaView", plot);
    cmd.addInt   ("",  "plotPatch",          "Patch index to plot (requires --plot)", plotPatch);
    cmd.addInt   ("",  "plotSample",         "Sample index within that patch to plot (requires --plot)", plotSample);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

    if ( ! gsFileManager::fileExists(geometry) )
    {
        gsInfo << "Geometry file could not be found.\n";
        gsInfo << "I was searching in the current directory and in: " << gsFileManager::getSearchPaths() << "\n";
        return EXIT_FAILURE;
    }

    gsInfo << "Run ieti_dataset_generation_example with options:\n" << cmd << std::endl;

    /******************* Define geometry ********************/

    gsMultiPatch<>::uPtr mpPtr = gsReadFile<>(geometry);
    if (!mpPtr)
    {
        gsInfo << "No geometry found in file " << geometry << ".\n";
        return EXIT_FAILURE;
    }
    gsMultiPatch<>& mp = *mpPtr;

    gsMultiBasis<> mb(mp);
    for ( size_t i = 0; i < mb.nBases(); ++ i )
        mb[i].setDegreePreservingMultiplicity(degree);
    for ( index_t i = 0; i < refinements; ++i )
        mb.uniformRefine();

    /************** Define boundary conditions **************/

    gsFunctionExpr<> gD("0", mp.geoDim());
    gsBoundaryConditions<> bc;
    {
        const index_t len = boundaryConditions.length();
        index_t i = 0;
        for (gsMultiPatch<>::const_biterator it = mp.bBegin(); it < mp.bEnd(); ++it)
        {
            char b_local = (len == 1) ? boundaryConditions[0] : boundaryConditions[i];
            if (b_local == 'd')
                bc.addCondition(*it, condition_type::dirichlet, &gD);
            else
            {
                gsInfo << "\nOnly Dirichlet ('d') boundary conditions are supported by this example.\n";
                return EXIT_FAILURE;
            }
            ++i;
        }
        gsInfo << "done. " << i << " boundary conditions set.\n";
    }
    bc.setGeoMap(mp);

    /*************** Setup gsIetiMapper **********************/

    gsIetiMapper<> ietiMapper;
    {
        typedef gsExprAssembler<>::space space;
        gsExprAssembler<> assembler;
        space u = assembler.getSpace(mb);
        u.setup(bc, dirichlet::interpolation, 0);
        ietiMapper.init(mb, u.mapper(), u.fixedPart());
    }
    ietiMapper.computeJumpMatrices(true, false);

    const index_t nPatches = mp.nPatches();
    gsInfo << "Yeti domain: " << nPatches << " patches.\n";

    /********* Per-patch local matrix + geometry + skeleton *********/

    struct PatchData
    {
        gsMatrix<>            zeta;          // (n_local x 4): x,y,z,w -- free dofs only
        std::vector<index_t>  skeletonDofs;  // indices into zeta's rows
        gsSparseMatrix<>      localMatrix;   // (n_local x n_local)
    };

    std::vector<PatchData> patches(nPatches);

    for (index_t k = 0; k < nPatches; ++k)
    {
        gsBoundaryConditions<> bc_local;
        bc.getConditionsForPatch(k, bc_local);
        gsMultiPatch<> mp_local = mp[k];
        gsMultiBasis<> mb_local = mb[k];
        bc_local.setGeoMap(mp_local);

        typedef gsExprAssembler<>::geometryMap geometryMap;
        typedef gsExprAssembler<>::space       space;

        gsExprAssembler<> assembler(1,1);
        assembler.setIntegrationDomain(mb_local.domain());
        geometryMap G = assembler.getMap(mp_local);
        space u = assembler.getSpace(mb_local);
        u.setup(bc_local, dirichlet::interpolation, 0);
        ietiMapper.initFeSpace(u, k);

        assembler.initSystem();
        assembler.assemble( igrad(u, G) * igrad(u, G).tr() * meas(G) );

        PatchData & pd = patches[k];
        pd.localMatrix = assembler.matrix();

        const gsDofMapper & dm = ietiMapper.dofMapperLocal(k);
        const index_t nFree = dm.freeSize();
        GISMO_ASSERT(nFree == pd.localMatrix.rows(),
            "Free-dof count does not match assembled local matrix size.");

        gsMatrix<> anchors = mb[k].anchors();
        gsMatrix<> phys;
        mp.patch(k).eval_into(anchors, phys);

        pd.zeta.setZero(nFree, 4);
        for (index_t j = 0; j < mb[k].size(); ++j)
        {
            if (dm.is_free(j,0))
            {
                const index_t i = dm.index(j,0);
                pd.zeta(i,0) = phys(0,j);
                pd.zeta(i,1) = (phys.rows() > 1) ? phys(1,j) : 0.0;
                pd.zeta(i,2) = (phys.rows() > 2) ? phys(2,j) : 0.0;
                pd.zeta(i,3) = 1.0; // NURBS weight; yeti is a plain B-spline domain
            }
        }

        pd.skeletonDofs = ietiMapper.skeletonDofs(k);

        gsInfo << "Patch " << k << ": n_basis=" << mb[k].size()
               << ", n_local(free)=" << nFree
               << ", n_skeleton=" << pd.skeletonDofs.size() << "\n";
    }

    /*************** Schur complement operators ******************/

    std::vector<gsLinearOperator<>::Ptr> schurOps(nPatches);
    for (index_t k = 0; k < nPatches; ++k)
        schurOps[k] = gsScaledDirichletPrec<>::schurComplement(
            patches[k].localMatrix, patches[k].skeletonDofs);

    // Cross-check patch 0's operator against a hand-rolled dense Schur
    // complement, as a one-time correctness diagnostic (not part of the
    // main data-generation path).
    {
        const index_t k = 0;
        gsMatrix<> denseSchur = denseSchurComplement(patches[k].localMatrix, patches[k].skeletonDofs);

        gsMatrix<> testVec, testOut;
        testVec.setRandom(static_cast<index_t>(patches[k].skeletonDofs.size()), 1);
        schurOps[k]->apply(testVec, testOut);

        gsMatrix<> refOut = denseSchur * testVec;
        const real_t maxDiff = (testOut - refOut).array().abs().maxCoeff();
        gsInfo << "Verification (patch 0): max |S_builtin*v - S_dense*v| = " << maxDiff << "\n";
    }

    /************ Gaussian-random-field probe sampling ***********/

    struct SampleData
    {
        gsMatrix<> dirichlet; // (n_skeleton x n_samples_k): probe vectors v
        gsMatrix<> output;    // (n_skeleton x n_samples_k): targets S_k * v
    };

    std::vector<SampleData> samples(nPatches);

    std::mt19937_64 rng(static_cast<std::uint64_t>(seed));
    std::normal_distribution<real_t> normal(0.0, 1.0);

    const real_t sigma   = 1.0;
    const real_t d_param = 0.05;
    const real_t lambda  = 2.0 * d_param * d_param;

    const index_t base = numSamples / nPatches;
    const index_t rem  = numSamples % nPatches;

    for (index_t k = 0; k < nPatches; ++k)
    {
        const std::vector<index_t> & skel = patches[k].skeletonDofs;
        const index_t ns = skel.size();
        const index_t nSamplesK = base + (k < rem ? 1 : 0);

        gsMatrix<> cov(ns, ns);
        for (index_t a = 0; a < ns; ++a)
        {
            const real_t xa = patches[k].zeta(skel[a],0);
            const real_t ya = patches[k].zeta(skel[a],1);
            const real_t za = patches[k].zeta(skel[a],2);
            for (index_t b = 0; b < ns; ++b)
            {
                const real_t xb = patches[k].zeta(skel[b],0);
                const real_t yb = patches[k].zeta(skel[b],1);
                const real_t zb = patches[k].zeta(skel[b],2);
                const real_t dist = std::sqrt( (xa-xb)*(xa-xb) + (ya-yb)*(ya-yb) + (za-zb)*(za-zb) );
                cov(a,b) = sigma*sigma*std::exp(-dist/lambda);
            }
        }

        auto llt = cov.llt();
        GISMO_ENSURE(llt.info() == gsEigen::Success,
            "GRF covariance matrix is not positive definite for patch " << k);
        gsMatrix<> L = llt.matrixL();

        SampleData & sd = samples[k];
        sd.dirichlet.resize(ns, nSamplesK);
        sd.output.resize(ns, nSamplesK);

        for (index_t s = 0; s < nSamplesK; ++s)
        {
            gsMatrix<> z(ns,1);
            for (index_t i = 0; i < ns; ++i)
                z(i,0) = normal(rng);
            gsMatrix<> v = L * z;

            gsMatrix<> y;
            schurOps[k]->apply(v, y);

            sd.dirichlet.col(s) = v;
            sd.output.col(s)    = y;
        }

        gsInfo << "Patch " << k << ": generated " << nSamplesK << " samples.\n";
    }

    /********************** Write output CSV **********************/

    index_t Lmax = 0, Mmax = 0;
    for (index_t k = 0; k < nPatches; ++k)
    {
        Lmax = std::max(Lmax, patches[k].zeta.rows());
        Mmax = std::max(Mmax, static_cast<index_t>(patches[k].skeletonDofs.size()));
    }

    {
        const std::string outDir = gsFileManager::getPath(out);
        if (!outDir.empty())
            gsFileManager::mkdir(outDir);
    }

    std::ofstream file(out.c_str());
    GISMO_ENSURE(file.is_open(), "Could not open output file: " << out);
    file << std::setprecision(std::numeric_limits<real_t>::max_digits10);

    file << "patch_id,n_local,n_skeleton";
    for (index_t i = 0; i < Lmax; ++i)
        file << ",geom_" << i << "_x,geom_" << i << "_y,geom_" << i << "_z,geom_" << i << "_w";
    for (index_t i = 0; i < Mmax; ++i)
        file << ",dirichlet_" << i;
    for (index_t i = 0; i < Mmax; ++i)
        file << ",output_" << i;
    file << "\n";

    index_t rowsWritten = 0;
    for (index_t k = 0; k < nPatches; ++k)
    {
        const index_t nLocal    = patches[k].zeta.rows();
        const index_t nSkeleton = static_cast<index_t>(patches[k].skeletonDofs.size());
        const index_t nSamplesK = samples[k].dirichlet.cols();

        for (index_t s = 0; s < nSamplesK; ++s)
        {
            file << k << "," << nLocal << "," << nSkeleton;

            for (index_t i = 0; i < Lmax; ++i)
            {
                if (i < nLocal)
                    file << "," << patches[k].zeta(i,0) << "," << patches[k].zeta(i,1)
                         << "," << patches[k].zeta(i,2) << "," << patches[k].zeta(i,3);
                else
                    file << ",0,0,0,0";
            }

            for (index_t i = 0; i < Mmax; ++i)
                file << "," << (i < nSkeleton ? samples[k].dirichlet(i,s) : real_t(0));

            for (index_t i = 0; i < Mmax; ++i)
                file << "," << (i < nSkeleton ? samples[k].output(i,s) : real_t(0));

            file << "\n";
            ++rowsWritten;
        }
    }
    file.close();

    gsInfo << "Wrote " << rowsWritten << " samples to " << out << "\n";

    /********************* Optional ParaView plot ********************/

    if (plot)
    {
        GISMO_ENSURE(plotPatch >= 0 && plotPatch < nPatches,
            "--plotPatch must be in [0, " << nPatches-1 << "], got " << plotPatch);
        GISMO_ENSURE(plotSample >= 0 && plotSample < samples[plotPatch].dirichlet.cols(),
            "--plotSample must be in [0, " << samples[plotPatch].dirichlet.cols()-1
            << "] for patch " << plotPatch << ", got " << plotSample);

        // Report which of this patch's 4 sides are on the domain's outer
        // Dirichlet boundary (eliminated entirely, so both the input and
        // output fields are exactly zero there) versus an interface shared
        // with a neighboring patch (the only sides that carry skeleton
        // dofs, i.e. nonzero probe/target data).
        {
            static const char * sideName[4] = {"west", "east", "south", "north"};
            gsInfo << "Patch " << plotPatch << " side classification:\n";
            for (short_t s = 1; s <= 4; ++s)
            {
                const bool isDirichlet = mp.isBoundary(plotPatch, boxSide(s));
                gsInfo << "  " << sideName[s-1] << ": "
                       << (isDirichlet ? "outer Dirichlet boundary (no probe/target data)"
                                       : "interface with a neighbor (skeleton dofs live here)")
                       << "\n";
            }
        }

        // Independent check that the specific (v, y) pair being plotted
        // really satisfies y = S_k*v: recompute S_k densely (inverting the
        // interior block directly) and compare against the stored sample,
        // rather than trusting the same schurOps[k] that produced it.
        {
            gsMatrix<> denseSchur = denseSchurComplement(
                patches[plotPatch].localMatrix, patches[plotPatch].skeletonDofs);
            gsMatrix<> v = samples[plotPatch].dirichlet.col(plotSample);
            gsMatrix<> y = samples[plotPatch].output.col(plotSample);
            gsMatrix<> yCheck = denseSchur * v;
            const real_t maxDiff = (y - yCheck).array().abs().maxCoeff();
            const real_t normV = v.norm();
            const real_t normY = y.norm();
            const real_t cosine = (v.transpose() * y).value() / (normV * normY);
            gsInfo << "Plotted sample (patch " << plotPatch << ", sample " << plotSample << "): "
                   << "max |y - S_dense*v| = " << maxDiff << "\n"
                   << "  ||v|| = " << normV << ", ||y|| = " << normY
                   << ", cos(v,y) = " << cosine << "\n";
        }

        const std::vector<index_t> & skel = patches[plotPatch].skeletonDofs;
        const index_t nFree = patches[plotPatch].localMatrix.rows();

        gsMatrix<> freeDirichlet = gsMatrix<>::Zero(nFree, 1);
        gsMatrix<> freeOutput    = gsMatrix<>::Zero(nFree, 1);
        for (size_t i = 0; i < skel.size(); ++i)
        {
            freeDirichlet(skel[i], 0) = samples[plotPatch].dirichlet(i, plotSample);
            freeOutput(skel[i], 0)    = samples[plotPatch].output(i, plotSample);
        }

        // Expand from the free-dof numbering back to the full local basis
        // (filling the Dirichlet-eliminated dofs with their fixed values,
        // here all zero since gD = 0), so the vectors can be turned into
        // spline functions over the whole patch for plotting.
        gsMatrix<> fullDirichlet = ietiMapper.incorporateFixedPart(plotPatch, freeDirichlet);
        gsMatrix<> fullOutput    = ietiMapper.incorporateFixedPart(plotPatch, freeOutput);

        gsMultiPatch<> geomPlot = mp[plotPatch];

        gsMultiPatch<> dirichletPlot;
        dirichletPlot.addPatch(mb[plotPatch].makeGeometry(fullDirichlet));
        gsWriteParaview<>(gsField<>(geomPlot, dirichletPlot), "ieti_dataset_example_input", 1000);

        gsMultiPatch<> outputPlot;
        outputPlot.addPatch(mb[plotPatch].makeGeometry(fullOutput));
        gsWriteParaview<>(gsField<>(geomPlot, outputPlot), "ieti_dataset_example_output", 1000);

        gsInfo << "Wrote ieti_dataset_example_input.pvd / ieti_dataset_example_output.pvd "
               << "(patch " << plotPatch << ", sample " << plotSample << ")\n";
    }

    return EXIT_SUCCESS;
}
