/** @file gsBenchDiscretization.h

    @brief Shared "fair benchmark" discretization helpers.

    Builds a per-patch basis where every patch gets the identical
    ("square") discretization, and makes glued interfaces conforming
    afterwards. Factored out of solver_benchmark_example.cpp so that other
    examples (ieti_nn_example.cpp, ieti_dataset_generation_example.cpp) can
    reproduce the exact same discretized mesh instead of each rolling its
    own variant.

    This file is part of the G+Smo library.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.

    Author(s): M. Ghiotto
*/

#pragma once

#include <gismo.h>

#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace gismo {

// Make a 2d multi-basis interface-conforming.
//
// IGES/CAD imports give patches with different knot structures along shared
// interfaces AND different parametric domains (e.g. [0,1] vs [0,2]). The IETI
// solver and the global dof-mapper require matching numbers of basis functions
// on both sides of every interface (gsTensorBasis::matchWith only checks the
// counts), which the count-equalising "global" discretization does not by itself
// guarantee once the per-patch knot vectors differ. We build a graph whose nodes
// are (patch, direction) pairs joined by interfaces, find connected components by
// BFS (tracking orientation flips), normalise all interior knots to [0,1], take
// their union, and insert the missing ones back into every patch of the component
// (de-normalised to its own domain). This is a no-op for already-conforming XML
// multipatches (every interface union equals each side's own knots).
inline void makeInterfacesConforming(const gsMultiPatch<>& mp, gsMultiBasis<>& mb)
{
    if (mp.domainDim() != 2)
        return;

    auto getKV = [&](index_t k, short_t d) -> gsKnotVector<real_t>&
    {
        if (auto* tb = dynamic_cast<gsTensorBSplineBasis<2,real_t>*>(&mb[k]))
            return tb->knots(d);
        if (auto* tn = dynamic_cast<gsTensorNurbsBasis<2,real_t>*>(&mb[k]))
            return tn->knots(d);
        GISMO_ERROR("makeInterfacesConforming: unsupported basis type for patch " << k);
    };

    auto toRef = [](real_t t, real_t lo, real_t hi, bool isFlip) -> real_t
    {
        real_t s = (t - lo) / (hi - lo);
        return isFlip ? 1.0 - s : s;
    };
    auto fromRef = [](real_t r, real_t lo, real_t hi, bool isFlip) -> real_t
    {
        real_t s = isFlip ? 1.0 - r : r;
        return lo + s * (hi - lo);
    };

    typedef std::pair<index_t,short_t> PD;

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
        std::map<PD, bool> flipped;
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
            for (auto& nb : it->second)
            {
                const PD& nbr = nb.first; const bool orient = nb.second;
                if (visited.count(nbr)) continue;
                flipped[nbr] = flipped[cur] ^ !orient;
                visited.insert(nbr);
                comp.push_back(nbr);
                bfsq.push(nbr);
            }
        }

        // Orientation-cycle inconsistency (odd number of flips in a cycle):
        // if a visited neighbour contradicts its flip, symmetrise the union.
        bool needSym = false;
        for (auto& pd : comp)
        {
            auto it = adj.find(pd);
            if (it == adj.end()) continue;
            for (auto& nb : it->second)
                if (flipped[nb.first] != (flipped[pd] ^ !nb.second))
                { needSym = true; break; }
            if (needSym) break;
        }

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

        for (auto& pd : comp)
        {
            gsKnotVector<real_t>& kv = getKV(pd.first, pd.second);
            real_t lo = kv.first(), hi = kv.last();
            bool isFlip = flipped[pd];

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

    for (auto& kv : adj) processComp(kv.first);
    for (index_t k = 0; k < (index_t)mb.nBases(); ++k)
        for (short_t d = 0; d < mp.domainDim(); ++d)
            processComp({k, d});
}

// Elements (knot spans) of patch k of mb in parameter direction d.
inline index_t elementsInDir(const gsMultiBasis<>& mb, size_t k, short_t d)
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
inline gsMultiBasis<> makeBenchBasis(const gsMultiPatch<>& mp, index_t degree,
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

} // namespace gismo
