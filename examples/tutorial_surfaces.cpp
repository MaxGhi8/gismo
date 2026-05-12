/** @file tutorial_surfaces.cpp

    @brief Tutorial on constructing B-spline and NURBS surfaces in 2D and 3D.

    Author(s): Massimiliano Ghiotto, Gemini CLI
*/

#include <iostream>
#include <gismo.h>

using namespace gismo;

int main(int argc, char* argv[])
{
    std::string output("");
    gsCmdLine cmd("Tutorial on B-spline and NURBS surfaces.");
    cmd.addString("o", "output", "Name of the output file prefix.", output);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

    gsInfo << "G+Smo Tutorial: B-spline and NURBS surfaces.\n";

    // ======================================================================
    // 1. Deformed 2D B-spline Surface (Distorted Patch)
    // ======================================================================
    gsInfo << "\n--- 1. Deformed 2D B-spline Surface (Non-Square) ---\n";

    // Using degree 2 to allow for curved boundaries
    gsKnotVector<> kv_u(0, 1, 0, 3); // degree 2, 1 element
    gsKnotVector<> kv_v(0, 1, 0, 3); // degree 2, 1 element
    
    // Basis: 2D tensor product (degree 2 x 2)
    gsTensorBSplineBasis<2> basis2D(kv_u, kv_v);
    
    // Control points for a 3x3 grid (size = 3*3 = 9)
    gsMatrix<> coefs2D(9, 2);
    // We arrange them to create a distorted shape
    coefs2D << 0.0, 0.0,   0.5, -0.1,  1.0, 0.0,  // Bottom row (curved down)
              -0.2, 0.5,   0.5,  0.5,  1.2, 0.5,  // Middle row (expanded)
               0.1, 1.1,   0.5,  1.0,  0.9, 0.9;  // Top row (skewed)

    gsTensorBSpline<2> surface2D(basis2D, coefs2D);
    gsInfo << "Deformed 2D Surface: " << surface2D << "\n";

    if (output != "")
    {
        gsWriteParaview(surface2D, output + "_surf2D_deformed", 100);
        gsMesh<> cnet;
        surface2D.controlNet(cnet);
        gsWriteParaview(cnet, output + "_surf2D_deformed_cnet");
    }

    // ======================================================================
    // 2. 3D B-spline Surface (A wave shape)
    // ======================================================================
    gsInfo << "\n--- 2. 3D B-spline Surface (Wave) ---\n";

    gsKnotVector<> kv_u3(0, 1, 1, 3); // degree 2, 2 elements
    gsKnotVector<> kv_v3(0, 1, 1, 3); // degree 2, 2 elements
    gsTensorBSplineBasis<2> basis3D(kv_u3, kv_v3);

    // Number of control points = (2+1+1) * (2+1+1) = 4 * 4 = 16
    gsMatrix<> coefs3D(basis3D.size(), 3);
    gsMatrix<> anchors = basis3D.anchors(); // Greville abscissae (u, v)

    for (index_t i = 0; i < basis3D.size(); ++i)
    {
        real_t u = anchors(0, i);
        real_t v = anchors(1, i);
        coefs3D(i, 0) = u;
        coefs3D(i, 1) = v;
        // z = sin(pi*u) * sin(pi*v)
        coefs3D(i, 2) = 0.5 * math::sin(EIGEN_PI * u) * math::sin(EIGEN_PI * v);
    }

    gsTensorBSpline<2> surface3D(basis3D, coefs3D);
    gsInfo << "3D Wave Surface: " << surface3D << "\n";

    if (output != "")
    {
        gsWriteParaview(surface3D, output + "_surf3D", 1000);
        gsMesh<> cnet;
        surface3D.controlNet(cnet);
        gsWriteParaview(cnet, output + "_surf3D_cnet");
    }

    // ======================================================================
    // 3. 3D NURBS Surface (Circular Cylinder / Arc)
    // ======================================================================
    gsInfo << "\n--- 3. 3D NURBS Surface (Circular Cylinder Arc) ---\n";
    gsInfo << "NURBS can represent conic sections (like circles) exactly using weights.\n";

    // Knot vector for circular direction (u): degree 2 -> multiplicity 3 at ends
    // To represent a 90-degree arc exactly:
    gsKnotVector<> kv_arc(0, 1, 0, 3); // degree 2, 1 element
    // Knot vector for height direction (v): degree 1
    gsKnotVector<> kv_height(0, 1, 0, 2); // degree 1, 1 element

    // Control points for the 90 degree arc (u-direction)
    // We need 3 points for a quadratic Bezier arc: (1,0), (1,1), (0,1)
    // And we have 2 levels in height (v-direction)
    // Total points: 3 * 2 = 6
    gsMatrix<> coefsNurbs(6, 3);
    // Height v=0
    coefsNurbs(0, 0) = 1; coefsNurbs(0, 1) = 0; coefsNurbs(0, 2) = 0;
    coefsNurbs(1, 0) = 1; coefsNurbs(1, 1) = 1; coefsNurbs(1, 2) = 0;
    coefsNurbs(2, 0) = 0; coefsNurbs(2, 1) = 1; coefsNurbs(2, 2) = 0;
    // Height v=1
    coefsNurbs(3, 0) = 1; coefsNurbs(3, 1) = 0; coefsNurbs(3, 2) = 1;
    coefsNurbs(4, 0) = 1; coefsNurbs(4, 1) = 1; coefsNurbs(4, 2) = 1;
    coefsNurbs(5, 0) = 0; coefsNurbs(5, 1) = 1; coefsNurbs(5, 2) = 1;

    // Weights: The middle point of a quadratic circular arc needs weight cos(45deg) = 1/sqrt(2)
    gsMatrix<> weights(6, 1);
    real_t w = 1.0 / math::sqrt(2.0);
    weights << 1, w, 1, 1, w, 1;

    // Construct NURBS surface
    // gsTensorNurbs<dim, T>
    gsTensorNurbs<2> nurbsSurface(kv_arc, kv_height, coefsNurbs, weights);
    
    gsInfo << "3D NURBS Cylinder Arc: " << nurbsSurface << "\n";
    gsInfo << "Weights used for middle row in u-direction: " << w << "\n";

    if (output != "")
    {
        gsWriteParaview(nurbsSurface, output + "_nurbs3D", 1000, false, true);
    }

    // ======================================================================
    // 4. Scalar Function on 3D NURBS Surface (Colormap)
    // ======================================================================
    gsInfo << "\n--- 4. Scalar Function on 3D NURBS Surface ---\n";
    gsInfo << "We can define a real-valued function on the surface and visualize it as a colormap.\n";

    // We reuse the basis and geometry from the NURBS example
    // A gsField connects a MultiPatch (geometry) with a MultiBasis and coefficients.
    
    // 1. Create a MultiPatch with our NURBS surface
    gsMultiPatch<> mp;
    mp.addPatch(nurbsSurface.clone());

    // 2. The function lives in the same basis as the geometry
    // For scalar functions, we need one coefficient per basis function.
    gsMatrix<> funcCoefs(nurbsSurface.basis().size(), 1);
    
    // Let's define a "hotspot" in the middle of the surface
    gsMatrix<> bAnchors = nurbsSurface.basis().anchors();
    for (index_t i = 0; i < nurbsSurface.basis().size(); ++i)
    {
        real_t u = bAnchors(0, i);
        real_t v = bAnchors(1, i);
        // Distance from center (0.5, 0.5)
        real_t dist = math::sqrt( (u-0.5)*(u-0.5) + (v-0.5)*(v-0.5) );
        funcCoefs(i, 0) = math::exp(-10.0 * dist * dist); // Gaussian bump
    }

    // 3. Create the geometry for the scalar function
    gsTensorNurbs<2> funcGeometry(nurbsSurface.basis(), funcCoefs);

    // 4. Create the field linking the surface geometry and the scalar function
    gsField<> field(nurbsSurface, funcGeometry);

    if (output != "")
    {
        std::string fn = output + "_nurbsField";
        gsInfo << "Writing NURBS surface with colormap to: " << fn << ".pvd\n";
        // gsWriteParaview for a gsField creates a colored visualization
        gsWriteParaview(field, fn, 1000);
    }

    if (output == "")
    {
        gsInfo << "\nDone. No output created, re-run with --output <prefix> to get ParaView files.\n";
    }
    else
    {
        gsInfo << "\nDone. Open the " << output << "*.pvd files in ParaView to see the surfaces!\n";
    }

    return 0;
}
