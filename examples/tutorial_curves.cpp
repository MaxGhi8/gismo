/** @file tutorial_curves.cpp

    @brief Tutorial on constructing 2D and 3D B-spline curves.

    Author(s): Massimiliano Ghiotto, Gemini CLI
*/

#include <iostream>
#include <gismo.h>

using namespace gismo;

int main(int argc, char* argv[])
{
    std::string output("");
    gsCmdLine cmd("Tutorial on constructing 2D and 3D B-spline curves.");
    cmd.addString("o", "output", "Name of the output file prefix.", output);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

    gsInfo << "G+Smo Tutorial: Constructing 2D and 3D curves.\n";

    // Common Basis: Degree 3, 4 elements
    gsKnotVector<> kv(0.0, 1.0, 3, 3 + 1);
    gsBSplineBasis<> basis(kv);

    gsInfo << "Basis size: " << basis.size() << "\n";

    // ======================================================================
    // 1. 2D Curve
    // ======================================================================
    gsInfo << "\n--- Creating a 2D Curve ---\n";

    // Coefficients matrix: (number of basis functions) x (physical dimension)
    gsMatrix<> coefs2D(basis.size(), 2);
    coefs2D << 0, 0,
               1, 1,
               2, 1,
               3, 0,
               2, -1,
               1, -1,
               0, 0;

    gsBSpline<> curve2D(basis, coefs2D);
    gsInfo << "2D Curve: " << curve2D << "\n";

    if (output != "")
    {
        std::string fn = output + "_curve2D";
        gsInfo << "Writing 2D curve to: " << fn << ".vtp\n";
        // Note: For 1D objects like curves, G+Smo produces .vtp (PolyData)
        gsWriteParaview(curve2D, fn, 1000);
        
        // Export the control net
        gsMesh<> cnet;
        curve2D.controlNet(cnet);
        gsWriteParaview(cnet, fn + "_cnet");
    }

    // ======================================================================
    // 2. 3D Curve
    // ======================================================================
    gsInfo << "\n--- Creating a 3D Curve ---\n";

    gsMatrix<> coefs3D(basis.size(), 3);
    // Define a helix (spiral) shape in 3D
    coefs3D << 1, 0, 0,
               1, 1, 1,
               0, 1, 2,
               -1, 1, 3,
               -1, -1, 4,
               0, -1, 5,
               1, -1, 6;

    gsBSpline<> curve3D(basis, coefs3D);
    gsInfo << "3D Curve: " << curve3D << "\n";

    if (output != "")
    {
        std::string fn = output + "_curve3D";
        gsInfo << "Writing 3D curve to: " << fn << ".vtp\n";
        gsWriteParaview(curve3D, fn, 1000);
        
        // Export the control net
        gsMesh<> cnet;
        curve3D.controlNet(cnet);
        gsWriteParaview(cnet, fn + "_cnet");
    }

    if (output == "")
    {
        gsInfo << "\nDone. No output created, re-run with --output <prefix> to get ParaView files.\n";
    }
    else
    {
        gsInfo << "\nDone. Open the " << output << "*.vtp files in ParaView to see the curves!\n";
    }

    return 0;
}
