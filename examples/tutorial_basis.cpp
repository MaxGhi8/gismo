/** @file tutorial_basis.cpp

    @brief Tutorial on constructing and visualizing B-spline bases in 1D.

    Author(s): Massimiliano Ghiotto, Gemini CLI
*/

#include <iostream>
#include <gismo.h>

using namespace gismo;

/**
 * @brief Helper to write all basis functions of a 1D basis to a single VTP file.
 * 
 * We use gsMesh to collect all functions as separate lines in a single geometry.
 */
void writeBasisVTP(const gsBSplineBasis<>& basis, const std::string& filename)
{
    gsMesh<> mesh;
    const index_t npts = 1000;
    
    // Sample the basis functions over the domain
    gsMatrix<> samples = gsPointGrid(basis.support(), npts);
    gsMatrix<> values = basis.eval(samples);

    // For each basis function, add a "line" to the mesh
    for (index_t i = 0; i < basis.size(); ++i)
    {
        index_t firstV = mesh.numVertices();
        for (index_t j = 0; j < samples.cols(); ++j)
        {
            // Create a 2D plot: (x, value, 0)
            mesh.addVertex(samples(0, j), values(i, j), 0.0);
            if (j > 0)
                mesh.addEdge(firstV + j - 1, firstV + j);
        }
    }
    
    // gsWriteParaview for gsMesh creates a single .vtp file
    gsWriteParaview(mesh, filename);
}

/**
 * @brief Helper to write a single 1D spline function to a single VTP file.
 */
void writeSplineVTP(const gsBSpline<>& spline, const std::string& filename)
{
    gsMesh<> mesh;
    const index_t npts = 1000;
    
    gsMatrix<> samples = gsPointGrid(spline.parameterRange(), npts);
    gsMatrix<> values = spline.eval(samples);

    for (index_t j = 0; j < samples.cols(); ++j)
    {
        // For scalar function, values has 1 row
        mesh.addVertex(samples(0, j), values(0, j), 0.0);
        if (j > 0)
            mesh.addEdge(j - 1, j);
    }
    
    gsWriteParaview(mesh, filename);

    // Also export the control net
    gsMesh<> cnet;
    spline.controlNet(cnet);
    gsWriteParaview(cnet, filename + "_cnet");
}

int main(int argc, char* argv[])
{
    std::string output("");
    gsCmdLine cmd("G+Smo Tutorial: Playing with basis functions in 1D.");
    cmd.addString("o", "output", "Name of the output file prefix.", output);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

    gsInfo << "G+Smo Tutorial: Playing with basis functions in 1D.\n";

    // ======================================================================
    // 1. Bases with 6 elements and different degrees
    // ======================================================================

    for (int p = 0; p <= 3; ++p)
    {
        gsInfo << "\n--- Degree " << p << " ---\n";

        // Create a clamped knot vector with 5 interior knots (resulting in 6 elements)
        // range [0, 1], 5 interior knots, multiplicity at endpoints is p+1
        gsKnotVector<> KV(0.0, 1.0, 5, p + 1);

        // Create the B-spline basis
        gsBSplineBasis<> basis(KV);

        gsInfo << "Basis: " << basis << "\n";
        gsInfo << "Number of elements: " << basis.numElements() << "\n";
        gsInfo << "Number of basis functions: " << basis.size() << "\n";

        if (output != "")
        {
            std::string filename = output + "_basis_p" + util::to_string(p);
            gsInfo << "Writing all basis functions to: " << filename << ".vtp\n";
            writeBasisVTP(basis, filename);
        }
    }

    // ======================================================================
    // 2. Effect of knot multiplicity
    // ======================================================================

    gsInfo << "\n--- Effect of Multiplicity (Degree 2) ---\n";

    // Degree 2, 6 elements
    gsKnotVector<> KV_mult(0.0, 1.0, 5, 2 + 1);

    // Let's look at the unique knots
    gsInfo << "Unique knots: ";
    for (auto k : KV_mult.unique()) gsInfo << k << " ";
    gsInfo << "\n";

    // We make the middle knot (0.5) have multiplicity 2.
    // In G+Smo, we can insert a knot value to increase its multiplicity.
    KV_mult.insert(0.5, 1); // Insert 0.5 once more (it was already there once)

    gsBSplineBasis<> basis_mult(KV_mult);
    gsInfo << "Basis with multiplicity 2 at 0.5: \n" << basis_mult << "\n";
    gsInfo << "Functions are C^0 continuous at 0.5 instead of C^1.\n";

    if (output != "")
    {
        std::string filename = output + "_basis_p2_mult2";
        gsInfo << "Writing all basis functions to: " << filename << ".vtp\n";
        writeBasisVTP(basis_mult, filename);
    }


    // ======================================================================
    // 3. Constructing a function
    // ======================================================================

    gsInfo << "\n--- Constructing a Function (Degree 3) ---\n";

    // Use degree 3 basis from before
    gsKnotVector<> KV_f(0.0, 1.0, 5, 3 + 1); // To construct functions
    gsBSplineBasis<> basis_f(KV_f);

    // Create a vector of coefficients
    // For a 1D scalar function, we need a matrix of size (basis.size() x 1)
    gsMatrix<> coefs(basis_f.size(), 1);

    // Create the coefficient
    for (index_t i = 0; i < basis_f.size(); ++i)
    {
        if (i == 4) coefs(i, 0) = 1.0;
        else if (i == 3 || i == 5) coefs(i, 0) = 0.5;
        else coefs(i, 0) = 0.1;
    }

    gsInfo << "Coefficients: \n" << coefs.transpose() << "\n";

    // Create the B-spline function (curve with target dimension 1)
    gsBSpline<> spline(basis_f, coefs);

    if (output != "")
    {
        std::string filename = output + "_my_function";
        gsInfo << "Writing the single function to: " << filename << ".vtp\n";
        writeSplineVTP(spline, filename);
    }

    if (output == "")
    {
        gsInfo << "\nDone. No output created, re-run with --output <prefix> to get ParaView "
                  "files containing the solution.\n";
    }
    else
    {
        gsInfo << "\nDone. Open the " << output << "*.vtp files in ParaView to see the results!\n";
    }

    return 0;
}
