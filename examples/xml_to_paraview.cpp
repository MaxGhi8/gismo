/** @file xml_to_paraview.cpp
    @brief Utility to convert G+Smo XML files to ParaView format (.vtp/.pvd)
*/

#include <gismo.h>
#include <gsIO/gsParaviewCollection.h>

using namespace gismo;

int main(int argc, char *argv[])
{
    std::string filename = "";
    int numPoints = 100;

    gsCmdLine cmd("Convert G+Smo XML geometry to ParaView format.");
    cmd.addPlainString("filename", "G+Smo XML file to read", filename);
    cmd.addInt("n", "points", "Number of sampling points per dimension", numPoints);

    try { cmd.getValues(argc,argv); } catch (int rv) { return rv; }

    if (filename.empty())
    {
        gsInfo << "Please provide an input XML file.\n";
        return EXIT_FAILURE;
    }

    gsFileData<> data(filename);
    
    // Determine the base name for output
    size_t nameStartIdx = filename.rfind('/');
    if (nameStartIdx == std::string::npos) nameStartIdx = 0; else nameStartIdx += 1;
    size_t nameEndIdx = filename.rfind('.');
    std::string baseName = filename.substr(nameStartIdx, nameEndIdx - nameStartIdx);

    bool handled = false;

    // Try to read as MultiPatch (common for NURBS)
    if (data.has<gsMultiPatch<>>() )
    {
        gsInfo << "Detected gsMultiPatch in " << filename << "\n";
        typename gsMultiPatch<>::uPtr mp( data.getFirst<gsMultiPatch<>>() );
        if (mp)
        {
            gsWriteParaview(*mp, baseName, numPoints);
            gsInfo << "Exported to " << baseName << ".pvd\n";
            handled = true;
        }
    }
    
    // Try to read as Solid (trimmed surfaces)
    if (!handled && data.has<gsSolid<>>())
    {
        gsInfo << "Detected gsSolid in " << filename << "\n";
        typename gsSolid<>::uPtr sl( data.getFirst<gsSolid<>>() );
        if (sl)
        {
            gsWriteParaviewSolid(*sl, baseName, numPoints);
            gsInfo << "Exported to " << baseName << ".pvd\n";
            handled = true;
        }
    }

    if (!handled)
    {
        gsInfo << "Error: Could not find or read a gsMultiPatch or gsSolid in the provided XML file.\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
