#include <gismo.h>
#include <gsMesh2/gsSurfMesh.h>
#include <gsMesh2/IO.h>

using namespace gismo;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        gsInfo << "Usage: " << argv[0] << " input.obj output.xml\n";
        return 1;
    }

    std::string input = argv[1];
    std::string output = argv[2];

    gsSurfMesh mesh;
    if (!read_mesh(mesh, input))
    {
        gsInfo << "Failed to read OBJ file: " << input << "\n";
        return 1;
    }

    gsInfo << "Mesh has " << mesh.n_vertices() << " vertices and " << mesh.n_faces() << " faces.\n";

    gsMultiPatch<> mp = mesh.linear_patches();
    gsInfo << "Created MultiPatch with " << mp.nPatches() << " patches.\n";

    if (mp.empty())
    {
        gsInfo << "MultiPatch is empty, conversion failed.\n";
        return 1;
    }

    gsWrite(mp, output);
    gsInfo << "Successfully converted " << input << " to " << output << "\n";
    return 0;
}
