#include <gismo.h>
#include <gsMesh2/gsSurfMesh.h>
#include <gsMesh2/IO.h>

using namespace gismo;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        gsInfo << "Usage: " << argv[0] << " input.obj output.off\n";
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

    if (!write_mesh(mesh, output))
    {
        gsInfo << "Failed to write OFF file: " << output << "\n";
        return 1;
    }

    gsInfo << "Successfully converted " << input << " to " << output << "\n";
    return 0;
}
