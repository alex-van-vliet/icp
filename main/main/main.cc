#include "libcpu/point-3d.hh"

#include <iostream>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path/to/data.txt>" << std::endl;
        return 1;
    }

    for (const auto& point : libcpu::read_csv(argv[1],
        "Points_0", "Points_1", "Points_2"))
        std::cout << point << std::endl;
}