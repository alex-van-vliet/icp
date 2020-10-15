#include <iostream>
#include <unistd.h>

#include "libcpu/icp.hh"
#include "libcpu/point-3d.hh"

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <path/to/data.txt> <path/to/data.txt>" << std::endl;
        return 1;
    }

    srand(static_cast<unsigned>(getpid()));

    auto m = libcpu::read_csv(argv[1], "Points_0", "Points_1", "Points_2");
    auto p = libcpu::read_csv(argv[2], "Points_0", "Points_1", "Points_2");

    auto [transform, new_p] = libcpu::icp(m, p);

    for (size_t i = 0; i < transform.lines; ++i)
    {
        for (size_t j = 0; j < transform.columns; ++j)
        {
            std::cout << transform.get(i, j) << ' ';
        }
        std::cout << std::endl;
    }
    for (size_t i = 0; i < new_p.size() && i < 10; ++i)
    {
        std::cout << p[i] << " - " << new_p[i] << std::endl;
    }
}