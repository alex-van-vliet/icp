#include <Eigen/Dense>
#include <iostream>

#include "libcpu/icp.hh"
#include "libgpu/icp.hh"
#include "options.hh"

int main(int argc, char* argv[])
{
    options::options options;
    if (!options::parse_options(options, argc - 1, argv + 1))
    {
        options::show_help(argv[0]);
        return 1;
    }

    std::cout << "Parameters:" << std::endl;
    std::cout << "    Device: " << (options.gpu ? "GPU" : "CPU") << std::endl;
    std::cout << "    Capacity: " << options.capacity << std::endl;
    std::cout << "    Max iterations: " << options.iterations << std::endl;
    std::cout << "    Error threshold: " << options.error << std::endl;

    const char* q_path = options.reference;
    auto q = libcpu::read_csv(q_path, "Points_0", "Points_1", "Points_2");
    const char* p_path = options.transformed;
    auto p = libcpu::read_csv(p_path, "Points_0", "Points_1", "Points_2");

    auto [transform, new_p] = options.gpu
        ? libgpu::icp(q, p, options.iterations, options.error, options.capacity)
        : libcpu::icp(q, p, options.iterations, options.error,
                      options.capacity);

    std::cout << "Transformation: " << std::endl;
    for (size_t i = 0; i < transform.rows; ++i)
    {
        for (size_t j = 0; j < transform.cols; ++j)
        {
            std::cout << transform(i, j) << ' ';
        }
        std::cout << std::endl;
    }

    std::cout << "Points" << std::endl;
    for (size_t i = 0; i < new_p.size() && i < 10; ++i)
    {
        std::cout << q[i] << " - " << new_p[i] << std::endl;
    }
}