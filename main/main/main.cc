#include <Eigen/Dense>
#include <iostream>
#include <sys/time.h>

#include "../../libcpu/libcpu/icp.hh"
#include "../../libgpu/libgpu/icp.hh"
#include "options.hh"

int main(int argc, char* argv[])
{
    options::options options;
    if (!options::parse_options(options, argc - 1, argv + 1))
    {
        options::show_help(argv[0]);
        return 1;
    }

    std::cerr << "Parameters:" << std::endl;
    std::cerr << "    Device: " << (options.gpu ? "GPU" : "CPU") << std::endl;
    std::cerr << "    Capacity: " << options.capacity << std::endl;
    std::cerr << "    Max iterations: " << options.iterations << std::endl;
    std::cerr << "    Error threshold: " << options.error << std::endl;

    const char* q_path = options.reference;
    auto q = libcpu::read_csv(q_path, "Points_0", "Points_1", "Points_2");
    const char* p_path = options.transformed;
    auto p = libcpu::read_csv(p_path, "Points_0", "Points_1", "Points_2");

    struct timeval start;
    gettimeofday(&start, nullptr);
    auto [transform, new_p] = options.gpu
        ? libgpu::icp(q, p, options.iterations, options.error, options.capacity)
        : libcpu::icp(q, p, options.iterations, options.error,
                      options.capacity);
    struct timeval end;
    gettimeofday(&end, nullptr);
    long start_time = start.tv_sec * 1000 + start.tv_usec / 1000;
    long end_time = end.tv_sec * 1000 + end.tv_usec / 1000;
    std::cerr << "Execution time: " << (end_time - start_time) << "ms"
              << std::endl;

    std::cout << "Transformation: " << std::endl;
    for (size_t i = 0; i < transform.rows; ++i)
    {
        std::cout << transform(i, 0);
        for (size_t j = 1; j < transform.cols; ++j)
            std::cout << ' ' << transform(i, j);
        std::cout << std::endl;
    }

    std::cout << "Points" << std::endl;
    if (q.size() > 10)
        q.resize(10);
    auto closest = libcpu::closest(q, new_p);
    for (size_t i = 0; i < q.size(); ++i)
        std::cerr << q[i] << " - " << closest[i] << std::endl;
}