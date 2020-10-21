#include <iostream>

#include "icp.hh"

namespace libgpu
{
    __global__ void mykernel()
    {
        printf("Hello World from GPU!\n");
    }

    std::tuple<utils::Matrix<float>, libcpu::point_list>
    icp(const libcpu::point_list& m, const libcpu::point_list& p,
        size_t iterations, float threshold)
    {
        mykernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return {utils::Matrix<float>(4, 4), p};
    }
} // namespace libgpu