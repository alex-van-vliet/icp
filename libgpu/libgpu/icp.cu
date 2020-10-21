#include <functional>
#include <iostream>
#include <memory>

#include "icp.hh"

namespace libgpu
{
    namespace cuda
    {
        template <typename T>
        void free(T* ptr)
        {
            void* void_ptr = static_cast<void*>(ptr);
            cudaError_t error = cudaFree(void_ptr);
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
        }

        template <typename T>
        std::unique_ptr<T, decltype(&cuda::free<T>)> mallocManaged(size_t size)
        {
            void* ptr = nullptr;
            cudaError_t error = cudaMallocManaged(&ptr, sizeof(T) * size);
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
            T* t_ptr = static_cast<T*>(ptr);
            return std::unique_ptr<T, decltype(&cuda::free<T>)>(t_ptr,
                                                                cuda::free<T>);
        }
    } // namespace cuda

    __global__ void mykernel(float* ptr)
    {
        printf("Hello World from GPU!\n");
        printf("%.2f\n", ptr[0]);
    }

    std::tuple<utils::Matrix<float>, libcpu::point_list>
    icp(const libcpu::point_list& m, const libcpu::point_list& p,
        size_t iterations, float threshold)
    {
        auto ptr = cuda::mallocManaged<float>(3 * p.size());
        ptr.get()[0] = 4;
        cudaDeviceSynchronize();
        mykernel<<<1, 1>>>(ptr.get());
        cudaDeviceSynchronize();
        return {utils::Matrix<float>(4, 4), p};
    }
} // namespace libgpu