#pragma once

#include <memory>
#include <system_error>

namespace libgpu
{
    namespace cuda
    {
        template <typename T>
        void free(T* ptr)
        {
            if (!ptr)
                return;

            void* void_ptr = static_cast<void*>(ptr);
            cudaError_t error = cudaFree(void_ptr);
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
        }

        template <typename T>
        using ptr_t = std::unique_ptr<T, decltype(&cuda::free<T>)>;

        template <typename T>
        T* mallocManagedRaw(size_t size)
        {
            void* ptr = nullptr;
            cudaError_t error = cudaMallocManaged(&ptr, sizeof(T) * size);
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
            return static_cast<T*>(ptr);
        }

        template <typename T>
        T* mallocRaw(size_t size)
        {
            void* ptr = nullptr;
            cudaError_t error = cudaMalloc(&ptr, sizeof(T) * size);
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
            return static_cast<T*>(ptr);
        }

        template <typename T>
        ptr_t<T> malloc(size_t size)
        {
            return ptr_t<T>(mallocRaw<T>(size), cuda::free<T>);
        }

        template <typename T>
        ptr_t<T> mallocManaged(size_t size)
        {
            return ptr_t<T>(mallocManagedRaw<T>(size), cuda::free<T>);
        }
    } // namespace cuda
} // namespace libgpu