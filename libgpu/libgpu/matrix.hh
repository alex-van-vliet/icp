#pragma once

#include <cassert>

#include "libcpu/point-3d.hh"

namespace libgpu
{
    class GPUMatrix
    {
        float* ptr;

    public:
        const size_t rows;
        const size_t cols;

        GPUMatrix(size_t rows, size_t cols);
        ~GPUMatrix();

        GPUMatrix(const GPUMatrix& other) = delete;
        GPUMatrix& operator=(const GPUMatrix& other) = delete;

        GPUMatrix(GPUMatrix&& other) noexcept;
        GPUMatrix& operator=(GPUMatrix&& other) noexcept;

        static GPUMatrix zero(size_t rows, size_t cols);
        static GPUMatrix eye(size_t n);

        inline float& operator()(size_t i, size_t j)
        {
            assert(i < rows);
            assert(j < cols);
            return this->ptr[i * cols + j];
        }

        inline float operator()(size_t i, size_t j) const
        {
            assert(i < rows);
            assert(j < cols);
            return this->ptr[i * cols + j];
        }

        static GPUMatrix from_point_list(const libcpu::point_list& p);

        libcpu::point_list to_point_list() const;

        GPUMatrix mean() const;

        GPUMatrix subtract(const GPUMatrix& matrix) const;

        GPUMatrix subtract_rowwise(const GPUMatrix& matrix) const;

        GPUMatrix dot(const GPUMatrix& matrix) const;

        GPUMatrix closest(const GPUMatrix& matrix) const;

        GPUMatrix transpose() const;

        static GPUMatrix find_covariance(const GPUMatrix& a,
                                         const GPUMatrix& b);

        static float distance(const GPUMatrix& a, size_t a_i,
                              const GPUMatrix& b, size_t b_i);
    };
} // namespace libgpu