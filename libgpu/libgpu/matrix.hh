#pragma once

#include <cassert>

#include "libcpu/point-3d.hh"
#include "libcpu/utils/matrix.hh"

namespace libgpu
{
    class GPUMatrix
    {
        float* ptr;
        bool should_delete;

    public:
        const size_t rows;
        const size_t cols;

        GPUMatrix(size_t rows, size_t cols);
        ~GPUMatrix();

        GPUMatrix(const GPUMatrix& other);
        GPUMatrix& operator=(const GPUMatrix& other) = delete;

        GPUMatrix(GPUMatrix&& other) noexcept;
        GPUMatrix& operator=(GPUMatrix&& other) noexcept;

        static GPUMatrix zero(size_t rows, size_t cols);
        static GPUMatrix eye(size_t n);

        inline __device__ float& operator()(size_t i, size_t j)
        {
            assert(i < rows);
            assert(j < cols);
            return this->ptr[i * cols + j];
        }

        inline __device__ float operator()(size_t i, size_t j) const
        {
            assert(i < rows);
            assert(j < cols);
            return this->ptr[i * cols + j];
        }

        static GPUMatrix from_point_list(const libcpu::point_list& p);

        libcpu::point_list to_point_list() const;

        static GPUMatrix from_cpu(const utils::Matrix<float>& cpu);

        utils::Matrix<float> to_cpu() const;

        GPUMatrix mean() const;

        GPUMatrix sum_colwise() const;

        GPUMatrix subtract(const GPUMatrix& matrix) const;

        GPUMatrix subtract_rowwise(const GPUMatrix& matrix) const;

        GPUMatrix dot(const GPUMatrix& matrix) const;

        GPUMatrix closest(const GPUMatrix& matrix) const;

        GPUMatrix transpose() const;

        static GPUMatrix find_covariance(const GPUMatrix& a,
                                         const GPUMatrix& b);

        __device__ static float distance(const GPUMatrix& a, size_t a_i,
                                         const GPUMatrix& b, size_t b_i)
        {
            assert(a.cols == b.cols);

            float dist = 0;
            for (size_t j = 0; j < a.cols; ++j)
            {
                float diff = a(a_i, j) - b(b_i, j);
                dist += diff * diff;
            }
            return dist;
        }
    };
} // namespace libgpu