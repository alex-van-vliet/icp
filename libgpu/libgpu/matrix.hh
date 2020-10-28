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

        /**
         * @brief Create a new matrix.
         * @param rows The number of rows.
         * @param cols The number of columns.
         */
        GPUMatrix(size_t rows, size_t cols);
        /**
         * @brief Destroy a matrix.
         */
        ~GPUMatrix();

        /**
         * @brief Creates a shallow copy of a matrix.
         * @param other The matrix to copy from.
         * This is mostly used to pass the GPUMatrix to GPU.
         */
        GPUMatrix(const GPUMatrix& other);
        GPUMatrix& operator=(const GPUMatrix& other) = delete;

        /**
         * @brief Move the matrix into a new one.
         * @param other The matrix to move from.
         */
        GPUMatrix(GPUMatrix&& other) noexcept;
        /**
         * @brief Move the matrix into an existing one.
         * @param other The matrix to move from.
         * @return This.
         */
        GPUMatrix& operator=(GPUMatrix&& other) noexcept;

        /**
         * @brief Create a new zero matrix.
         * @param rows The number of rows.
         * @param cols The number of columns.
         * @return The 0-initialized matrix.
         */
        static GPUMatrix zero(size_t rows, size_t cols);
        /**
         * @brief Create a new identity matrix.
         * @param n The number of rows and columns.
         * @return The identity-initialized matrix.
         */
        static GPUMatrix eye(size_t n);

        /**
         * @brief Access to one element.
         * @param i The row number.
         * @param j The column number.
         * @return The element.
         */
        inline __device__ float& operator()(size_t i, size_t j)
        {
            assert(i < rows);
            assert(j < cols);
            return this->ptr[i * cols + j];
        }
        /**
         * @brief Access to one element.
         * @param i The row number.
         * @param j The column number.
         * @return The element.
         */
        inline __device__ float operator()(size_t i, size_t j) const
        {
            assert(i < rows);
            assert(j < cols);
            return this->ptr[i * cols + j];
        }

        /**
         * @brief Convert a point list to a matrix.
         * @param p The point list.
         * @return The new matrix.
         */
        static GPUMatrix from_point_list(const libcpu::point_list& p);

        /**
         * @brief Convert the matrix to a point list.
         * @return The point list.
         */
        libcpu::point_list to_point_list() const;

        /**
         * @brief Convert a cpu matrix to a gpu matrix.
         * @param cpu The cpu matrix.
         * @return The new matrix.
         */
        static GPUMatrix from_cpu(const utils::Matrix<float>& cpu);

        /**
         * @brief Convert the gpu matrix to a cpu matrix.
         * @return The new matrix.
         */
        utils::Matrix<float> to_cpu() const;

        /**
         * @brief Get the mean of the points.
         * @return The matrix containing the mean of the points.
         */
        GPUMatrix mean() const;

        /**
         * @brief Sum each rows of the matrix column by column.
         * @return The matrix containing the sums.
         */
        GPUMatrix sum_colwise() const;

        /**
         * @brief Subtract the matrix by another one.
         * @param matrix The other matrix.
         * @return The matrix containing the subtraction.
         */
        GPUMatrix subtract(const GPUMatrix& matrix) const;

        /**
         * @brief Subtract each row of the matrix by a row vector.
         * @param matrix The row vector.
         * @return The matrix containing the subtraction.
         */
        GPUMatrix subtract_rowwise(const GPUMatrix& matrix) const;

        /**
         * @brief Compute the dot product between the matrix and another one.
         * @param matrix The other matrix.
         * @return The matrix containing the dot product.
         */
        GPUMatrix dot(const GPUMatrix& matrix) const;

        /**
         * @brief Find the closest point of each point in this matrix in the
         * other matrix using a simple search.
         * @param matrix The other matrix.
         * @return The matrix containing the closest points.
         */
        GPUMatrix closest(const GPUMatrix& matrix) const;

        /**
         * @brief Transpose the matrix.
         * @return The matrix containing the transpose.
         */
        GPUMatrix transpose() const;

        /**
         * @brief Find the covariance matrix between two matrices.
         * @param a The first matrix.
         * @param b The second matrix.
         * @return The covariance matrix.
         */
        static GPUMatrix find_covariance(const GPUMatrix& a,
                                         const GPUMatrix& b);

        /**
         * @brief Find the squared distance between two points in to matrix.
         * @param a The first matrix.
         * @param a_i The index of the point in the first matrix.
         * @param b The second matrix.
         * @param b_i The index of the point in the second matrix.
         * @return The squared distance between the two points.
         */
        __device__ static float squared_distance(const GPUMatrix& a, size_t a_i,
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