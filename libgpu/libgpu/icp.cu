#include <iostream>
#include <limits>

#include "cuda/memory.hh"
#include "icp.hh"
#include "svd.hh"

namespace libgpu
{
    /*
    GPUMatrix find_rotation(const GPUMatrix& covariance)
    {
        assert(covariance.rows == 3);
        assert(covariance.cols == 3);

        cusolverDnHandle_t cusolverH;
        cublasHandle_t cublasH;
        const int lda = covariance.rows;

        auto matrix = cuda::mallocManaged<float>(lda * covariance.cols);
        for (size_t i = 0; i < covariance.rows; ++i)
            for (size_t j = 0; j < covariance.cols; ++j)
                matrix.get()[j * covariance.rows + i] = covariance(i, j);

        cudaDeviceSynchronize();

        auto U = cuda::mallocManaged<float>(lda * covariance.rows);
        auto VT = cuda::mallocManaged<float>(lda * covariance.cols);
        auto S = cuda::mallocManaged<float>(covariance.cols);
        auto Info = cuda::mallocManaged<int>(1);

        int lwork = 0;
        float *rwork;

        cusolverDnCreate(&cusolverH);
        cublasCreate(&cublasH);

        cusolverDnSgesvd_bufferSize(cusolverH, covariance.rows, covariance.cols,
    &lwork); auto work = cuda::mallocManaged<float>(lwork);

        cusolverDnSgesvd(cusolverH, 'A', 'A',
            covariance.rows, covariance.cols, matrix.get(), lda, S.get(),
    U.get(), lda, VT.get(), lda, work.get(), lwork, rwork, Info.get());
        cudaDeviceSynchronize();

        // free memory
        cudaFree(rwork);
        cublasDestroy(cublasH);
        cusolverDnDestroy(cusolverH);
    }
     */

    GPUMatrix to_transformation(const GPUMatrix& rotation,
                                const GPUMatrix& translation)
    {
        assert(rotation.rows == 3);
        assert(rotation.cols == 3);
        assert(translation.rows == 1);
        assert(translation.cols == 3);

        GPUMatrix transformation(4, 4);
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
                transformation(i, j) = rotation(i, j);

            transformation(i, 3) = translation(0, i);
            transformation(3, i) = 0;
        }

        transformation(3, 3) = 1;
        return transformation;
    }

    GPUMatrix find_alignment(const GPUMatrix& p, const GPUMatrix& m)
    {
        auto mu_p = p.mean();
        auto mu_m = m.mean();

        auto p_centered = p.subtract_rowwise(mu_p);
        auto m_centered = m.subtract_rowwise(mu_m);

        auto y = p_centered.closest(m_centered);

        auto covariance = GPUMatrix::find_covariance(p_centered, y);

        auto rotation = find_rotation(covariance);

        auto translation = mu_m.subtract(mu_p.dot(rotation.transpose()));

        return to_transformation(rotation, translation);
    }

    float compute_error(const GPUMatrix& m, const GPUMatrix& p)
    {
        float error = 0;

        for (size_t i = 0; i < m.rows; ++i)
            error += GPUMatrix::distance(m, i, p, i);

        return error;
    }

    void apply_alignment(GPUMatrix& p, const GPUMatrix& transformation)
    {
        assert(p.cols == 3);
        assert(transformation.rows == 4);
        assert(transformation.cols == 4);
        for (size_t i = 0; i < p.rows; ++i)
        {
            float values[3] = {0};
            for (size_t j = 0; j < p.cols; ++j)
            {
                for (size_t k = 0; k < 3; ++k)
                    values[j] += p(i, k) * transformation(j, k);
                values[j] += transformation(j, 3);
            }

            for (size_t j = 0; j < p.cols; ++j)
                p(i, j) = values[j];
        }
    }

    std::tuple<GPUMatrix, libcpu::point_list>
    icp(const libcpu::point_list& m_cpu, const libcpu::point_list& p,
        size_t iterations, float threshold)
    {
        auto new_p = GPUMatrix::from_point_list(p);
        auto m = GPUMatrix::from_point_list(m_cpu);

        auto transformation = GPUMatrix::eye(4);

        float error = std::numeric_limits<float>::infinity();

        for (size_t i = 0; i < iterations && error > threshold; ++i)
        {
            std::cerr << "Starting iter " << (i + 1) << "/" << iterations
                      << std::endl;
            auto new_transformation = find_alignment(new_p, m);

            transformation = new_transformation.dot(transformation);
            apply_alignment(new_p, new_transformation);
            error = compute_error(m, new_p);
            std::cerr << "Error: " << error << std::endl;
        }

        return {std::move(transformation), new_p.to_point_list()};
    }
} // namespace libgpu