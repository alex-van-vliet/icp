#include <iostream>
#include <limits>

#include "cuda/memory.hh"
#include "icp.hh"
#include "matrix.hh"
#include "svd.hh"

namespace libgpu
{
    __global__ void to_transformation_kernel(GPUMatrix rotation,
                                             GPUMatrix translation,
                                             GPUMatrix res)
    {
        uint i = threadIdx.x;
        uint j = threadIdx.y;

        if (i < 3)
        {
            if (j < 3)
                res(i, j) = rotation(i, j);
            else // i < 3 && j == 3
                res(i, 3) = translation(0, i);
        }
        else // i == 3
        {
            if (j < 3)
                res(3, j) = 0;
            else // i == 3 && j == 3
                res(3, 3) = 1;
        }
    }

    GPUMatrix to_transformation(const GPUMatrix& rotation,
                                const GPUMatrix& translation)
    {
        assert(rotation.rows == 3);
        assert(rotation.cols == 3);
        assert(translation.rows == 1);
        assert(translation.cols == 3);

        GPUMatrix transformation(4, 4);

        dim3 blockdim(4, 4);
        to_transformation_kernel<<<1, blockdim>>>(rotation, translation,
                                                  transformation);

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

        auto rotation = GPUMatrix::from_cpu(find_rotation(covariance.to_cpu()));

        auto translation = mu_m.subtract(mu_p.dot(rotation.transpose()));

        return to_transformation(rotation, translation);
    }

    __global__ void compute_error_kernel(GPUMatrix m, GPUMatrix p,
                                         float* error_d)
    {
        float error = 0;

        for (size_t i = 0; i < m.rows; ++i)
            error += GPUMatrix::distance(m, i, p, i);

        *error_d = error;
    }

    float compute_error(const GPUMatrix& m, const GPUMatrix& p)
    {
        auto error_d = cuda::malloc<float>(1);

        compute_error_kernel<<<1, 1>>>(m, p, error_d.get());

        float error = 0;
        cudaMemcpy(&error, error_d.get(), sizeof(float),
                   cudaMemcpyDeviceToHost);

        return error;
    }

    __global__ void apply_alignment_kernel(GPUMatrix p,
                                           GPUMatrix transformation)
    {
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

    void apply_alignment(GPUMatrix& p, const GPUMatrix& transformation)
    {
        assert(p.cols == 3);
        assert(transformation.rows == 4);
        assert(transformation.cols == 4);

        apply_alignment_kernel<<<1, 1>>>(p, transformation);
    }

    std::tuple<utils::Matrix<float>, libcpu::point_list>
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

        return {transformation.to_cpu(), new_p.to_point_list()};
    }
} // namespace libgpu