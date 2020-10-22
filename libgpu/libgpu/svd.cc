#include "svd.hh"

#include <Eigen/Dense>

namespace libgpu
{
    utils::Matrix<float> find_rotation(const utils::Matrix<float>& covariance)
    {
        assert(covariance.rows == 3);
        assert(covariance.cols == 3);

        Eigen::Matrix3f matrix(3, 3);
        for (size_t i = 0; i < covariance.rows; ++i)
            for (size_t j = 0; j < covariance.cols; ++j)
                matrix(i, j) = covariance(i, j);

        Eigen::JacobiSVD svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix3f rotation = svd.matrixU() * svd.matrixV().transpose();

        // Transpose incorporated
        utils::Matrix<float> res(3, 3);
        for (size_t i = 0; i < res.rows; ++i)
            for (size_t j = 0; j < res.cols; ++j)
                res(i, j) = rotation(j, i);

        return res;
    }
} // namespace libgpu