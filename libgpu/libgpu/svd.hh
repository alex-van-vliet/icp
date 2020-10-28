#pragma once

#include "libcpu/utils/matrix.hh"

namespace libgpu
{
    /**
     * @brief Determine the rotation matrix using a SVD with eigen.
     * @param covariance The covariance matrix.
     * @return The rotation matrix.
     */
    utils::Matrix<float> find_rotation(const utils::Matrix<float>& covariance);
} // namespace libgpu