#pragma once

#include <vector>

#include "libcpu/point-3d.hh"
#include "libcpu/utils/matrix.hh"

namespace libgpu
{
    /**
     * @brief The ICP algorithm on GPU.
     * @param m The reference point cloud.
     * @param p The transformed point cloud.
     * @param iterations The maximum number of iterations.
     * @param threshold The error threshold.
     * @param vp_threshold The vp tree capacity.
     * @return The transformation matrix and the p transformed.
     */
    std::tuple<utils::Matrix<float>, libcpu::point_list>
    icp(const libcpu::point_list& m, const libcpu::point_list& p,
        size_t iterations, float threshold, uint vp_threshold);
} // namespace libgpu