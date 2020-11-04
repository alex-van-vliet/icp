#pragma once

#include <vector>

#include "point-3d.hh"

namespace libcpu
{
    /**
     * @brief Apply the transformation matrix on the point cloud.
     * @param p The point cloud.
     * @param transformation The transformation matrix.
     */
    void apply_alignment(point_list& p, utils::Matrix<float> transformation);

    /**
     * @brief THE ICP algorithm on CPU.
     * @param m The reference point cloud.
     * @param p The transformed point cloud.
     * @param iterations The maximum number of iterations.
     * @param threshold The error threshold.
     * @param vp_threshold The vp tree capacity.
     * @return The transformation matrix and the p transformed.
     */
    std::tuple<utils::Matrix<float>, point_list>
    icp(const point_list& m, const point_list& p, size_t iterations,
        float threshold, uint vp_threshold);
} // namespace libcpu
