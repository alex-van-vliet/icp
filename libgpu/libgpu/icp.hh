#pragma once

#include <vector>

#include "libcpu/point-3d.hh"
#include "libcpu/utils/matrix.hh"

namespace libgpu
{
    std::tuple<utils::Matrix<float>, libcpu::point_list>
    icp(const libcpu::point_list& m, const libcpu::point_list& p,
        size_t iterations, float threshold);
} // namespace libgpu