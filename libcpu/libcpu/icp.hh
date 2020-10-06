#pragma once

#include <vector>

#include "point-3d.hh"

namespace libcpu
{
    std::tuple<float, utils::Matrix<float>, Point3D>
    find_alignment(const point_list& p, const point_list& y);
}