#pragma once

#include <vector>

#include "point-3d.hh"

namespace libcpu
{
    utils::Matrix<float> find_alignment(const point_list& p,
                                        const point_list& y);

    std::tuple<utils::Matrix<float>, point_list> icp(const point_list& m,
                                                     const point_list& p);
} // namespace libcpu