#pragma once

#include <vector>

#include "point-3d.hh"

namespace libcpu
{
    utils::Matrix<float> find_alignment(const point_list& p,
                                        const point_list& y);

    std::tuple<utils::Matrix<float>, point_list> icp(const point_list& m,
                                                     const point_list& p,
                                                     size_t iterations,
                                                     float threshold);

    void apply_alignment(point_list& p, utils::Matrix<float> transformation);

    float compute_error(const point_list& m, const point_list& p);
} // namespace libcpu