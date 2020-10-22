#pragma once

#include "libcpu/utils/matrix.hh"

namespace libgpu
{
    utils::Matrix<float> find_rotation(const utils::Matrix<float>& covariance);
}