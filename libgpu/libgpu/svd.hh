#pragma once

#include "matrix.hh"

namespace libgpu
{
    GPUMatrix find_rotation(const GPUMatrix& covariance);
}