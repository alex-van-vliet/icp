#pragma once

#include <vector>
#include <string>
#include <ostream>

namespace libcpu
{
    struct Point3D
    {
        float x;
        float y;
        float z;
    };

    auto operator<<(std::ostream& os, const Point3D& point)
    -> std::ostream&;

    auto read_csv(const std::string& path, const std::string& x_field,
        const std::string& y_field, const std::string& z_field)
    -> std::vector<Point3D>;
}
