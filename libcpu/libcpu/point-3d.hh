#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace libcpu
{
    struct Point3D
    {
        float x;
        float y;
        float z;
    };

    auto operator<<(std::ostream& os, const Point3D& point) -> std::ostream&;

    auto read_csv(const std::string& path, const std::string& x_field,
                  const std::string& y_field, const std::string& z_field)
        -> std::vector<Point3D>;

    float squared_distance(const Point3D& a, const Point3D& b);

    size_t closest(const Point3D& a, const std::vector<Point3D>& v);

    std::vector<Point3D> closest(const std::vector<Point3D>& a, const std::vector<Point3D>& b);
} // namespace libcpu
