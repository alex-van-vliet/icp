#pragma once

#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace libcpu
{
    struct Point3D
    {
        float x;
        float y;
        float z;
    };

    using point_list = std::vector<Point3D>;

    auto operator<<(std::ostream& os, const Point3D& point) -> std::ostream&;

    auto read_csv(const std::string& path, const std::string& x_field,
                  const std::string& y_field, const std::string& z_field)
        -> point_list;

    float squared_distance(const Point3D& a, const Point3D& b);

    size_t closest(const Point3D& a, const point_list& v);

    point_list closest(const point_list& a, const point_list& b);

    Point3D mean(const point_list& a);

    float sum_of_squared_norms(const point_list& a);

    point_list subtract(const point_list& points, const Point3D& mean);

    std::tuple<float, float, float, float, float, float, float, float, float>
    find_covariance(const point_list& p_centered, const point_list& y_centered);
} // namespace libcpu
