#include "libcpu/point-3d.hh"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <tuple>

namespace
{
    auto parse_line(const std::string& line, size_t x_field_id,
                    size_t y_field_id, size_t z_field_id) -> libcpu::Point3D
    {
        size_t max_id = std::max(std::max(x_field_id, y_field_id), z_field_id);

        libcpu::Point3D point{0, 0, 0};

        size_t i = 0;
        size_t start = 0;
        size_t end = 0;
        do
        {
            end = line.find(',', start);

            if (i == x_field_id)
            {
                std::string token(line, start, end - start);
                point.x = std::stof(token);
            }
            else if (i == y_field_id)
            {
                std::string token(line, start, end - start);
                point.y = std::stof(token);
            }
            else if (i == z_field_id)
            {
                std::string token(line, start, end - start);
                point.z = std::stof(token);
            }

            start = end + 1;
            i += 1;
        } while (end != std::string::npos && i <= max_id);

        if (i <= max_id)
        {
            throw std::runtime_error("could not find one of the coordinates");
        }

        return point;
    }

    auto parse_header(const std::string& header, const std::string& x_field,
                      const std::string& y_field, const std::string& z_field)
        -> std::tuple<size_t, size_t, size_t>
    {
        size_t x_field_id = -1;
        size_t y_field_id = -1;
        size_t z_field_id = -1;

        size_t i = 0;
        size_t start = 0;
        size_t end = 0;
        do
        {
            end = header.find(',', start);

            std::string token(header, start, end - start);

            if (token == x_field)
                x_field_id = i;
            else if (token == y_field)
                y_field_id = i;
            else if (token == z_field)
                z_field_id = i;

            start = end + 1;
            i += 1;
        } while (end != std::string::npos);

        if ((x_field_id == size_t(-1)) or (y_field_id == size_t(-1))
            or (z_field_id == size_t(-1)))
        {
            throw std::runtime_error("could not find one of the coordinates");
        }

        return {x_field_id, y_field_id, z_field_id};
    }
} // namespace

namespace libcpu
{
    auto operator<<(std::ostream& os, const Point3D& point) -> std::ostream&
    {
        return os << "(" << point.x << ", " << point.y << ", " << point.z
                  << ")";
    }

    bool operator==(const Point3D& p1, const Point3D& p2)
    {
        float epsilon = 0.005f;
        return abs(p1.x - p2.x) < epsilon && abs(p1.y - p2.y) < epsilon
            && abs(p1.z - p2.z) < epsilon;
    }

    auto read_csv(const std::string& path, const std::string& x_field,
                  const std::string& y_field, const std::string& z_field)
        -> point_list
    {
        point_list points;

        std::ifstream stream(path);
        if (!stream)
        {
            throw std::runtime_error("could not open file");
        }

        std::string header;
        if (!std::getline(stream, header))
        {
            throw std::runtime_error("header line not found");
        }

        auto fields_id = parse_header(header, x_field, y_field, z_field);

        std::string line;
        while (std::getline(stream, line))
        {
            points.push_back(parse_line(line, std::get<0>(fields_id),
                                        std::get<1>(fields_id),
                                        std::get<2>(fields_id)));
        }

        return points;
    }

    float squared_distance(const Point3D& a, const Point3D& b)
    {
        float x = a.x - b.x;
        float y = a.y - b.y;
        float z = a.z - b.z;

        return x * x + y * y + z * z;
    }

    size_t closest(const Point3D& a, const point_list& v)
    {
        assert(v.size() > 0);

        size_t ret = 0;
        float dist = squared_distance(a, v[ret]);

        for (size_t i = 1; i < v.size(); i++)
        {
            float tmp_dist = squared_distance(a, v[i]);
            if (tmp_dist < dist)
            {
                dist = tmp_dist;
                ret = i;
            }
        }

        return ret;
    }

    point_list closest(const point_list& a, const point_list& b)
    {
        point_list v;
        v.resize(a.size());

#pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i)
            v[i] = b[closest(a[i], b)];

        return v;
    }

    Point3D mean(const point_list& a)
    {
        size_t len = a.size();
        float x = 0;
        float y = 0;
        float z = 0;

        for (size_t i = 0; i < len; ++i)
        {
            x += a[i].x / len;
            y += a[i].y / len;
            z += a[i].z / len;
        }

        return Point3D{x, y, z};
    }

    point_list subtract(const point_list& points, const Point3D& mean)
    {
        point_list centered;
        centered.resize(points.size());

#pragma omp parallel for
        for (size_t i = 0; i < points.size(); ++i)
        {
            centered[i] = Point3D{
                points[i].x - mean.x,
                points[i].y - mean.y,
                points[i].z - mean.z,
            };
        }

        return centered;
    }

    std::tuple<float, float, float, float, float, float, float, float, float>
    find_covariance(const point_list& p_centered, const point_list& y_centered)
    {
        float sxx = 0, sxy = 0, sxz = 0, syx = 0, syy = 0, syz = 0, szx = 0,
              szy = 0, szz = 0;

        for (size_t i = 0; i < p_centered.size(); ++i)
        {
#define ADDPROD(FIRST_COORD, SECOND_COORD)                                     \
    s##FIRST_COORD##SECOND_COORD +=                                            \
        (p_centered[i].FIRST_COORD) * (y_centered[i].SECOND_COORD)
            ADDPROD(x, x);
            ADDPROD(x, y);
            ADDPROD(x, z);
            ADDPROD(y, x);
            ADDPROD(y, y);
            ADDPROD(y, z);
            ADDPROD(z, x);
            ADDPROD(z, y);
            ADDPROD(z, z);
#undef ADDPROD
        }

        return {sxx, sxy, sxz, syx, syy, syz, szx, szy, szz};
    }

    float sum_of_squared_norms(const point_list& a)
    {
        float sum = 0;

        for (const auto& value : a)
            sum += value.x * value.x + value.y * value.y + value.z * value.z;

        return sum;
    }

    Point3D dot(const utils::Matrix<float>& a, const Point3D& b)
    {
        assert(a.rows == 3);
        assert(a.cols == 3);
        return {
            a.get(0, 0) * b.x + a.get(0, 1) * b.y + a.get(0, 2) * b.z,
            a.get(1, 0) * b.x + a.get(1, 1) * b.y + a.get(1, 2) * b.z,
            a.get(2, 0) * b.x + a.get(2, 1) * b.y + a.get(2, 2) * b.z,
        };
    }

    Point3D subtract(const Point3D& a, const Point3D& b)
    {
        return {
            a.x - b.x,
            a.y - b.y,
            a.z - b.z,
        };
    }

    Point3D scale(float a, const Point3D& b)
    {
        return {
            b.x * a,
            b.y * a,
            b.z * a,
        };
    }
} // namespace libcpu
