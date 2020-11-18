#include "libcpu/point-3d.hh"

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

    auto read_csv(const std::string& path, const std::string& x_field,
                  const std::string& y_field, const std::string& z_field)
        -> std::vector<Point3D>
    {
        std::vector<Point3D> points;

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
} // namespace libcpu