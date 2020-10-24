#pragma once

#include <memory>
#include <tuple>

#include "point-3d.hh"

namespace libcpu
{
    class VPTree
    {
    private:
        // Internal nodes
        std::unique_ptr<VPTree> inside;
        std::unique_ptr<VPTree> outside;
        Point3D center;
        float radius;
        // Leaves
        point_list points;

    public:
        VPTree(const point_list& points);

        std::tuple<Point3D, float> search(const Point3D& query) const;

        point_list closest(const point_list& queries);
    };
    float median(std::vector<float> array);
} // namespace libcpu
