#pragma once

#include <memory>
#include <tuple>

#include "point-3d.hh"

namespace libgpu
{
    class GPUVPTree;
}

namespace libcpu
{
    class VPTree
    {
        friend class libgpu::GPUVPTree;

    private:
        // Internal nodes
        std::unique_ptr<VPTree> inside;
        std::unique_ptr<VPTree> outside;
        Point3D center;
        float radius;
        // Leaves
        point_list points;

    public:
        /**
         * @brief Create a vp tree.
         * @param threshold The capacity.
         * @param points The points.
         */
        VPTree(uint threshold, point_list points);

        /**
         * @brief Search the tree.
         * @param query The query point.
         * @return The closest point and its distance to the query.
         */
        std::tuple<Point3D, float> search(const Point3D& query) const;

        /**
         * @brief Find the closest point of each point in the matrix.
         * @param queries The queries.
         * @return The point list containing the closest points.
         */
        point_list closest(const point_list& queries);
    };
} // namespace libcpu
