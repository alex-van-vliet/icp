#include "vp-tree.hh"

#include <iostream>

namespace libcpu
{
    size_t partition(float* array, size_t n)
    {
        float pivot = array[0];

        size_t left = 0;
        size_t right = n - 1;

        while (true)
        {
            while (array[right] > pivot && right > left)
                right -= 1;
            while (array[left] <= pivot && right > left)
                left += 1;
            if (right <= left)
                break;
            std::swap(array[left], array[right]);
        }
        std::swap(array[left], array[0]);
        return left;
    }

    /**
     * @brief Select the nth value in order in an array using quick select.
     * @param array The array.
     * @param n The number of the value in order.
     * @param s The size of the array.
     * @return The value at the position.
     */
    float select(float* array, size_t n, size_t s)
    {
        size_t pivot_position = partition(array, n);

        if (s < pivot_position)
            return select(array, pivot_position, s);
        else if (s == pivot_position)
            return array[pivot_position];
        else
            return select(array + pivot_position + 1, n - pivot_position - 1,
                          s - pivot_position - 1);
    }

    float median(std::vector<float> array)
    {
        // Does not handle specially the case when array.size() is even
        return select(array.data(), array.size(), array.size() / 2);
    }

    VPTree::VPTree(uint threshold, point_list points)
    {
        if (points.size() < threshold)
        {
            this->points = points;
            this->inside = nullptr;
            this->outside = nullptr;
        }
        else
        {
            // Use the last point as center
            this->center = points.back();
            points.pop_back();

            // Compute the distances from each point to the center.
            std::vector<float> distances;
            distances.resize(points.size());

#pragma omp parallel for
            for (size_t i = 0; i < points.size(); ++i)
                distances[i] = squared_distance(this->center, points[i]);

            // Get the median distance.
            float squared_radius = median(distances);
            this->radius = sqrt(squared_radius);

            // Split the points into inside and outside.
            point_list inside_points;
            point_list outside_points;

            float inside_max = 0;
            uint inside_keep = 0;

            float outside_max = 0;
            uint outside_keep = 0;

            for (size_t i = 0; i < points.size(); ++i)
            {
                float distance = distances[i];
                if (distance < squared_radius)
                {
                    if (distance > inside_max)
                    {
                        inside_max = distance;
                        inside_keep = inside_points.size();
                    }
                    inside_points.push_back(points[i]);
                }
                else
                {
                    if (distance > outside_max)
                    {
                        outside_max = distance;
                        outside_keep = outside_points.size();
                    }
                    outside_points.push_back(points[i]);
                }
            }

            // Put the furthest from the center at the end.
            std::swap(inside_points.back(), inside_points[inside_keep]);
            std::swap(outside_points.back(), outside_points[outside_keep]);

            // Recurse
            this->inside =
                std::make_unique<VPTree>(threshold, std::move(inside_points));
            this->outside =
                std::make_unique<VPTree>(threshold, std::move(outside_points));
        }
    }

    std::tuple<Point3D, float> VPTree::search(const Point3D& query) const
    {
        // If the node doesn't have children.
        if (!this->inside)
        {
            // Search the closest points linearly
            size_t closest_i = 0;
            float closest_dist = sqrt(squared_distance(query, points[0]));
            for (size_t i = 1; i < points.size(); ++i)
            {
                float dist = sqrt(squared_distance(query, points[i]));
                if (dist < closest_dist)
                {
                    closest_i = i;
                    closest_dist = dist;
                }
            }

            return {points[closest_i], closest_dist};
        }

        // Get the distance between the query and the center and recurse
        float d = sqrt(squared_distance(query, center));
        auto n = d < radius ? inside->search(query) : outside->search(query);

        // If the distance to the closest point is less than the distance to the
        // outside
        if (std::get<1>(n) < abs(radius - d))
        {
            if (d < std::get<1>(n))
                return {this->center, d};
            return n;
        }

        // Recurse on the other side
        auto o = d < radius ? outside->search(query) : inside->search(query);

        // Return the smallest
        if (std::get<1>(o) < std::get<1>(n))
        {
            if (d < std::get<1>(o))
                return {this->center, d};
            return o;
        }
        else
        {
            if (d < std::get<1>(n))
                return {this->center, d};
            return n;
        }
    }

    point_list VPTree::closest(const point_list& queries)
    {
        point_list v;
        v.resize(queries.size());

#pragma omp parallel for
        for (size_t i = 0; i < queries.size(); ++i)
            v[i] = std::get<0>(search(queries[i]));

        return v;
    }
} // namespace libcpu