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

    VPTree::VPTree(uint threshold, const point_list& points)
    {
        if (points.size() < threshold)
        {
            this->points = points;
            this->inside = nullptr;
            this->outside = nullptr;
        }
        else
        {
            this->center = points.back();

            std::vector<float> distances;
            distances.resize(points.size());

#pragma omp parallel for
            for (size_t i = 0; i < points.size(); ++i)
                distances[i] = squared_distance(this->center, points[i]);

            float squared_radius = median(distances);
            this->radius = sqrt(squared_radius);

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

            std::swap(inside_points.back(), inside_points[inside_keep]);
            std::swap(outside_points.back(), outside_points[outside_keep]);

            this->inside = std::make_unique<VPTree>(threshold, inside_points);
            this->outside = std::make_unique<VPTree>(threshold, outside_points);
        }
    }

    std::tuple<Point3D, float> VPTree::search(const Point3D& query) const
    {
        if (!this->inside)
        {
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

        float d = sqrt(squared_distance(query, center));
        auto n = d < radius ? inside->search(query) : outside->search(query);

        if (std::get<1>(n) < abs(radius - d))
            return n;

        auto o = d < radius ? outside->search(query) : inside->search(query);

        if (std::get<1>(o) < std::get<1>(n))
            return o;
        else
            return n;
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