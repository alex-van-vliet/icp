#pragma once

#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "matrix.hh"

namespace libcpu
{
    struct Point3D
    {
        float x;
        float y;
        float z;
    };

    using point_list = std::vector<Point3D>;

    /**
     * @brief Print a point to a stream.
     * @param os The stream.
     * @param point The point.
     * @return The stream.
     */
    auto operator<<(std::ostream& os, const Point3D& point) -> std::ostream&;

    /**
     * @brief Check if two points are the same, in a float compatible way.
     * @param p1 The first point.
     * @param p2 The second point.
     * @return Whether they are the same.
     */
    bool operator==(const Point3D& p1, const Point3D& p2);

    /**
     * @brief Parse a csv.
     * @param path The path to the file.
     * @param x_field The name of the x field.
     * @param y_field The name of the y field.
     * @param z_field The name of the z field.
     * @return The point list.
     */
    auto read_csv(const std::string& path, const std::string& x_field,
                  const std::string& y_field, const std::string& z_field)
        -> point_list;

    /**
     * @brief Compute the squared distance between two points.
     * @param a The first point.
     * @param b The second point.
     * @return The squared distance.
     */
    float squared_distance(const Point3D& a, const Point3D& b);

    /**
     * @brief Find the closest point to a point in a point list.
     * @param a The query.
     * @param v The points.
     * @return The index of the closest point in the points.
     */
    size_t closest(const Point3D& a, const point_list& v);

    /**
     * @brief Find the closest points in the second point list to each point in
     * the first point list.
     * @param a The queries.
     * @param b The points.
     * @return The closest points.
     */
    point_list closest(const point_list& a, const point_list& b);

    /**
     * @brief Compute the mean of a point list.
     * @param a The point list.
     * @return The mean.
     */
    Point3D mean(const point_list& a);

    /**
     * @brief Compute the sum of squared norms of a point list.
     * @param a The point list.
     * @return The sum of squared norms.
     */
    float sum_of_squared_norms(const point_list& a);

    /**
     * @brief Subtract a vector to each point in a point list.
     * @param points The point list.
     * @param mean The vector.
     * @return The point list containing the subtraction.
     */
    point_list subtract(const point_list& points, const Point3D& mean);

    /**
     * @brief Compute the covariance matrix between two point clouds.
     * @param p_centered The first point cloud.
     * @param y_centered The second point cloud.
     * @return The covariance matrix.
     */
    std::tuple<float, float, float, float, float, float, float, float, float>
    find_covariance(const point_list& p_centered, const point_list& y_centered);

    /**
     * @brief Compute the dot product between a matrix and a point.
     * @param a The matrix.
     * @param b The point.
     * @return The point containing the dot product.
     */
    Point3D dot(const utils::Matrix<float>& a, const Point3D& b);

    /**
     * @brief Compute the subtraction between two points.
     * @param a The first point.
     * @param b The second point.
     * @return The point containing the subtraction.
     */
    Point3D subtract(const Point3D& a, const Point3D& b);

    /**
     * @brief Scale a point by a float.
     * @param a The float.
     * @param b The point.
     * @return The scaled point.
     */
    Point3D scale(float a, const Point3D& b);
} // namespace libcpu
