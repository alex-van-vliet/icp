#include "icp.hh"

#include <Eigen/Dense>
#include <iostream>

#include "matrix.hh"

namespace libcpu
{
    utils::Matrix<float> to_transformation(const utils::Matrix<float>& r,
                                           Point3D p)
    {
        utils::Matrix<float> transformation(4, 4);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                transformation.set(i, j, r.get(i, j));

        transformation.set(0, 3, p.x);
        transformation.set(1, 3, p.y);
        transformation.set(2, 3, p.z);

        transformation.set(3, 3, 1);
        return transformation;
    }

    utils::Matrix<float> find_alignment(const point_list& p,
                                        const point_list& y)
    {
        auto mu_p = mean(p);
        auto mu_y = mean(y);

        auto p_centered = subtract(p, mu_p);
        auto y_centered = subtract(y, mu_y);

        auto [sxx, sxy, sxz, syx, syy, syz, szx, szy, szz] =
            find_covariance(p_centered, y_centered);

        Eigen::Matrix3f matrix;
        matrix(0, 0) = sxx;
        matrix(0, 1) = syx;
        matrix(0, 2) = szx;
        matrix(1, 0) = sxy;
        matrix(1, 1) = syy;
        matrix(1, 2) = szy;
        matrix(2, 0) = sxz;
        matrix(2, 1) = syz;
        matrix(2, 2) = szz;

        Eigen::JacobiSVD svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix3f rotation = svd.matrixV() * svd.matrixU().transpose();

        utils::Matrix<float> r{
            {rotation(0, 0), rotation(0, 1), rotation(0, 2)},
            {rotation(1, 0), rotation(1, 1), rotation(1, 2)},
            {rotation(2, 0), rotation(2, 1), rotation(2, 2)},
        };

        /*
        // TRANSPOSEE INCORPOREE
        utils::Matrix<float> qbar{
            {q0, q1, q2, q3},
            {-q1, q0, -q3, q2},
            {-q2, q3, q0, -q1},
            {-q3, -q2, q1, q0},
        };
        utils::Matrix<float> q{
            {q0, -q1, -q2, -q3},
            {q1, q0, -q3, q2},
            {q2, q3, q0, -q1},
            {q3, -q2, q1, q0},
        };
        auto rotation = utils::dot(qbar, q);
        auto r = rotation.submatrix(1, 4, 1, 4);

        for (size_t i = 0; i < 3; ++i)
        {
            r.set(i, 0, r.get(i, 0) * s.x);
            r.set(i, 1, r.get(i, 1) * s.y);
            r.set(i, 2, r.get(i, 2) * s.z);
        }
         */

        auto t = subtract(mu_y, dot(r, mu_p));

        return to_transformation(r, t);
    }

    void apply_alignment(point_list& p, utils::Matrix<float> transformation)
    {
        for (auto& value : p)
        {
            float x = value.x * transformation.get(0, 0)
                + value.y * transformation.get(0, 1)
                + value.z * transformation.get(0, 2) + transformation.get(0, 3);
            float y = value.x * transformation.get(1, 0)
                + value.y * transformation.get(1, 1)
                + value.z * transformation.get(1, 2) + transformation.get(1, 3);
            float z = value.x * transformation.get(2, 0)
                + value.y * transformation.get(2, 1)
                + value.z * transformation.get(2, 2) + transformation.get(2, 3);
            value.x = x;
            value.y = y;
            value.z = z;
        }
    }

    std::tuple<utils::Matrix<float>, point_list> icp(const point_list& m,
                                                     const point_list& p)
    {
        auto transformation = utils::eye<float>(4);

        auto new_p = p;

        size_t max_iters = 3;
        for (size_t i = 0; i < max_iters; ++i)
        {
            std::cerr << "Starting iter " << (i + 1) << "/" << max_iters
                      << std::endl;
            auto y = closest(new_p, m);
            auto new_transformation = find_alignment(new_p, y);

            transformation = dot(new_transformation, transformation);
            apply_alignment(new_p, new_transformation);
        }

        return {transformation, new_p};
    }
} // namespace libcpu