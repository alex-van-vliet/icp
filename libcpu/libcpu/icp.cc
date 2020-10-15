#include "icp.hh"

#include <iostream>

#include "matrix.hh"

namespace libcpu
{
    utils::Matrix<float> to_transformation(float s, utils::Matrix<float> r,
                                           Point3D p)
    {
        utils::Matrix<float> transformation(4, 4);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                transformation.set(i, j, r.get(i, j) * s);

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

        utils::Matrix<float> matrix{
            {sxx + syy + szz, syz - szy, -sxz + szx, sxy - syx},
            {-szy + syz, sxx - szz - syy, sxy + syx, sxz + szx},
            {szx - sxz, syx + sxy, syy - szz - sxx, syz + szy},
            {-syx + sxy, szx + sxz, szy + syz, szz - syy - sxx},
        };

        auto quaternion = matrix.largest_eigenvector<10>();

        auto q0 = quaternion.get(0, 0);
        auto q1 = quaternion.get(1, 0);
        auto q2 = quaternion.get(2, 0);
        auto q3 = quaternion.get(3, 0);

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

        auto s = sqrt(sum_of_squared_norms(y_centered)
                      / sum_of_squared_norms(p_centered));

        auto t = subtract(mu_y, dot(scale(s, r), mu_p));

        return to_transformation(s, r, t);
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
            auto new_transformation = find_alignment(y, new_p);

            transformation = dot(new_transformation, transformation);
            apply_alignment(new_p, new_transformation);
        }

        return {transformation, new_p};
    }
} // namespace libcpu