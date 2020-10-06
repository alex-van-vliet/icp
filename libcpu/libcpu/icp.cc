#include "icp.hh"

#include "matrix.hh"

namespace libcpu
{
    std::tuple<float, utils::Matrix<float>, Point3D>
    find_alignment(const point_list& p, const point_list& y)
    {
        auto mu_p = mean(p);
        auto mu_y = mean(y);

        auto p_centered = subtract(p, mu_p);
        auto y_centered = subtract(p, mu_y);

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

        auto t = subtract(mu_y, scale(s, dot(r, mu_p)));

        return {s, r, t};
    }
} // namespace libcpu