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
                                        const point_list& m)
    {
        auto mu_p = mean(p);
        auto mu_m = mean(m);

        auto p_centered = subtract(p, mu_p);
        auto m_centered = subtract(m, mu_m);

        // auto y = closest(p_centered, m_centered);
        auto y = m_centered;

        auto [sxx, sxy, sxz, syx, syy, syz, szx, szy, szz] =
            find_covariance(p_centered, y);

        Eigen::Matrix3f matrix;
        matrix(0, 0) = sxx;
        matrix(0, 1) = sxy;
        matrix(0, 2) = sxz;
        matrix(1, 0) = syx;
        matrix(1, 1) = syy;
        matrix(1, 2) = syz;
        matrix(2, 0) = szx;
        matrix(2, 1) = szy;
        matrix(2, 2) = szz;

        Eigen::JacobiSVD svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix3f rotation = svd.matrixU() * svd.matrixV().transpose();

        utils::Matrix<float> r{
            {rotation(0, 0), rotation(0, 1), rotation(0, 2)},
            {rotation(1, 0), rotation(1, 1), rotation(1, 2)},
            {rotation(2, 0), rotation(2, 1), rotation(2, 2)},
        };

        auto t = subtract(mu_m, dot(r, mu_p));

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

    float compute_error(const point_list& m, const point_list& p)
    {
        float error = 0;

        for (size_t i = 0; i < m.size(); ++i)
        {
            float x = m[i].x - p[i].x;
            float y = m[i].y - p[i].y;
            float z = m[i].z - p[i].z;
            error += x * x + y * y + z * z;
        }

        return error;
    }

    std::tuple<utils::Matrix<float>, point_list>
    icp(const point_list& m, const point_list& p, size_t iterations)
    {
        auto transformation = utils::eye<float>(4);

        auto new_p = p;

        for (size_t i = 0; i < iterations; ++i)
        {
            std::cerr << "Starting iter " << (i + 1) << "/" << iterations
                      << std::endl;
            auto new_transformation = find_alignment(new_p, m);

            transformation = dot(new_transformation, transformation);
            apply_alignment(new_p, new_transformation);
            float error = compute_error(m, new_p);
            std::cerr << "Error: " << error << std::endl;
        }

        return {transformation, new_p};
    }
} // namespace libcpu