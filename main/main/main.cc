#include <Eigen/Dense>
#include <iostream>
#include <unistd.h>

#include "libcpu/icp.hh"
#include "libcpu/point-3d.hh"

Eigen::Matrix3Xf to_eigen(const libcpu::point_list& point_list)
{
    Eigen::Matrix3Xf result(3, point_list.size());
    for (size_t i = 0; i < point_list.size(); ++i)
    {
        result(0, i) = point_list[i].x;
        result(1, i) = point_list[i].y;
        result(2, i) = point_list[i].z;
    }

    return result;
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <path/to/data.txt> <path/to/data.txt>" << std::endl;
        return 1;
    }

    srand(static_cast<unsigned>(getpid()));

    auto q = libcpu::read_csv(argv[1], "Points_0", "Points_1", "Points_2");
    auto p = libcpu::read_csv(argv[2], "Points_0", "Points_1", "Points_2");

    auto q_e = to_eigen(q);
    std::cout << "Q:" << std::endl;
    std::cout << q_e << std::endl;
    auto p_e = to_eigen(p);
    std::cout << "P:" << std::endl;
    std::cout << p_e << std::endl;

    Eigen::Vector3f q_center = q_e.rowwise().mean();
    std::cout << "Q center:" << std::endl;
    std::cout << q_center << std::endl;
    Eigen::Vector3f p_center = p_e.rowwise().mean();
    std::cout << "P center:" << std::endl;
    std::cout << p_center << std::endl;

    auto q_centered = q_e.colwise() - q_center;
    std::cout << "Q centered:" << std::endl;
    std::cout << q_centered << std::endl;
    auto p_centered = p_e.colwise() - p_center;
    std::cout << "P centered:" << std::endl;
    std::cout << p_centered << std::endl;

    Eigen::Matrix3f covariance = p_centered * q_centered.transpose();
    std::cout << "Covariance:" << std::endl;
    std::cout << covariance << std::endl;

    Eigen::JacobiSVD svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

    auto rotation = svd.matrixU() * svd.matrixV().transpose();
    std::cout << "Rotation:" << std::endl;
    std::cout << rotation << std::endl;

    auto translation = q_center - rotation * p_center;
    std::cout << "Translation:" << std::endl;
    std::cout << translation << std::endl;

    auto new_p_e = (rotation * p_e).colwise() + translation;
    std::cout << "New P:" << std::endl;
    std::cout << new_p_e << std::endl;

    auto [transform, new_p] = libcpu::icp(q, p, 1);

    std::cout << "Transformation: " << std::endl;
    for (size_t i = 0; i < transform.lines; ++i)
    {
        for (size_t j = 0; j < transform.columns; ++j)
        {
            std::cout << transform.get(i, j) << ' ';
        }
        std::cout << std::endl;
    }

    std::cout << "Points" << std::endl;
    for (size_t i = 0; i < new_p.size() && i < 10; ++i)
    {
        std::cout << q[i] << " - " << new_p[i] << std::endl;
    }
}