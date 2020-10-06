#include "point-3d.hh"

#include "gtest/gtest.h"

namespace
{
    using namespace libcpu;

    void compare_points(const Point3D& a, const Point3D& b)
    {
        EXPECT_FLOAT_EQ(a.x, b.x);
        EXPECT_FLOAT_EQ(a.y, b.y);
        EXPECT_FLOAT_EQ(a.z, b.z);
    }

    TEST(Points3DTest, squared_dist)
    {
        Point3D a {1, 1, 1};
        Point3D b {2, 2, 2};

        EXPECT_FLOAT_EQ(libcpu::squared_distance(a, b), 3.);
    }

    TEST(Points3DTest, closest)
    {
        Point3D a {1, 1, 1};
        Point3D b {2, 2, 2};
        Point3D c {3, 3, 3};
        Point3D d {4, 4, 4};

        std::vector<Point3D> v{b, c, d};

        EXPECT_FLOAT_EQ(libcpu::closest(a, v), 0);
    }

    TEST(Points3DTest, closest_pairing)
    {
        Point3D a {1, 1, 1};
        Point3D b {2, 2, 2};
        Point3D c {3, 3, 3};
        Point3D d {4, 4, 4};

        std::vector<Point3D> v{a, b, c, d};

        auto result = closest(v, v);

        ASSERT_EQ(result.size(), v.size());

        for (size_t i = 0; i < result.size(); i++)
            compare_points(result[i], v[i]);
    }

    TEST(Points3DTest, mean)
    {
        Point3D a {1, 1, 1};
        Point3D b {2, 2, 2};
        Point3D c {3, 3, 3};

        std::vector<Point3D> v{a, b, c};

        compare_points(mean(v), b);
    }
}
