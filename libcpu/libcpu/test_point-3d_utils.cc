#include "gtest/gtest.h"
#include "point-3d.hh"

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
        Point3D a{1, 1, 1};
        Point3D b{2, 2, 2};

        EXPECT_FLOAT_EQ(libcpu::squared_distance(a, b), 3.);
    }

    TEST(Points3DTest, closest)
    {
        Point3D a{1, 1, 1};
        Point3D b{2, 2, 2};
        Point3D c{3, 3, 3};
        Point3D d{4, 4, 4};

        point_list v{b, c, d};

        EXPECT_FLOAT_EQ(libcpu::closest(a, v), 0);
    }

    TEST(Points3DTest, closest_pairing)
    {
        Point3D a{1, 1, 1};
        Point3D b{2, 2, 2};
        Point3D c{3, 3, 3};
        Point3D d{4, 4, 4};

        point_list v{a, b, c, d};

        auto result = closest(v, v);

        ASSERT_EQ(result.size(), v.size());

        for (size_t i = 0; i < result.size(); i++)
            compare_points(result[i], v[i]);
    }

    TEST(Points3DTest, mean)
    {
        Point3D a{1, 1, 1};
        Point3D b{2, 2, 2};
        Point3D c{3, 3, 3};

        point_list v{a, b, c};

        compare_points(mean(v), b);
    }

    TEST(Points3DTest, subtract)
    {
        Point3D a{1, 1, 1};
        Point3D b{2, 2, 2};
        Point3D c{3, 3, 3};

        point_list v{b, c};
        point_list expected{a, b};

        auto result = subtract(v, a);

        ASSERT_EQ(result.size(), expected.size());

        for (size_t i = 0; i < result.size(); i++)
            compare_points(result[i], expected[i]);
    }

    TEST(Points3DTest, sum_of_squared_norms)
    {
        Point3D a{1, 1, 1};
        Point3D b{2, 2, 2};

        point_list v{a, b};

        EXPECT_FLOAT_EQ(15, sum_of_squared_norms(v));
    }

    TEST(Points3DTest, dot)
    {
        Point3D a{1, 2, 3};
        auto res = dot(utils::eye<float>(3, 3), a);

        EXPECT_FLOAT_EQ(res.x, 1);
        EXPECT_FLOAT_EQ(res.y, 2);
        EXPECT_FLOAT_EQ(res.z, 3);
    }

    TEST(Points3DTest, subtract_point)
    {
        Point3D a{1, 2, 3};
        Point3D b{3, 2, 1};
        auto res = subtract(a, b);

        EXPECT_FLOAT_EQ(res.x, -2);
        EXPECT_FLOAT_EQ(res.y, 0);
        EXPECT_FLOAT_EQ(res.z, 2);
    }

    TEST(Points3DTest, res)
    {
        Point3D a{1, 2, 3};
        auto res = scale(2, a);

        EXPECT_FLOAT_EQ(res.x, 2);
        EXPECT_FLOAT_EQ(res.y, 4);
        EXPECT_FLOAT_EQ(res.z, 6);
    }
} // namespace
