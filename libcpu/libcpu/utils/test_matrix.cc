#include "gtest/gtest.h"
#include "matrix.hh"

namespace
{
    TEST(MatrixTest, constructor)
    {
        auto matrix = utils::Matrix<int>(1, 1);

        EXPECT_EQ(0, matrix.get(0, 0));
    }

    TEST(MatrixTest, properties)
    {
        auto matrix = utils::Matrix<int>(2, 3);

        EXPECT_EQ(2, matrix.lines);
        EXPECT_EQ(3, matrix.columns);
    }

    TEST(MatrixTest, set_small)
    {
        auto matrix = utils::Matrix<int>(1, 1);

        matrix.set(0, 0, 3);

        EXPECT_EQ(3, matrix.get(0, 0));
    }

    TEST(MatrixTest, eye_small)
    {
        auto matrix = utils::eye<int>(3);

        EXPECT_EQ(1, matrix.get(1, 1));
        EXPECT_EQ(0, matrix.get(1, 0));
    }

    TEST(MatrixTest, random)
    {
        auto matrix = utils::random<float>(1, 2);

        EXPECT_EQ(1, matrix.lines);
        EXPECT_EQ(2, matrix.columns);
    }
} // namespace
