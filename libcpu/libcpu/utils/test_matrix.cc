#include "matrix.hh"

#include "gtest/gtest.h"

namespace
{
    TEST(MatrixTest, constructor)
    {
        auto matrix = utils::Matrix<int>(1, 1);

        EXPECT_EQ(0, matrix.get(0, 0));
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
}
