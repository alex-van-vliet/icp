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

    TEST(MatrixTest, dot_right)
    {
        auto matrix = utils::random<float>(3, 4);

        auto res = utils::dot(matrix, utils::eye<float>(4));
        ASSERT_EQ(matrix.lines, res.lines);
        ASSERT_EQ(matrix.columns, res.columns);
        for (size_t i = 0; i < matrix.lines; ++i)
            for (size_t j = 0; j < matrix.columns; ++j)
                EXPECT_FLOAT_EQ(matrix.get(i, j), res.get(i, j));
    }

    TEST(MatrixTest, dot_left)
    {
        auto matrix = utils::random<float>(3, 4);

        auto res = utils::dot(utils::eye<float>(3), matrix);
        ASSERT_EQ(matrix.lines, res.lines);
        ASSERT_EQ(matrix.columns, res.columns);
        for (size_t i = 0; i < matrix.lines; ++i)
            for (size_t j = 0; j < matrix.columns; ++j)
                EXPECT_FLOAT_EQ(matrix.get(i, j), res.get(i, j));
    }
} // namespace
