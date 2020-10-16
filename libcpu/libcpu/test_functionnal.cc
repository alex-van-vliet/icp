#include "gtest/gtest.h"
#include "icp.hh"
#include "point-3d.hh"


namespace
{
    using namespace libcpu;

    void wrapper(const char* file1, const char* file2)
    {
        auto q = libcpu::read_csv(file1, "Points_0", "Points_1", "Points_2");
        auto p = libcpu::read_csv(file2, "Points_0", "Points_1", "Points_2");

        auto [transform, new_p] = libcpu::icp(q, p, 200, 1e-5);

        ASSERT_EQ(q == new_p, true);
    }

    TEST(FunctionalTest, basic_translation)
    {
        auto file1 = "../data/line/line_ref.txt";
        auto file2 = "../data/line/line_translated_2_3_4.txt";

        wrapper(file1, file2);
    }

    TEST(FunctionalTest, basic_rotation)
    {
        auto file1 = "../data/line/line_ref.txt";
        auto file2 = "../data/line/line_rotated_90_135_180.txt";

        wrapper(file1, file2);
    }

    TEST(FunctionalTest, basic_scale)
    {
        auto file1 = "../data/line/line_ref.txt";
        auto file2 = "../data/line/line_scaled_2_3_4.txt";

        wrapper(file1, file2);
    }


} //namespace