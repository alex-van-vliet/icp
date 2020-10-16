#include "gtest/gtest.h"
#include "icp.hh"
#include "point-3d.hh"


namespace
{
    using namespace libcpu;

    TEST(FunctionalTest, basic_translation)
    {
        auto file1 = "../data/line/line_ref.txt";
        auto file2 = "../data/line/line_translated_2_3_4.txt";

        auto m = libcpu::read_csv(file1, "Points_0", "Points_1", "Points_2");
        auto p = libcpu::read_csv(file2, "Points_0", "Points_1", "Points_2");

        auto transform = libcpu::find_alignment(p, m);

        auto result = utils::eye<float>(4);
        result.set(0, 3, 2);
        result.set(1, 3, 3);
        result.set(2, 3, 4);

        ASSERT_EQ(result == transform, true);
    }


} //namespace