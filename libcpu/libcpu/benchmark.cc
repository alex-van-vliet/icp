#include <benchmark/benchmark.h>

#include "icp.hh"

static void BM_CPU_cow2(benchmark::State& state)
{
    auto file1 = "../data/cow/cow_ref.txt";
    auto file2 = "../data/cow/cow_tr2.txt";

    auto q = libcpu::read_csv(file1, "Points_0", "Points_1", "Points_2");
    auto p = libcpu::read_csv(file2, "Points_0", "Points_1", "Points_2");

    // Perform setup here
    for (auto _ : state)
    {
        // This code gets timed
        auto [transform, new_p] = libcpu::icp(q, p, 200, 1e-5);
    }
}

static void BM_CPU_cow1(benchmark::State& state)
{
    auto file1 = "../data/cow/cow_ref.txt";
    auto file2 = "../data/cow/cow_tr1.txt";

    auto q = libcpu::read_csv(file1, "Points_0", "Points_1", "Points_2");
    auto p = libcpu::read_csv(file2, "Points_0", "Points_1", "Points_2");

    // Perform setup here
    for (auto _ : state)
    {
        // This code gets timed
        auto [transform, new_p] = libcpu::icp(q, p, 200, 1e-5);
    }
}

static void BM_CPU_horse1(benchmark::State& state)
{
    auto file1 = "../data/horse/horse_ref.txt";
    auto file2 = "../data/horse/horse_tr1.txt";

    auto q = libcpu::read_csv(file1, "Points_0", "Points_1", "Points_2");
    auto p = libcpu::read_csv(file2, "Points_0", "Points_1", "Points_2");

    // Perform setup here
    for (auto _ : state)
    {
        // This code gets timed
        auto [transform, new_p] = libcpu::icp(q, p, 200, 1e-5);
    }
}

static void BM_CPU_horse2(benchmark::State& state)
{
    auto file1 = "../data/horse/horse_ref.txt";
    auto file2 = "../data/horse/horse_tr2.txt";

    auto q = libcpu::read_csv(file1, "Points_0", "Points_1", "Points_2");
    auto p = libcpu::read_csv(file2, "Points_0", "Points_1", "Points_2");

    // Perform setup here
    for (auto _ : state)
    {
        // This code gets timed
        auto [transform, new_p] = libcpu::icp(q, p, 200, 1e-5);
    }
}

static void BM_CPU_line_translated(benchmark::State& state)
{
    auto file1 = "../data/line/line_ref.txt";
    auto file2 = "../data/line/line_translated_2_3_4.txt";

    auto q = libcpu::read_csv(file1, "Points_0", "Points_1", "Points_2");
    auto p = libcpu::read_csv(file2, "Points_0", "Points_1", "Points_2");

    // Perform setup here
    for (auto _ : state)
    {
        // This code gets timed
        auto [transform, new_p] = libcpu::icp(q, p, 200, 1e-5);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_CPU_line_translated)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_CPU_cow1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_CPU_cow2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_CPU_horse1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_CPU_horse2)->Unit(benchmark::kMillisecond)->UseRealTime();
// Run the benchmark
BENCHMARK_MAIN();
