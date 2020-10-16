#include <benchmark/benchmark.h>
#include "icp.hh"

static void BM_CPU_cow2(benchmark::State& state)
{
    auto file1 = "../data/cow/cow_ref.txt";
    auto file2 = "../data/cow/cow_tr2.txt";

    auto q = libcpu::read_csv(file1, "Points_0", "Points_1", "Points_2");
    auto p = libcpu::read_csv(file2, "Points_0", "Points_1", "Points_2");
    
    // Perform setup here
    for (auto _ : state) {
        // This code gets timed
        auto [transform, new_p] = libcpu::icp(q, p, 200, 1e-5);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_CPU_cow2)->Unit(benchmark::kMillisecond)->UseRealTime();
// Run the benchmark
BENCHMARK_MAIN();


