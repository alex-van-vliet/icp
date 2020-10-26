#include <benchmark/benchmark.h>

#include "libcpu/icp.hh"
#include "libgpu/icp.hh"

struct test
{
    const char* ref;
    const char* transformed;
};

constexpr test files[] = {
    {"../data/line/line_ref.txt", "../data/line/line_translated_2_3_4.txt"},
    {"../data/cow/cow_ref.txt", "../data/cow/cow_tr1.txt"},
    {"../data/cow/cow_ref.txt", "../data/cow/cow_tr2.txt"},
    {"../data/horse/horse_ref.txt", "../data/horse/horse_tr1.txt"},
    {"../data/horse/horse_ref.txt", "../data/horse/horse_tr2.txt"},
};
constexpr uint nb_files = sizeof(files) / sizeof(files[0]);

static void CustomArguments(benchmark::internal::Benchmark* b)
{
    for (uint vp_threshold = 8; vp_threshold <= 1024; vp_threshold *= 2)
        for (uint i = 0; i < nb_files; ++i)
            b->Args({vp_threshold, i});
}

std::string get_name(const std::string& file)
{
    size_t pos_slash = file.rfind("/") + 1;
    size_t pos_dot = file.rfind(".");
    return file.substr(pos_slash, pos_dot - pos_slash);
}

std::string get_label(const std::string& ref, const std::string& transformed)
{
    return get_name(transformed) + " -> " + get_name(ref);
}

static void BM_CPU(benchmark::State& state)
{
    uint vp_threshold = state.range(0);
    uint file_i = state.range(1);

    std::string ref = files[file_i].ref;
    std::string transformed = files[file_i].transformed;

    state.SetLabel(get_label(ref, transformed));

    auto q = libcpu::read_csv(ref, "Points_0", "Points_1", "Points_2");
    auto p = libcpu::read_csv(transformed, "Points_0", "Points_1", "Points_2");

    // Perform setup here
    for (auto _ : state)
    {
        // This code gets timed
        benchmark::DoNotOptimize(libcpu::icp(q, p, 200, 1e-5, vp_threshold));
    }
}

static void BM_GPU(benchmark::State& state)
{
    uint vp_threshold = state.range(0);
    uint file_i = state.range(1);

    std::string ref = files[file_i].ref;
    std::string transformed = files[file_i].transformed;

    state.SetLabel(get_label(ref, transformed));

    auto q = libcpu::read_csv(ref, "Points_0", "Points_1", "Points_2");
    auto p = libcpu::read_csv(transformed, "Points_0", "Points_1", "Points_2");

    // Perform setup here
    for (auto _ : state)
    {
        // This code gets timed
        benchmark::DoNotOptimize(libgpu::icp(q, p, 200, 1e-5, vp_threshold));
    }
}

// Register the function as a benchmark
BENCHMARK(BM_CPU)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Apply(CustomArguments);
BENCHMARK(BM_GPU)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Apply(CustomArguments);
// Run the benchmark
BENCHMARK_MAIN();
