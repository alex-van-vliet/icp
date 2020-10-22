add_executable(bench
  bench/benchmark.cc
)
target_link_libraries(bench PRIVATE libcpu libgpu benchmark)
