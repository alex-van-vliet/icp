add_executable(bench
  EXCLUDE_FROM_ALL
  bench/benchmark.cc
)
target_link_libraries(bench PRIVATE libcpu libgpu benchmark)
