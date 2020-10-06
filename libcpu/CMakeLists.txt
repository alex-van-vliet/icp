add_library(libcpu
  libcpu/point-3d.cc libcpu/point-3d.hh
  libcpu/utils/matrix.cc libcpu/utils/matrix.hh
)
target_include_directories(libcpu PUBLIC .)
target_include_directories(libcpu PUBLIC libcpu/utils)

add_executable(libcpu_tests
  libcpu/utils/test_matrix.cc
  libcpu/test_point-3d_utils.cc
)
target_link_libraries(libcpu_tests PRIVATE libcpu gtest_main)

add_test(libcpu_tests libcpu_tests)
