add_library(libcpu
  libcpu/point-3d.cc libcpu/point-3d.hh
  libcpu/icp.cc libcpu/icp.hh
  libcpu/utils/matrix.cc libcpu/utils/matrix.hh
)
target_compile_features(libcpu PUBLIC cxx_std_17)
target_include_directories(libcpu PUBLIC .)
target_include_directories(libcpu PUBLIC libcpu/utils)
target_link_libraries(libcpu eigen)
target_link_libraries(libcpu OpenMP::OpenMP_CXX)

add_executable(libcpu_tests
  libcpu/utils/test_matrix.cc
  libcpu/test_point-3d_utils.cc
  libcpu/test_functionnal.cc
)
target_compile_features(libcpu_tests PUBLIC cxx_std_17)
target_link_libraries(libcpu_tests PRIVATE libcpu gtest_main)

add_test(libcpu_tests libcpu_tests)
