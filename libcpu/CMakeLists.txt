add_library(libcpu libcpu/point-3d.cc libcpu/point-3d.hh)
target_include_directories(libcpu PUBLIC .)

# add_executable(libcpu_tests SOURCES)
# target_link_libraries(libcpu_tests PRIVATE libcpu gtest_main)