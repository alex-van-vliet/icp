cmake_minimum_required(VERSION 3.17)
project(gpgpu CXX)

include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(STATUS "No CUDA support")
endif()

include(FetchContent)

find_package(OpenMP)

FetchContent_Declare(
    GoogleTest
    URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
)

FetchContent_Declare(
    GoogleBenchmark
    URL https://github.com/google/benchmark/archive/v1.4.1.tar.gz
)

FetchContent_Declare(
    Eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
)

FetchContent_GetProperties(GoogleTest)
if (NOT GoogleTest_POPULATED)
    FetchContent_Populate(GoogleTest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif ()

FetchContent_GetProperties(Eigen3)
if (NOT Eigen3_POPULATED)
    FetchContent_Populate(Eigen3)
    add_subdirectory(${eigen3_SOURCE_DIR})
endif ()

enable_testing()

FetchContent_GetProperties(GoogleBenchmark)
if (NOT GoogleBenchmark_POPULATED)
    FetchContent_Populate(GoogleBenchmark)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
    add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif ()

add_subdirectory(libcpu)
add_subdirectory(libgpu)
add_subdirectory(main)
add_subdirectory(bench)