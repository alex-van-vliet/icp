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

add_subdirectory(libcpu)
add_subdirectory(main)
