cmake_minimum_required(VERSION 3.17)
project(gpgpu CUDA CXX)

include(FetchContent)

FetchContent_Declare(
    GoogleTest
    URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
)

FetchContent_GetProperties(GoogleTest)
if (NOT GoogleTest_POPULATED)
    FetchContent_Populate(GoogleTest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif ()

add_subdirectory(libcpu)
add_subdirectory(main)
