if(CMAKE_CUDA_COMPILER)
    add_library(libgpu
        libgpu/icp.cu libgpu/icp.hh
        libgpu/cuda/memory.cu libgpu/cuda/memory.hh
        libgpu/matrix.cu libgpu/matrix.hh
        libgpu/svd.cc libgpu/svd.hh
        libgpu/vp-tree.cu libgpu/vp-tree.hh
        )
    target_compile_features(libgpu PUBLIC cxx_std_17)
    target_include_directories(libgpu PUBLIC .)
    target_link_libraries(libgpu PRIVATE libcpu eigen)
endif()
