add_executable(main main/main.cc)
target_compile_features(main PUBLIC cxx_std_17)
target_include_directories(main PRIVATE .)

target_link_libraries(main PRIVATE libcpu libgpu eigen)