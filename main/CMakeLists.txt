add_executable(main main/main.cc)
target_include_directories(main PRIVATE .)

target_link_libraries(main PRIVATE libcpu)