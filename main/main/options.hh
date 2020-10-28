#pragma once

#include "Eigen/Core"

namespace options
{
    struct options
    {
        char* reference;
        char* transformed;
        bool gpu;
        uint capacity;
        uint iterations;
    };

    int parse_options(options& options, int argc, char* argv[]);
    void show_help(char* program);
} // namespace options