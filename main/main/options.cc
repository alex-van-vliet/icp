#include "options.hh"

#include <iostream>

namespace
{
    using option_callback = int (*)(options::options& options, int argc,
                                    char* argv[]);
    struct available_options
    {
        const char* flag;
        const char* help;
        option_callback callback;
    };

    int handle_device(options::options& options, int argc, char* argv[])
    {
        if (argc <= 1)
        {
            std::cerr << "Missing device." << std::endl;
            return 0;
        }
        if (strcmp(argv[1], "gpu") == 0)
            options.gpu = true;
        else if (strcmp(argv[1], "cpu") == 0)
            options.gpu = false;
        else
        {
            std::cerr << "Invalid device: " << argv[1] << std::endl;
            return 0;
        }
        return 2;
    }

    int handle_capacity(options::options& options, int argc, char* argv[])
    {
        if (argc <= 1)
        {
            std::cerr << "Missing capacity." << std::endl;
            return 0;
        }
        try
        {
            int capacity = std::stoi(argv[1]);
            if (capacity >= 0)
                options.capacity = capacity;
            else
            {
                std::cerr << "Invalid capacity: " << capacity << std::endl;
                return false;
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Invalid capacity: " << argv[1] << std::endl;
            return false;
        }
        return 2;
    }

    int handle_iterations(options::options& options, int argc, char* argv[])
    {
        if (argc <= 1)
        {
            std::cerr << "Missing iterations." << std::endl;
            return 0;
        }
        try
        {
            int iterations = std::stoi(argv[1]);
            if (iterations >= 0)
                options.iterations = iterations;
            else
            {
                std::cerr << "Invalid iterations: " << iterations << std::endl;
                return false;
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Invalid iterations: " << argv[1] << std::endl;
            return false;
        }
        return 2;
    }

    int handle_error(options::options& options, int argc, char* argv[])
    {
        if (argc <= 1)
        {
            std::cerr << "Missing error." << std::endl;
            return 0;
        }
        try
        {
            float error = std::stof(argv[1]);
            if (error >= 0)
                options.error = error;
            else
            {
                std::cerr << "Invalid error: " << error << std::endl;
                return false;
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Invalid error: " << argv[1] << std::endl;
            return false;
        }
        return 2;
    }

    constexpr available_options available_options[] = {
        {
            .flag = "--device",
            .help = "cpu|gpu: the device to run on [default: gpu]",
            .callback = handle_device,
        },
        {
            .flag = "--capacity",
            .help = "<capacity>: the vp tree capacity [default: 32 on cpu, 256 "
                    "on gpu]",
            .callback = handle_capacity,
        },
        {
            .flag = "--iterations",
            .help =
                "<iterations>: the maximum number of iterations [default: 200]",
            .callback = handle_iterations,
        },
        {
            .flag = "--error",
            .help = "<error>: the error threshold [default: 1e-5]",
            .callback = handle_error,
        },
    };

} // namespace

namespace options
{
    int parse_options(options& options, int argc, char* argv[])
    {
        options.reference = argv[0];
        options.transformed = argv[1];
        options.gpu = true;
        options.capacity = 0;
        options.iterations = 200;
        options.error = 1e-5;
        if (argc < 2)
            return false;

        for (int i = 2; i < argc;)
        {
            bool found = false;
            for (const auto& option : available_options)
            {
                if (strcmp(argv[i], option.flag) == 0)
                {
                    found = true;
                    int res = option.callback(options, argc - i, argv + i);
                    if (res == 0)
                        return false;
                    i += res;
                    if (i >= argc)
                        break;
                }
            }
            if (!found)
            {
                std::cerr << "Invalid argument: " << argv[i] << std::endl;
                return false;
            }
        }
        if (options.capacity == 0)
        {
            options.capacity = options.gpu ? 256 : 32;
        }
        return true;
    }

    void show_help(char* program)
    {
        std::cerr
            << "Usage: " << program
            << " <path/to/reference.txt> <path/to/transformed.txt> [options]"
            << std::endl;
        std::cerr << "Options:" << std::endl;
        for (const auto& option : available_options)
            std::cerr << "    " << option.flag << ' ' << option.help
                      << std::endl;
    }
} // namespace options