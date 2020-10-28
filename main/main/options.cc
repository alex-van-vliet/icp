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

    constexpr available_options available_options[] = {
        {
            .flag = "--device",
            .help = "cpu|gpu: to run on gpu [default: gpu]",
            .callback = handle_device,
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
                }
            }
            if (!found)
            {
                std::cerr << "Invalid argument: " << argv[i] << std::endl;
                return false;
            }
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