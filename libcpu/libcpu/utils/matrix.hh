#pragma once

#include <vector>

#include <assert.h>

namespace utils
{
    template<typename DATA>
    class Matrix
    {
        using mat = std::vector<std::vector<DATA>>;

        public:
            /*
            ** i: lines
            ** j: columns
            */
            Matrix(size_t i, size_t j, DATA default_val = 0)
                : lines(i),
                  columns(j)
            {
                values = std::vector<std::vector<DATA>>(i, std::vector<DATA>(j, default_val));
            }

            DATA get(size_t i, size_t j);

            void set(size_t i, size_t j, DATA val);

            const size_t lines;
            const size_t columns;

        private:
            mat values;
    };

    template<typename DATA>
    auto Matrix<DATA>::get(size_t i, size_t j) -> DATA
    {
        assert(i < lines);
        assert(j < columns);

        return values[i][j];
    }

    template<typename DATA>
    void Matrix<DATA>::set(size_t i, size_t j, DATA val)
    {
        assert(i < lines);
        assert(j < columns);

        values[i][j] = val;
    }

    template<typename DATA>
    Matrix<DATA> eye(size_t n, size_t m)
    {
        auto matrix = Matrix<DATA>(n, m);

        auto min = std::min(n, m);
        for (size_t i = 0; i < min; i++)
        {
            matrix.set(i, i, 1);
        }

        return matrix;
    }

    template<typename DATA>
    Matrix<DATA> eye(size_t n)
    {
        return eye<DATA>(n, n);
    }
}
