#pragma once

#include <assert.h>
#include <vector>

namespace utils
{
    template <typename DATA>
    class Matrix
    {
        using mat = std::vector<std::vector<DATA>>;

    public:
        /*
        ** i: lines
        ** j: columns
        */
        Matrix(size_t i, size_t j, DATA default_val = 0)
            : lines(i)
            , columns(j)
        {
            values = std::vector<std::vector<DATA>>(
                i, std::vector<DATA>(j, default_val));
        }

        Matrix(std::initializer_list<std::initializer_list<DATA>> list)
            : lines(list.size())
            , columns(std::begin(list)->size())
        {
            values.reserve(lines);
            for (const auto& line : list)
            {
                assert(line.size() == columns);
                values.push_back(line);
            }
        }

        Matrix(const Matrix<DATA>&) = default;

        Matrix& operator=(const Matrix<DATA>& other)
        {
            assert(other.columns == columns);
            assert(other.lines == lines);
            values = other.values;
            return *this;
        }

        Matrix(Matrix<DATA>&&) = default;

        Matrix& operator=(Matrix<DATA>&& other)
        {
            assert(other.columns == columns);
            assert(other.lines == lines);
            values = std::move(other.values);
            return *this;
        }

        ~Matrix() = default;

        DATA get(size_t i, size_t j) const;

        void set(size_t i, size_t j, DATA val);


        const size_t lines;
        const size_t columns;

    private:
        mat values;
    };

    template <typename DATA>
    auto Matrix<DATA>::get(size_t i, size_t j) const -> DATA
    {
        assert(i < lines);
        assert(j < columns);

        return values[i][j];
    }

    template <typename DATA>
    void Matrix<DATA>::set(size_t i, size_t j, DATA val)
    {
        assert(i < lines);
        assert(j < columns);

        values[i][j] = val;
    }

    template <typename DATA>
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

    template <typename DATA>
    Matrix<DATA> eye(size_t n)
    {
        return eye<DATA>(n, n);
    }
} // namespace utils
