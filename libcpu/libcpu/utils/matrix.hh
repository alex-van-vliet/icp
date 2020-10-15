#pragma once

#include <assert.h>
#include <math.h>
#include <stdlib.h>
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

        template <int nb_iter>
        Matrix<DATA> largest_eigenvector();

        void normalize();

        Matrix<DATA> submatrix(size_t i_start, size_t i_end, size_t j_start,
                               size_t j_end);

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

    template <typename DATA>
    Matrix<DATA> random(size_t n, size_t m)
    {
        auto matrix = Matrix<DATA>(n, m);

        for (size_t i = 0; i < matrix.lines; ++i)
        {
            for (size_t j = 0; j < matrix.columns; ++j)
            {
                matrix.set(i, j, DATA(rand()) / RAND_MAX);
            }
        }

        return matrix;
    }

    template <typename DATA>
    auto dot(const Matrix<DATA>& a, const Matrix<DATA>& b) -> Matrix<DATA>
    {
        assert(a.columns == b.lines);

        Matrix<DATA> result(a.lines, b.columns);

        for (size_t i = 0; i < result.lines; ++i)
        {
            for (size_t j = 0; j < result.columns; ++j)
            {
                DATA val = 0;
                for (size_t k = 0; k < a.columns; ++k)
                    val += a.get(i, k) * b.get(k, j);
                result.set(i, j, val);
            }
        }

        return result;
    }

    template <typename DATA>
    void Matrix<DATA>::normalize()
    {
        DATA norm = 0;
        for (size_t i = 0; i < lines; ++i)
        {
            for (size_t j = 0; j < columns; ++j)
            {
                float val = get(i, j);
                norm += val * val;
            }
        }
        norm = sqrt(norm);

        for (size_t i = 0; i < lines; ++i)
        {
            for (size_t j = 0; j < columns; ++j)
            {
                set(i, j, get(i, j) / norm);
            }
        }
    }

    template <typename DATA>
    template <int nb_iter>
    auto Matrix<DATA>::largest_eigenvector() -> Matrix<DATA>
    {
        assert(lines == columns);
        auto b = eye<DATA>(lines, 1);

        // TODO: check update
        for (int i = 0; i < nb_iter; ++i)
        {
            b = dot(*this, b);
            b.normalize();
        }

        return b;
    }

    template <typename DATA>
    auto Matrix<DATA>::submatrix(size_t i_start, size_t i_end, size_t j_start,
                                 size_t j_end) -> Matrix<DATA>
    {
        Matrix<DATA> res(i_end - i_start, j_end - j_start);

        for (size_t i = 0; i < res.lines; ++i)
            for (size_t j = 0; j < res.columns; ++j)
                res.set(i, j, get(i_start + i, j_start + j));

        return res;
    }
} // namespace utils
