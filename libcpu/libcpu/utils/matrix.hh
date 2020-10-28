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
        using mat = std::vector<DATA>;

    public:
        /**
         * @brief Create a new matrix initialized with a given value.
         * @param i The number of rows.
         * @param j The number of columns.
         * @param default_val The default value.
         */
        Matrix(size_t i, size_t j, DATA default_val = 0)
            : rows(i)
            , cols(j)
        {
            values = mat(j * i, default_val);
        }

        /**
         * @brief Create a new matrix using values.
         * @param list The values.
         */
        Matrix(std::initializer_list<std::initializer_list<DATA>> list)
            : rows(list.size())
            , cols(std::begin(list)->size())
        {
            values.reserve(rows * cols);
            for (const auto& line : list)
            {
                assert(line.size() == cols);
                for (const auto& elt : line)
                    values.push_back(elt);
            }
        }

        Matrix(const Matrix<DATA>&) = default;

        /**
         * @brief Copy the matrix into an existing one.
         * @param other The matrix to copy from.
         * @return This.
         */
        Matrix& operator=(const Matrix<DATA>& other)
        {
            assert(other.cols == cols);
            assert(other.rows == rows);
            values = other.values;
            return *this;
        }

        Matrix(Matrix<DATA>&&) noexcept = default;

        /**
         * @brief Move the matrix into an existing one.
         * @param other The matrix to move from.
         * @return This.
         */
        Matrix& operator=(Matrix<DATA>&& other) noexcept
        {
            assert(other.cols == cols);
            assert(other.rows == rows);
            values = std::move(other.values);
            return *this;
        }

        ~Matrix() = default;

        /**
         * @brief Get an element.
         * @param i The row number.
         * @param j The column number.
         * @return The element.
         */
        DATA get(size_t i, size_t j) const;

        /**
         * @brief Set an element.
         * @param i The row number.
         * @param j The column number.
         * @param val The value.
         */
        void set(size_t i, size_t j, DATA val);

        /**
         * @brief Access to one element.
         * @param i The row number.
         * @param j The column number.
         * @return The element.
         */
        DATA& operator()(size_t i, size_t j);
        /**
         * @brief Access to one element.
         * @param i The row number.
         * @param j The column number.
         * @return The element.
         */
        DATA operator()(size_t i, size_t j) const;

        /**
         * @brief Normalize the matrix.
         */
        void normalize();

        /**
         * @brief Get a submatrix.
         * @param i_start The starting row (included).
         * @param i_end The ending row (not included).
         * @param j_start The starting column (included).
         * @param j_end The ending column (not included).
         * @return The submatrix.
         */
        Matrix<DATA> submatrix(size_t i_start, size_t i_end, size_t j_start,
                               size_t j_end);

        const size_t rows;
        const size_t cols;

        mat values;
    };

    template <typename DATA>
    auto Matrix<DATA>::get(size_t i, size_t j) const -> DATA
    {
        return (*this)(i, j);
    }

    template <typename DATA>
    void Matrix<DATA>::set(size_t i, size_t j, DATA val)
    {
        (*this)(i, j) = val;
    }

    template <typename DATA>
    DATA& Matrix<DATA>::operator()(size_t i, size_t j)
    {
        assert(i < rows);
        assert(j < cols);

        return values[i * cols + j];
    }

    template <typename DATA>
    DATA Matrix<DATA>::operator()(size_t i, size_t j) const
    {
        assert(i < rows);
        assert(j < cols);

        return values[i * cols + j];
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

        for (size_t i = 0; i < matrix.rows; ++i)
        {
            for (size_t j = 0; j < matrix.cols; ++j)
            {
                matrix.set(i, j, DATA(rand()) / RAND_MAX);
            }
        }

        return matrix;
    }

    template <typename DATA>
    bool operator==(const Matrix<DATA>& a, const Matrix<DATA>& b)
    {
        if (a.cols != b.cols || a.rows != b.rows)
            return false;
        for (size_t i = 0; i < a.rows; ++i)
        {
            for (size_t j = 0; j < a.cols; ++j)
            {
                if (a.get(i, j) != b.get(i, j))
                    return false;
            }
        }
        return true;
    }

    template <typename DATA>
    auto dot(const Matrix<DATA>& a, const Matrix<DATA>& b) -> Matrix<DATA>
    {
        assert(a.cols == b.rows);

        Matrix<DATA> result(a.rows, b.cols);

        for (size_t i = 0; i < result.rows; ++i)
        {
            for (size_t j = 0; j < result.cols; ++j)
            {
                DATA val = 0;
                for (size_t k = 0; k < a.cols; ++k)
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
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                float val = get(i, j);
                norm += val * val;
            }
        }
        norm = sqrt(norm);

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                set(i, j, get(i, j) / norm);
            }
        }
    }

    template <typename DATA>
    auto Matrix<DATA>::submatrix(size_t i_start, size_t i_end, size_t j_start,
                                 size_t j_end) -> Matrix<DATA>
    {
        Matrix<DATA> res(i_end - i_start, j_end - j_start);

        for (size_t i = 0; i < res.rows; ++i)
            for (size_t j = 0; j < res.cols; ++j)
                res.set(i, j, get(i_start + i, j_start + j));

        return res;
    }
    template <typename DATA>
    auto scale(float a, const Matrix<DATA>& b) -> Matrix<DATA>
    {
        Matrix<DATA> res(b.rows, b.cols);

        for (size_t i = 0; i < res.rows; ++i)
            for (size_t j = 0; j < res.cols; ++j)
                res.set(i, j, a * b.get(i, j));

        return res;
    }
    template <typename DATA>
    void operator+=(Matrix<DATA>& a, const Matrix<DATA>& b)
    {
        assert(a.rows == b.rows);
        assert(b.cols == a.cols);

        for (size_t i = 0; i < a.rows; ++i)
        {
            for (size_t j = 0; j < a.cols; ++j)
            {
                a.set(i, j, a.get(i, j) + b.get(i, j));
            }
        }
    }
} // namespace utils
