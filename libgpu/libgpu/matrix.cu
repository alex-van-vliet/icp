#include "cuda/memory.hh"
#include "matrix.hh"

namespace libgpu
{
    GPUMatrix::GPUMatrix(size_t rows, size_t cols)
        : ptr{cuda::mallocManagedRaw<float>(rows * cols)}
        , rows{rows}
        , cols{cols}
    {}

    GPUMatrix::~GPUMatrix()
    {
        cuda::free(ptr);
    }

    GPUMatrix::GPUMatrix(GPUMatrix&& other) noexcept
        : ptr{std::exchange(other.ptr, nullptr)}
        , rows{other.rows}
        , cols{other.cols}
    {}

    GPUMatrix& GPUMatrix::operator=(GPUMatrix&& other) noexcept
    {
        assert(cols == other.cols);
        assert(rows == other.rows);

        ptr = std::exchange(other.ptr, nullptr);
        return *this;
    }

    GPUMatrix GPUMatrix::zero(size_t rows, size_t cols)
    {
        GPUMatrix res(rows, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                res(i, j) = 0;
        return res;
    }

    GPUMatrix GPUMatrix::eye(size_t n)
    {
        GPUMatrix res(n, n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                res(i, j) = i == j ? 1 : 0;
        return res;
    }

    GPUMatrix GPUMatrix::from_point_list(const libcpu::point_list& p)
    {
        auto matrix = GPUMatrix(p.size(), 3);
        for (size_t i = 0; i < p.size(); ++i)
        {
            matrix(i, 0) = p[i].x;
            matrix(i, 1) = p[i].y;
            matrix(i, 2) = p[i].z;
        }
        return matrix;
    }

    libcpu::point_list GPUMatrix::to_point_list() const
    {
        assert(cols == 3);
        libcpu::point_list list;
        list.reserve(rows);
        for (size_t i = 0; i < rows; ++i)
        {
            list.push_back(libcpu::Point3D{
                (*this)(i, 0),
                (*this)(i, 1),
                (*this)(i, 2),
            });
        }
        return list;
    }

    GPUMatrix GPUMatrix::mean() const
    {
        auto mean = GPUMatrix::zero(1, 3);
        for (size_t i = 0; i < rows; ++i)
        {
            mean(0, 0) += (*this)(i, 0) / rows;
            mean(0, 1) += (*this)(i, 1) / rows;
            mean(0, 2) += (*this)(i, 2) / rows;
        }
        return mean;
    }

    GPUMatrix GPUMatrix::subtract_rowwise(const GPUMatrix& matrix) const
    {
        assert(matrix.rows == 1);
        assert(matrix.cols == cols);

        auto res = GPUMatrix(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
                res(i, j) = (*this)(i, j) - matrix(0, j);
        }
        return res;
    }

    GPUMatrix GPUMatrix::subtract(const GPUMatrix& matrix) const
    {
        assert(matrix.rows == rows);
        assert(matrix.cols == cols);

        auto res = GPUMatrix(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
                res(i, j) = (*this)(i, j) - matrix(i, j);
        }
        return res;
    }

    GPUMatrix GPUMatrix::dot(const GPUMatrix& matrix) const
    {
        assert(cols == matrix.rows);

        GPUMatrix res(rows, matrix.cols);

        for (size_t i = 0; i < res.rows; ++i)
        {
            for (size_t j = 0; j < res.cols; ++j)
            {
                res(i, j) = 0;
                for (size_t k = 0; k < cols; ++k)
                    res(i, j) += (*this)(i, k) * matrix(k, j);
            }
        }
        return res;
    }

    GPUMatrix GPUMatrix::closest(const GPUMatrix& matrix) const
    {
        assert(matrix.cols == cols);
        assert(matrix.rows > 0);

        auto res = GPUMatrix(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            size_t closest_i = 0;
            float closest_dist = distance(*this, i, matrix, closest_i);
            for (size_t j = 0; j < matrix.rows; ++j)
            {
                float dist = distance(*this, i, matrix, j);
                if (dist < closest_dist)
                {
                    closest_i = j;
                    closest_dist = dist;
                }
            }

            for (size_t j = 0; j < cols; ++j)
            {
                res(i, j) = matrix(closest_i, j);
            }
        }

        return res;
    }

    GPUMatrix GPUMatrix::find_covariance(const GPUMatrix& a, const GPUMatrix& b)
    {
        assert(a.cols == b.cols);
        assert(a.rows == b.rows);

        auto res = GPUMatrix::zero(a.cols, b.cols);

        for (size_t i = 0; i < a.rows; ++i)
            for (size_t j = 0; j < a.cols; ++j)
                for (size_t k = 0; k < b.cols; ++k)
                    res(j, k) += a(i, j) * b(i, k);

        return res;
    }

    float GPUMatrix::distance(const GPUMatrix& a, size_t a_i,
                              const GPUMatrix& b, size_t b_i)
    {
        assert(a.cols == b.cols);

        float dist = 0;
        for (size_t j = 0; j < a.cols; ++j)
        {
            float diff = a(a_i, j) - b(b_i, j);
            dist += diff * diff;
        }
        return dist;
    }

    GPUMatrix GPUMatrix::transpose() const
    {
        GPUMatrix res(cols, rows);

        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                res(j, i) = (*this)(i, j);

        return res;
    }
} // namespace libgpu