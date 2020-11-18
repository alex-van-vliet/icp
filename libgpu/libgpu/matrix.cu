#include "cuda/memory.hh"
#include "matrix.hh"

namespace libgpu
{
    GPUMatrix::GPUMatrix(size_t rows, size_t cols)
        : ptr{cuda::mallocRaw<float>(rows * cols)}
        , rows{rows}
        , cols{cols}
        , should_delete{true}
    {}

    GPUMatrix::~GPUMatrix()
    {
        if (should_delete)
            cuda::free(ptr);
    }

    GPUMatrix::GPUMatrix(const GPUMatrix& other)
        : ptr{other.ptr}
        , rows{other.rows}
        , cols{other.cols}
        , should_delete{false}
    {}

    GPUMatrix::GPUMatrix(GPUMatrix&& other) noexcept
        : ptr{std::exchange(other.ptr, nullptr)}
        , rows{other.rows}
        , cols{other.cols}
        , should_delete{other.should_delete}
    {}

    GPUMatrix& GPUMatrix::operator=(GPUMatrix&& other) noexcept
    {
        assert(cols == other.cols);
        assert(rows == other.rows);

        ptr = std::exchange(other.ptr, nullptr);
        should_delete = other.should_delete;
        return *this;
    }

    __global__ void zero_kernel(GPUMatrix matrix)
    {
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= matrix.rows)
            return;

        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (j >= matrix.cols)
            return;

        matrix(i, j) = 0;
    }

    GPUMatrix GPUMatrix::zero(size_t rows, size_t cols)
    {
        GPUMatrix res(rows, cols);

        dim3 blockdim(32, 32);
        dim3 griddim((rows + blockdim.x - 1) / blockdim.x,
                     (cols + blockdim.y - 1) / blockdim.y);
        zero_kernel<<<griddim, blockdim>>>(res);

        return res;
    }

    __global__ void eye_kernel(GPUMatrix matrix)
    {
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= matrix.rows)
            return;

        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (j >= matrix.cols)
            return;

        matrix(i, j) = i == j ? 1 : 0;
    }

    GPUMatrix GPUMatrix::eye(size_t n)
    {
        GPUMatrix res(n, n);

        dim3 blockdim(32, 32);
        dim3 griddim((n + blockdim.x - 1) / blockdim.x,
                     (n + blockdim.y - 1) / blockdim.y);
        eye_kernel<<<griddim, blockdim>>>(res);

        return res;
    }

    GPUMatrix GPUMatrix::from_point_list(const libcpu::point_list& p)
    {
        std::unique_ptr<float[]> array(new float[p.size() * 3]);

        for (size_t i = 0; i < p.size(); ++i)
        {
            array[i + 0 * p.size()] = p[i].x;
            array[i + 1 * p.size()] = p[i].y;
            array[i + 2 * p.size()] = p[i].z;
        }

        auto matrix = GPUMatrix(p.size(), 3);

        cudaMemcpy(matrix.ptr, array.get(), p.size() * 3 * sizeof(float),
                   cudaMemcpyHostToDevice);

        return matrix;
    }

    libcpu::point_list GPUMatrix::to_point_list() const
    {
        assert(cols == 3);
        std::unique_ptr<float[]> array(new float[rows * 3]);

        cudaMemcpy(array.get(), ptr, rows * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost);

        libcpu::point_list list;
        list.reserve(rows);

        for (size_t i = 0; i < rows; ++i)
        {
            list[i].x = array[i + 0 * rows];
            list[i].y = array[i + 1 * rows];
            list[i].z = array[i + 2 * rows];
        }

        return list;
    }

    GPUMatrix GPUMatrix::from_cpu(const utils::Matrix<float>& cpu)
    {
        std::unique_ptr<float[]> array(
            new float[cpu.rows * cpu.cols * sizeof(float)]);

        for (size_t i = 0; i < cpu.rows; ++i)
            for (size_t j = 0; j < cpu.cols; ++j)
                array[i + j * cpu.rows] = cpu(i, j);

        GPUMatrix res(cpu.rows, cpu.cols);

        cudaMemcpy(res.ptr, array.get(), cpu.cols * cpu.rows * sizeof(float),
                   cudaMemcpyHostToDevice);

        return res;
    }

    utils::Matrix<float> GPUMatrix::to_cpu() const
    {
        std::unique_ptr<float[]> array(new float[cols * rows * sizeof(float)]);

        cudaMemcpy(array.get(), ptr, cols * rows * sizeof(float),
                   cudaMemcpyDeviceToHost);

        utils::Matrix<float> res(rows, cols);

        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                res(i, j) = array[i + j * rows];

        return res;
    }

    __global__ void divide_kernel(GPUMatrix matrix, GPUMatrix divided)
    {
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= matrix.rows)
            return;

        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (j >= matrix.cols)
            return;

        divided(i, j) = matrix(i, j) / matrix.rows;
    }

    __global__ void sum_kernel(GPUMatrix matrix, GPUMatrix mean)
    {
        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (j >= matrix.cols)
            return;

        for (size_t i = 0; i < matrix.rows; ++i)
            mean(0, j) += matrix(i, j);
    }

    GPUMatrix GPUMatrix::mean() const
    {
        auto divided = GPUMatrix(rows, cols);

        dim3 blockdim_divide(32, 32);
        dim3 griddim_divide((rows + blockdim_divide.x - 1) / blockdim_divide.x,
                            (cols + blockdim_divide.y - 1) / blockdim_divide.y);
        divide_kernel<<<griddim_divide, blockdim_divide>>>(*this, divided);

        auto mean = GPUMatrix::zero(1, cols);

        dim3 blockdim_sum(1, 32);
        dim3 griddim_sum(1, (cols + blockdim_sum.y - 1) / blockdim_sum.y);
        sum_kernel<<<griddim_sum, blockdim_sum>>>(divided, mean);

        return mean;
    }

    __global__ void subtract_rowwise_kernel(GPUMatrix a, GPUMatrix b,
                                            GPUMatrix res)
    {
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= a.rows)
            return;

        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (j >= a.cols)
            return;

        res(i, j) = a(i, j) - b(0, j);
    }

    GPUMatrix GPUMatrix::subtract_rowwise(const GPUMatrix& matrix) const
    {
        assert(matrix.rows == 1);
        assert(matrix.cols == cols);

        GPUMatrix res(rows, cols);

        dim3 blockdim(32, 32);
        dim3 griddim((rows + blockdim.x - 1) / blockdim.x,
                     (cols + blockdim.y - 1) / blockdim.y);
        subtract_rowwise_kernel<<<griddim, blockdim>>>(*this, matrix, res);

        return res;
    }

    __global__ void subtract_kernel(GPUMatrix a, GPUMatrix b, GPUMatrix res)
    {
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= a.rows)
            return;

        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (j >= a.cols)
            return;

        res(i, j) = a(i, j) - b(i, j);
    }

    GPUMatrix GPUMatrix::subtract(const GPUMatrix& matrix) const
    {
        assert(matrix.rows == rows);
        assert(matrix.cols == cols);

        GPUMatrix res(rows, cols);

        dim3 blockdim(32, 32);
        dim3 griddim((rows + blockdim.x - 1) / blockdim.x,
                     (cols + blockdim.y - 1) / blockdim.y);
        subtract_kernel<<<griddim, blockdim>>>(*this, matrix, res);

        return res;
    }

    __global__ void dot_kernel(GPUMatrix a, GPUMatrix b, GPUMatrix res)
    {
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= a.rows)
            return;

        uint j = blockIdx.y * blockDim.y + threadIdx.y;
        if (j >= a.cols)
            return;

        float total = 0;
        for (size_t k = 0; k < a.cols; ++k)
            total += a(i, k) * b(k, j);

        res(i, j) = total;
    }

    GPUMatrix GPUMatrix::dot(const GPUMatrix& matrix) const
    {
        assert(cols == matrix.rows);

        GPUMatrix res(rows, matrix.cols);

        dim3 blockdim(32, 32);
        dim3 griddim((res.rows + blockdim.x - 1) / blockdim.x,
                     (res.cols + blockdim.y - 1) / blockdim.y);
        dot_kernel<<<griddim, blockdim>>>(*this, matrix, res);

        return res;
    }

    __global__ void closest_kernel(GPUMatrix from, GPUMatrix to, GPUMatrix res)
    {
        uint row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= from.rows)
            return;

        size_t closest_i = 0;
        float closest_dist = GPUMatrix::distance(from, row, to, closest_i);
        for (size_t j = 1; j < to.rows; ++j)
        {
            float dist = GPUMatrix::distance(from, row, to, j);
            if (dist < closest_dist)
            {
                closest_i = j;
                closest_dist = dist;
            }
        }

        for (size_t j = 0; j < from.cols; ++j)
            res(row, j) = to(closest_i, j);
    }

    GPUMatrix GPUMatrix::closest(const GPUMatrix& matrix) const
    {
        assert(cols == 3);
        assert(matrix.cols == 3);
        assert(matrix.rows > 0);

        GPUMatrix res(rows, cols);

        dim3 blockdim(1024);
        dim3 griddim((rows + blockdim.x - 1) / blockdim.x);
        closest_kernel<<<griddim, blockdim>>>(*this, matrix, res);

        return res;
    }

    __global__ void find_covariance_kernel(GPUMatrix a, GPUMatrix b,
                                           GPUMatrix res)
    {
        uint j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= res.rows)
            return;

        uint k = blockIdx.y * blockDim.y + threadIdx.y;
        if (k >= res.cols)
            return;

        for (size_t i = 0; i < a.rows; ++i)
            res(j, k) += a(i, j) * b(i, k);
    }

    GPUMatrix GPUMatrix::find_covariance(const GPUMatrix& a, const GPUMatrix& b)
    {
        assert(a.cols == b.cols);
        assert(a.rows == b.rows);

        auto res = GPUMatrix::zero(a.cols, b.cols);

        dim3 blockdim(32, 32);
        dim3 griddim((res.rows + blockdim.x - 1) / blockdim.x,
                     (res.cols + blockdim.y - 1) / blockdim.y);
        find_covariance_kernel<<<griddim, blockdim>>>(a, b, res);

        return res;
    }

    __global__ void transpose_kernel(GPUMatrix a, GPUMatrix res)
    {
        for (size_t i = 0; i < a.rows; ++i)
            for (size_t j = 0; j < a.cols; ++j)
                res(j, i) = a(i, j);
    }

    GPUMatrix GPUMatrix::transpose() const
    {
        GPUMatrix res(cols, rows);

        transpose_kernel<<<1, 1>>>(*this, res);

        return res;
    }
} // namespace libgpu