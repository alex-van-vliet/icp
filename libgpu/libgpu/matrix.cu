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
        auto matrix = GPUMatrix(p.size(), 3);

        cudaMemcpy(matrix.ptr, p.data(), p.size() * sizeof(libcpu::Point3D),
                   cudaMemcpyHostToDevice);

        return matrix;
    }

    libcpu::point_list GPUMatrix::to_point_list() const
    {
        assert(cols == 3);
        libcpu::point_list list;
        list.resize(rows);

        cudaMemcpy(list.data(), ptr, rows * sizeof(libcpu::Point3D),
                   cudaMemcpyDeviceToHost);

        return list;
    }

    GPUMatrix GPUMatrix::from_cpu(const utils::Matrix<float>& cpu)
    {
        GPUMatrix res(cpu.rows, cpu.cols);

        cudaMemcpy(res.ptr, cpu.values.data(),
                   cpu.cols * cpu.rows * sizeof(float), cudaMemcpyHostToDevice);

        return res;
    }

    utils::Matrix<float> GPUMatrix::to_cpu() const
    {
        utils::Matrix<float> res(rows, cols);

        cudaMemcpy(res.values.data(), ptr, cols * rows * sizeof(float),
                   cudaMemcpyDeviceToHost);

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

    __device__ void block_sum_unroll(volatile float* memory, uint tidx)
    {
        memory[tidx] += memory[tidx + 32];
        memory[tidx] += memory[tidx + 16];
        memory[tidx] += memory[tidx + 8];
        memory[tidx] += memory[tidx + 4];
        memory[tidx] += memory[tidx + 2];
        memory[tidx] += memory[tidx + 1];
    }

    __global__ void block_sum_kernel(GPUMatrix inputs, GPUMatrix res)
    {
        extern __shared__ float all_memory[];

        uint line = blockIdx.x * blockDim.x * 2 + threadIdx.x;

        uint i = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= inputs.cols)
            return;

        float* memory = all_memory + i * blockDim.x;
        if (line >= inputs.rows)
            memory[threadIdx.x] = 0;
        else if (line + blockDim.x >= inputs.rows)
            memory[threadIdx.x] = inputs(line, i);
        else
            memory[threadIdx.x] =
                inputs(line, i) + inputs(line + blockDim.x, i);

        __syncthreads();

        for (uint s = blockDim.x / 2; s > 32; s >>= 1u)
        {
            if (threadIdx.x < s)
                memory[threadIdx.x] += memory[threadIdx.x + s];
            __syncthreads();
        }

        if (threadIdx.x < 32)
            block_sum_unroll(memory, threadIdx.x);

        if (threadIdx.x != 0)
            return;
        res(blockIdx.x, i) = memory[0];
    }

    GPUMatrix GPUMatrix::sum_colwise() const
    {
        assert(cols > 0 && cols <= 16);
        constexpr uint block_widths[] = {0,  1,  2,  4,  4,  8,  8,  8, 8,
                                         16, 16, 16, 16, 16, 16, 16, 16};
        uint block_width = block_widths[cols];

        dim3 blockdim_block_sum(1024 / block_width, block_width);
        blockdim_block_sum.x *= 2;
        dim3 griddim_block_sum(
            (rows + blockdim_block_sum.x - 1) / blockdim_block_sum.x,
            (cols + blockdim_block_sum.y - 1) / blockdim_block_sum.y);
        blockdim_block_sum.x /= 2;
        auto sums = GPUMatrix::zero(griddim_block_sum.x, cols);
        block_sum_kernel<<<griddim_block_sum, blockdim_block_sum,
                           blockdim_block_sum.x * cols * sizeof(float)>>>(*this,
                                                                          sums);

        if (sums.rows == 1)
            return sums;
        return sums.sum_colwise();
    }

    GPUMatrix GPUMatrix::mean() const
    {
        assert(cols == 3);

        auto divided = GPUMatrix(rows, cols);

        dim3 blockdim_divide(32, 32);
        dim3 griddim_divide((rows + blockdim_divide.x - 1) / blockdim_divide.x,
                            (cols + blockdim_divide.y - 1) / blockdim_divide.y);
        divide_kernel<<<griddim_divide, blockdim_divide>>>(*this, divided);

        return divided.sum_colwise();
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

    __global__ void product_kernel(GPUMatrix a, GPUMatrix b, GPUMatrix res)
    {
        uint line = blockIdx.x * blockDim.x + threadIdx.x;
        if (line >= res.rows)
            return;

        uint i_a = blockIdx.y * blockDim.y + threadIdx.y;
        if (i_a >= a.cols)
            return;

        uint i_b = blockIdx.z * blockDim.z + threadIdx.z;
        if (i_b >= b.cols)
            return;

        uint i = i_a * a.cols + i_b;
        res(line, i) = a(line, i_a) * b(line, i_b);
    }

    __global__ void find_covariance_kernel(GPUMatrix products, GPUMatrix res)
    {
        uint j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= res.rows)
            return;

        uint k = blockIdx.y * blockDim.y + threadIdx.y;
        if (k >= res.cols)
            return;

        uint pos = j * res.cols + k;
        res(j, k) = products(0, pos);
    }

    GPUMatrix GPUMatrix::find_covariance(const GPUMatrix& a, const GPUMatrix& b)
    {
        assert(a.cols == 3);
        assert(b.cols == 3);
        assert(a.rows == b.rows);

        auto products = GPUMatrix(a.rows, a.cols * b.cols);
        dim3 blockdim_product(64, 4, 4);
        dim3 griddim_product(
            (products.rows + blockdim_product.x - 1) / blockdim_product.x,
            (a.cols + blockdim_product.y - 1) / blockdim_product.y,
            (b.cols + blockdim_product.z - 1) / blockdim_product.z);
        product_kernel<<<griddim_product, blockdim_product>>>(a, b, products);

        auto sums = products.sum_colwise();

        auto res = GPUMatrix::zero(a.cols, b.cols);
        dim3 blockdim(32, 32);
        dim3 griddim((res.rows + blockdim.x - 1) / blockdim.x,
                     (res.cols + blockdim.y - 1) / blockdim.y);
        find_covariance_kernel<<<griddim, blockdim>>>(sums, res);

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
