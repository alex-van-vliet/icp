#include <iostream>

#include "vp-tree.hh"

namespace libgpu
{
    GPUVPTree::GPUVPTree(uint size)
        : pointer{cuda::mallocRaw<char>(size)}
        , should_delete{true}
    {}

    GPUVPTree::~GPUVPTree()
    {
        if (should_delete)
            cuda::free(pointer);
    }

    GPUVPTree::GPUVPTree(const GPUVPTree& other)
        : pointer{other.pointer}
        , should_delete{false}
    {}

    GPUVPTree::GPUVPTree(GPUVPTree&& other) noexcept
        : pointer{std::exchange(other.pointer, nullptr)}
        , should_delete{other.should_delete}
    {}

    GPUVPTree& GPUVPTree::operator=(GPUVPTree&& other) noexcept
    {
        pointer = std::exchange(other.pointer, nullptr);
        should_delete = other.should_delete;

        return *this;
    }

    uint GPUVPTree::memory_size(const libcpu::VPTree& tree)
    {
        if (!tree.inside)
        {
            uint size =
                sizeof(GPUVPTreeNode) + tree.points.size() * 3 * sizeof(float);
            return (((size + 32 - 1) / 32) * 32);
        }
        return sizeof(GPUVPTreeNode) + memory_size(*tree.inside)
            + memory_size(*tree.outside);
    }

    uint GPUVPTree::height(const libcpu::VPTree& tree)
    {
        if (!tree.inside)
            return 0;
        return 1 + std::max(height(*tree.inside), height(*tree.outside));
    }

    char* GPUVPTree::from_cpu(char* pointer, const libcpu::VPTree& tree)
    {
        GPUVPTreeNode* node = reinterpret_cast<GPUVPTreeNode*>(pointer);

        if (!tree.inside)
        {
            GPUVPTreeNode helper;
            helper.points = reinterpret_cast<float*>(node + 1);
            helper.nb_points = tree.points.size();

            cudaMemcpy(pointer, &helper, sizeof(GPUVPTreeNode),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(helper.points, tree.points.data(),
                       tree.points.size() * 3 * sizeof(float),
                       cudaMemcpyHostToDevice);

            return reinterpret_cast<char*>(pointer + memory_size(tree));
        }
        else
        {
            GPUVPTreeNode helper;
            helper.points = nullptr;
            helper.center_x = tree.center.x;
            helper.center_y = tree.center.y;
            helper.center_z = tree.center.z;
            helper.radius = tree.radius;

            helper.inside = node + 1;
            char* outside =
                from_cpu(reinterpret_cast<char*>(helper.inside), *tree.inside);

            helper.outside = reinterpret_cast<GPUVPTreeNode*>(outside);
            char* next = from_cpu(outside, *tree.outside);

            cudaMemcpy(pointer, &helper, sizeof(GPUVPTreeNode),
                       cudaMemcpyHostToDevice);

            return next;
        }
    }

    GPUVPTree GPUVPTree::from_cpu(const libcpu::VPTree& tree)
    {
        uint size = memory_size(tree);

        GPUVPTree tree_gpu(size);
        from_cpu(tree_gpu.pointer, tree);

        return tree_gpu;
    }

    __device__ float distance(const float* a, const float* b)
    {
        float dx = a[0] - b[0];
        float dy = a[1] - b[1];
        float dz = a[2] - b[2];
        return sqrt(dx * dx + dy * dy + dz * dz);
    }

    __device__ GPUVPTree::GPUVPTreeSearchResult
    search(const float query[3], const GPUVPTree::GPUVPTreeNode* node)
    {
        if (node->points)
        {
            uint nb_points = node->nb_points;
            float* points = node->points;

            uint closest_i = 0;
            float closest_dist = distance(query, points);
            for (size_t i = 1; i < nb_points; ++i)
            {
                points += 3;
                float dist = distance(query, points);
                if (dist < closest_dist)
                {
                    closest_i = i;
                    closest_dist = dist;
                }
            }

            float* point = node->points + 3 * closest_i;
            return {
                point[0],
                point[1],
                point[2],
                closest_dist,
            };
        }

        float d = distance(query, &node->center_x);
        auto n = d < node->radius ? search(query, node->inside)
                                  : search(query, node->outside);

        float threshold = node->radius - d;
        if (threshold < 0)
            threshold = -threshold;
        if (n.distance < threshold)
            return n;

        auto o = d < node->radius ? search(query, node->outside)
                                  : search(query, node->inside);

        if (o.distance < n.distance)
            return o;
        else
            return n;
    }

    __device__ auto GPUVPTree::search(const float* query)
        -> GPUVPTreeSearchResult
    {
        return ::libgpu::search(query,
                                reinterpret_cast<GPUVPTreeNode*>(pointer));
    }

    __global__ void closest_kernel(GPUMatrix from, GPUVPTree tree,
                                   GPUMatrix res)
    {
        uint row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= from.rows)
            return;

        float query[3] = {
            from(row, 0),
            from(row, 1),
            from(row, 2),
        };

        auto found = tree.search(query);

        res(row, 0) = found.x;
        res(row, 1) = found.y;
        res(row, 2) = found.z;
    }

    GPUMatrix GPUVPTree::closest(const GPUMatrix& queries)
    {
        assert(queries.cols == 3);

        GPUMatrix res(queries.rows, 3);

        dim3 blockdim(1024);
        dim3 griddim((queries.rows + blockdim.x - 1) / blockdim.x);
        closest_kernel<<<griddim, blockdim>>>(queries, *this, res);

        auto error = cudaGetLastError();
        if (error != cudaSuccess)
            printf("%s\n", cudaGetErrorString(error));

        cudaDeviceSynchronize();

        return res;
    }
} // namespace libgpu