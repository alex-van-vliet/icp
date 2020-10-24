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

    enum State
    {
        PREFIX,
        INFIX,
        SUFFIX,
    };

    struct StackFrame
    {
        State state;
        GPUVPTree::GPUVPTreeNode* node;
        float d;
        GPUVPTree::GPUVPTreeSearchResult result;
    };

    __device__ GPUVPTree::GPUVPTreeSearchResult
    search_points(GPUVPTree::GPUVPTreeNode* node, const float* query)
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

    __device__ void prefix(const float* query, StackFrame stack[16],
                           uint& stack_size, GPUVPTree::GPUVPTreeNode* node)
    {
        while (!node->points)
        {
            float d = distance(query, &node->center_x);

            stack[stack_size++] = {State::INFIX, node, d};

            node = d < node->radius ? node->inside : node->outside;
        }
        stack[stack_size].result = search_points(node, query);
    }

    __device__ auto GPUVPTree::search(const float* query)
        -> GPUVPTreeSearchResult
    {
        StackFrame stack[16];
        uint stack_size = 0;

        prefix(query, stack, stack_size,
               reinterpret_cast<GPUVPTreeNode*>(pointer));

        while (stack_size > 0)
        {
            StackFrame* next_frame = stack + stack_size;
            StackFrame* frame = next_frame - 1;
            GPUVPTreeNode* node = frame->node;

            if (frame->state == State::INFIX)
            {
                float d = frame->d;
                float threshold = fabsf(node->radius - d);

                frame->result = next_frame->result;
                if (frame->result.distance < threshold)
                    stack_size -= 1;
                else
                {
                    prefix(query, stack, stack_size,
                           d < node->radius ? node->outside : node->inside);

                    frame->state = State::SUFFIX;
                }
            }
            else if (frame->state == State::SUFFIX)
            {
                if (next_frame->result.distance < frame->result.distance)
                    frame->result = next_frame->result;
                stack_size -= 1;
            }
        }

        return stack[0].result;
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

        return res;
    }
} // namespace libgpu