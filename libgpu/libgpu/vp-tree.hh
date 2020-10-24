#pragma once

#include "cuda/memory.hh"
#include "libcpu/vp-tree.hh"
#include "matrix.hh"

namespace libgpu
{
    class GPUVPTree
    {
    private:
        char* pointer;
        bool should_delete;

        static uint memory_size(const libcpu::VPTree& tree);

        explicit GPUVPTree(uint size);

        static uint height(const libcpu::VPTree& tree);

        static char* from_cpu(char* pointer, const libcpu::VPTree& tree);

    public:
        ~GPUVPTree();

        GPUVPTree(const GPUVPTree& other);
        GPUVPTree& operator=(const GPUVPTree& other) = delete;

        GPUVPTree(GPUVPTree&& other) noexcept;
        GPUVPTree& operator=(GPUVPTree&& other) noexcept;

        struct GPUVPTreeNode
        {
            // Internal nodes
            GPUVPTreeNode* inside;
            GPUVPTreeNode* outside;
            float center_x;
            float center_y;
            float center_z;
            float radius;
            // Leaves
            float* points;
            uint nb_points;
        };

        static GPUVPTree from_cpu(const libcpu::VPTree& tree);

        GPUMatrix closest(const GPUMatrix& queries);

        struct GPUVPTreeSearchResult
        {
            float x;
            float y;
            float z;
            float distance;
        };

        __device__ GPUVPTreeSearchResult search(const float query[3]);
    };
} // namespace libgpu