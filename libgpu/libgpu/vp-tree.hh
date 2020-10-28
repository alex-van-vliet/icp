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

        /**
         * @brief Compute the total memory size of the vp tree.
         * @param tree The cpu tree.
         * @return The total memory size.
         */
        static uint memory_size(const libcpu::VPTree& tree);

        /**
         * @brief Create vp tree by allocating the sufficient memory.
         * @param size The total memory size.
         */
        explicit GPUVPTree(uint size);

        /**
         * @brief Compute the height of the vp tree.
         * @param tree The cpu tree.
         * @return The height of the tree.
         */
        static uint height(const libcpu::VPTree& tree);

        /**
         * @brief Transfer the vp tree to gpu.
         * @param pointer The pointer to gpu memory.
         * @param tree The cpu tree.
         * @return
         */
        static char* from_cpu(char* pointer, const libcpu::VPTree& tree);

    public:
        /**
         * @brief Destroy a vp tree.
         */
        ~GPUVPTree();

        /**
         * @brief Create a shallow copy of the vp tree.
         * @param other The vp tree to copy from.
         * This is mostly used to pass the GPUVPTree to GPU.
         */
        GPUVPTree(const GPUVPTree& other);
        GPUVPTree& operator=(const GPUVPTree& other) = delete;

        /**
         * @brief Move the vp tree into a new one.
         * @param other The vp tree to move from.
         */
        GPUVPTree(GPUVPTree&& other) noexcept;
        /**
         * @brief Move the vp tree into an existing one.
         * @param other The vp tree to move from.
         * @return This.
         */
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

        /**
         * @brief Transfer the vp tree to gpu.
         * @param tree The cpu vp tree.
         * @return The gpu vp tree.
         */
        static GPUVPTree from_cpu(const libcpu::VPTree& tree);

        /**
         * @brief Find the closest point of each point in the matrix.
         * @param queries The queries matrix.
         * @return The matrix containing the closest points.
         */
        GPUMatrix closest(const GPUMatrix& queries);

        struct GPUVPTreeSearchResult
        {
            float x;
            float y;
            float z;
            float distance;
        };

        /**
         * @brief Search the tree.
         * @param query The query point.
         * @return The closest point and its distance to the query.
         */
        __device__ GPUVPTreeSearchResult search(const float query[3]);
    };
} // namespace libgpu