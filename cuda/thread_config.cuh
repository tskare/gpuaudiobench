#pragma once

// Thread/block size presets shared by CUDA benchmarks.
namespace ThreadConfig {
    // Preferred 1D launch sizes.
    constexpr int DEFAULT_BLOCK_SIZE_1D = 256;  // 8 warps, good default occupancy.
    constexpr int SMALL_BLOCK_SIZE_1D = 128;    // Lower register pressure option.
    constexpr int LARGE_BLOCK_SIZE_1D = 512;    // Helpful for bandwidth-bound work.
    constexpr int MAX_BLOCK_SIZE_1D = 1024;

    // 3D launch presets used by the FDTD-style kernels.
    constexpr int BLOCK_SIZE_3D_X = 8;
    constexpr int BLOCK_SIZE_3D_Y = 8;
    constexpr int BLOCK_SIZE_3D_Z = 8;

    constexpr int BLOCK_SIZE_3D_THIN_X = 16;
    constexpr int BLOCK_SIZE_3D_THIN_Y = 16;
    constexpr int BLOCK_SIZE_3D_THIN_Z = 2;

    constexpr int BLOCK_SIZE_3D_SMALL_X = 4;
    constexpr int BLOCK_SIZE_3D_SMALL_Y = 4;
    constexpr int BLOCK_SIZE_3D_SMALL_Z = 4;

    inline int calculateGridSize1D(size_t totalThreads,
                                   int blockSize = DEFAULT_BLOCK_SIZE_1D) {
        return static_cast<int>((totalThreads + blockSize - 1) / blockSize);
    }

    inline dim3 calculateGridSize3D(int nx, int ny, int nz,
                                    int blockX = BLOCK_SIZE_3D_X,
                                    int blockY = BLOCK_SIZE_3D_Y,
                                    int blockZ = BLOCK_SIZE_3D_Z) {
        return dim3((nx + blockX - 1) / blockX,
                    (ny + blockY - 1) / blockY,
                    (nz + blockZ - 1) / blockZ);
    }
} // namespace ThreadConfig
