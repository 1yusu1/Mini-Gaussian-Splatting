#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

template<typename T>
static void obtain(char*& chunk, T*& ptr, size_t count, size_t alignment = 128) {
    size_t offset = (reinterpret_cast<uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}

struct PointState {
    float2* points2D;
    float* depths;
    int* tile_counts;
    float* cov3D;
    float4* quat;
    float3* scales;
    float3* conic;

    uint32_t* point_offsets;
    uint64_t* sort_keys;
    uint32_t* sort_values;
    uint2* tile_ranges;

    static PointState fromChunk(char*& chunk, size_t P, size_t L, size_t num_tiles) {
        PointState s;
        obtain(chunk, s.points2D, P);
        obtain(chunk, s.depths, P);
        obtain(chunk, s.tile_counts, P);
        obtain(chunk, s.cov3D, P * 6);
        obtain(chunk, s.quat, P);
        obtain(chunk, s.scales, P);
        obtain(chunk, s.conic, P);
        obtain(chunk, s.point_offsets, P);

        obtain(chunk, s.sort_keys, L);
        obtain(chunk, s.sort_values, L);
        obtain(chunk, s.tile_ranges, num_tiles);
        return s;
    }
};