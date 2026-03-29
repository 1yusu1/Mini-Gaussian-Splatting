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
    float4* conic_opacity;
    float3* colors;

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
        obtain(chunk, s.point_offsets, P);
        obtain(chunk, s.conic_opacity, P);
        obtain(chunk, s.colors, P);

        obtain(chunk, s.sort_keys, L);
        obtain(chunk, s.sort_values, L);
        obtain(chunk, s.tile_ranges, num_tiles);
        return s;
    }

    static size_t bytesRequired(size_t P, size_t L, size_t num_tiles) {
        uintptr_t offset = 0;
        auto bump = [&](size_t elem_size, size_t count, size_t alignment = 128) {
            offset = (offset + alignment - 1) & ~(alignment - 1);
            offset += elem_size * count;
        };

        bump(sizeof(float2), P);
        bump(sizeof(float), P);
        bump(sizeof(int), P);
        bump(sizeof(float), P * 6);
        bump(sizeof(float4), P);
        bump(sizeof(float3), P);
        bump(sizeof(uint32_t), P);
        bump(sizeof(float4), P);
        bump(sizeof(float3), P);

        bump(sizeof(uint64_t), L);
        bump(sizeof(uint32_t), L);
        bump(sizeof(uint2), num_tiles);
        return static_cast<size_t>(offset);
    }
};
