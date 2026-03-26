#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include "state.h"
#include "ops.h"
#include <cub/cub.cuh>

__device__ void computeCov3D(const float3 scale, const float4 q, float* cov3D)
{
    float w = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    float R[9];
    R[0] = 1.f - 2.f * (y * y + z * z);
    R[1] = 2.f * (x * y - w * z);
    R[2] = 2.f * (x * z + w * y);
    R[3] = 2.f * (x * y + w * z);
    R[4] = 1.f - 2.f * (x * x + z * z);
    R[5] = 2.f * (y * z - w * x);
    R[6] = 2.f * (x * z - w * y);
    R[7] = 2.f * (y * z + w * x);
    R[8] = 1.f - 2.f * (x * x + y * y);

    float L[9];
    L[0] = R[0] * scale.x; L[1] = R[1] * scale.y; L[2] = R[2] * scale.z;
    L[3] = R[3] * scale.x; L[4] = R[4] * scale.y; L[5] = R[5] * scale.z;
    L[6] = R[6] * scale.x; L[7] = R[7] * scale.y; L[8] = R[8] * scale.z;

    cov3D[0] = L[0] * L[0] + L[1] * L[1] + L[2] * L[2];
    cov3D[1] = L[0] * L[3] + L[1] * L[4] + L[2] * L[5];
    cov3D[2] = L[0] * L[6] + L[1] * L[7] + L[2] * L[8];
    cov3D[3] = L[3] * L[3] + L[4] * L[4] + L[5] * L[5];
    cov3D[4] = L[3] * L[6] + L[4] * L[7] + L[5] * L[8];
    cov3D[5] = L[6] * L[6] + L[7] * L[7] + L[8] * L[8];
}

__device__ float3 computeCov2D(
    const float3& mean_view,
    const float focal_x,
    const float focal_y,
    const float* cov3D,
    const float* viewmatrix)
{
    float x = mean_view.x;
    float y = mean_view.y;
    float z = mean_view.z;

    float J[6] = {
        focal_x / z, 0.f, -(focal_x * x) / (z * z),
        0.f, focal_y / z, -(focal_y * y) / (z * z)
    };

    float W[9] = {
        viewmatrix[0], viewmatrix[1], viewmatrix[2],
        viewmatrix[4], viewmatrix[5], viewmatrix[6],
        viewmatrix[8], viewmatrix[9], viewmatrix[10]
    };

    float T[6];
    T[0] = J[0] * W[0] + J[1] * W[3] + J[2] * W[6];
    T[1] = J[0] * W[1] + J[1] * W[4] + J[2] * W[7];
    T[2] = J[0] * W[2] + J[1] * W[5] + J[2] * W[8];
    T[3] = J[3] * W[0] + J[4] * W[3] + J[5] * W[6];
    T[4] = J[3] * W[1] + J[4] * W[4] + J[5] * W[7];
    T[5] = J[3] * W[2] + J[4] * W[5] + J[5] * W[8];

    float m11 = cov3D[0], m12 = cov3D[1], m13 = cov3D[2];
    float m22 = cov3D[3], m23 = cov3D[4], m33 = cov3D[5];

    float TS[6];
    TS[0] = T[0] * m11 + T[1] * m12 + T[2] * m13;
    TS[1] = T[0] * m12 + T[1] * m22 + T[2] * m23;
    TS[2] = T[0] * m13 + T[1] * m23 + T[2] * m33;
    TS[3] = T[3] * m11 + T[4] * m12 + T[5] * m13;
    TS[4] = T[3] * m12 + T[4] * m22 + T[5] * m23;
    TS[5] = T[3] * m13 + T[4] * m23 + T[5] * m33;

    float cov_xx = TS[0] * T[0] + TS[1] * T[1] + TS[2] * T[2];
    float cov_xy = TS[0] * T[3] + TS[1] * T[4] + TS[2] * T[5];
    float cov_yy = TS[3] * T[3] + TS[4] * T[4] + TS[5] * T[5];

    return { cov_xx + 0.3f, cov_xy, cov_yy + 0.3f };
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* m)
{
    float3 res;
    res.x = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
    res.y = m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7];
    res.z = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11];
    return res;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* m)
{
    float4 res;
    res.x = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
    res.y = m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7];
    res.z = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11];
    res.w = m[12] * p.x + m[13] * p.y + m[14] * p.z + m[15];
    return res;
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
    return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

__global__ void duplicateWithKeys(
    int P,
    const float2* points2D,
    const float* depths,
    const uint32_t* offsets,
    uint64_t* sort_keys,
    uint32_t* sort_values,
    int W, int H, float radius,
    dim3 grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    if (depths[idx] > 0.2f) {
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        float2 p2d = points2D[idx];

        int x_min = max(0, (int)floorf((p2d.x - radius) / 16.0f));
        int x_max = min((int)grid.x, (int)ceilf((p2d.x + radius) / 16.0f));
        int y_min = max(0, (int)floorf((p2d.y - radius) / 16.0f));
        int y_max = min((int)grid.y, (int)ceilf((p2d.y + radius) / 16.0f));

        uint32_t d_bit = __float_as_uint(depths[idx]);

        for (int y = y_min; y < y_max; y++) {
            for (int x = x_min; x < x_max; x++) {
                uint64_t key = (uint64_t)(y * grid.x + x);
                key <<= 32;
                key |= d_bit;

                sort_keys[off] = key;
                sort_values[off] = idx;
                off++;
            }
        }
    }
}

__global__ void identifyTileRanges(int L, const uint64_t* keys, uint2* ranges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;

    uint32_t cur_tile = (uint32_t)(keys[idx] >> 32);
    if (idx == 0)
        ranges[cur_tile].x = 0;
    else {
        uint32_t prev_tile = (uint32_t)(keys[idx - 1] >> 32);
        if (cur_tile != prev_tile) {
            ranges[prev_tile].y = idx;
            ranges[cur_tile].x = idx;
        }
    }
    if (idx == L - 1) ranges[cur_tile].y = L;
}

__global__ void preprocess_points_kernel(
    int P,
    const float3* points3D,
    const float* viewmatrix,
    const float* projmatrix,
    const float focal_x,
    const float focal_y,
    int W, int H,
    float radius,
    PointState s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    s.tile_counts[idx] = 0;
    s.point_offsets[idx] = 0;
    float3 p_origin = points3D[idx];
    float3 p_view = transformPoint4x3(p_origin, viewmatrix);
    if (p_view.z < 0.2f) return;
    float4 p_hom = transformPoint4x4(p_origin, projmatrix);

    float p_w = 1.0f / (p_hom.w + 1e-7f);
    float3 p_ndc = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
    s.points2D[idx] = { ndc2Pix(p_ndc.x, W), ndc2Pix(p_ndc.y, H) };
    float2 p2d = s.points2D[idx];
    s.depths[idx] = p_view.z;

    int x_min = max(0, (int)floorf((p2d.x - radius) / 16.0f));
    int x_max = min((W + 15) / 16, (int)ceilf((p2d.x + radius) / 16.0f));
    int y_min = max(0, (int)floorf((p2d.y - radius) / 16.0f));
    int y_max = min((H + 15) / 16, (int)ceilf((p2d.y + radius) / 16.0f));
    s.tile_counts[idx] = max(0, x_max - x_min) * max(0, y_max - y_min);

    computeCov3D(s.scales[idx], s.quat[idx], s.cov3D + (idx * 6));
    float3 cov2D = computeCov2D(p_view, focal_x, focal_y, s.cov3D + idx * 6, viewmatrix);
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det <= 0.0f) return;
    float det_inv = 1.f / det;
    s.conic[idx] = { cov2D.z * det_inv, -cov2D.y * det_inv, cov2D.x * det_inv };
}

void preprocess_points(int P, const float3* points3D, const float* viewmatrix, const float* projmatrix,
const float focal_x, const float focal_y, int W, int H, float r, PointState s) {
    preprocess_points_kernel<<<(P + 255) / 256, 256>>>(P, points3D, viewmatrix, projmatrix, focal_x, focal_y, W, H, r, s);
}

void duplicate_points(int P, int L, int W, int H, float radius, int grid_x, int grid_y, PointState s){
    dim3 grid(grid_x, grid_y, 1);
    duplicateWithKeys<<<(P + 255) / 256, 256>>>(P, s.points2D, s.depths, s.point_offsets, s.sort_keys, s.sort_values, W, H, radius, grid);
}

void identify_ranges(int L, const uint64_t* keys, uint2* ranges_ptr){
    identifyTileRanges<<<(L + 255) / 256, 256>>>(L, keys, ranges_ptr);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
run_tiler_cuda(
    torch::Tensor points3D,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor scales,
    torch::Tensor rotations,
    float focal_x,
    float focal_y,
    int W,
    int H,
    float radius)
{
    int P = points3D.size(0);
    const int grid_x = (W + 15) / 16;
    const int grid_y = (H + 15) / 16;
    const int num_tiles = grid_x * grid_y;

    auto pts_c = points3D.contiguous();
    auto view_c = viewmatrix.contiguous();
    auto proj_c = projmatrix.contiguous();
    auto sc_c = scales.contiguous();
    auto rot_c = rotations.contiguous();

    auto make_byte_tensor = [&](size_t bytes) {
        return torch::empty({static_cast<long long>(bytes)}, points3D.options().dtype(torch::kByte));
    };

    size_t geom_bytes = P * 96 + 1024;
    torch::Tensor geom_buffer = make_byte_tensor(geom_bytes);
    char* geom_chunk = reinterpret_cast<char*>(geom_buffer.contiguous().data_ptr());
    PointState s_geom = PointState::fromChunk(geom_chunk, P, 0, 0);

    cudaMemcpyAsync(s_geom.scales, sc_c.data_ptr<float>(), P * sizeof(float3), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(s_geom.quat, rot_c.data_ptr<float>(), P * sizeof(float4), cudaMemcpyDeviceToDevice);

    preprocess_points(
        P,
        reinterpret_cast<float3*>(pts_c.data_ptr<float>()),
        view_c.data_ptr<float>(),
        proj_c.data_ptr<float>(),
        focal_x,
        focal_y,
        W,
        H,
        radius,
        s_geom);

    size_t scan_temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, scan_temp_bytes, s_geom.tile_counts, s_geom.point_offsets, P);

    torch::Tensor scan_temp = make_byte_tensor(scan_temp_bytes);
    cub::DeviceScan::InclusiveSum(
        scan_temp.data_ptr(),
        scan_temp_bytes,
        s_geom.tile_counts,
        s_geom.point_offsets,
        P);

    uint32_t L = 0;
    cudaMemcpy(&L, s_geom.point_offsets + P - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_temp_bytes,
        static_cast<uint64_t*>(nullptr),
        static_cast<uint64_t*>(nullptr),
        static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr),
        L);

    size_t cub_temp_bytes = scan_temp_bytes > sort_temp_bytes ? scan_temp_bytes : sort_temp_bytes;
    size_t final_bytes =
        geom_bytes +
        (2 * L * sizeof(uint64_t)) +
        (2 * L * sizeof(uint32_t)) +
        (num_tiles * sizeof(uint2)) +
        cub_temp_bytes +
        2048;

    torch::Tensor final_buffer = make_byte_tensor(final_bytes);
    char* chunk = reinterpret_cast<char*>(final_buffer.contiguous().data_ptr());
    PointState s = PointState::fromChunk(chunk, P, L, num_tiles);

    uint64_t* sort_keys_out = nullptr;
    uint32_t* sort_values_out = nullptr;
    obtain(chunk, sort_keys_out, L);
    obtain(chunk, sort_values_out, L);

    char* cub_temp_ptr = chunk;

    cudaMemcpyAsync(s.scales, sc_c.data_ptr<float>(), P * sizeof(float3), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(s.quat, rot_c.data_ptr<float>(), P * sizeof(float4), cudaMemcpyDeviceToDevice);

    preprocess_points(
        P,
        reinterpret_cast<float3*>(pts_c.data_ptr<float>()),
        view_c.data_ptr<float>(),
        proj_c.data_ptr<float>(),
        focal_x,
        focal_y,
        W,
        H,
        radius,
        s);

    cub::DeviceScan::InclusiveSum(
        cub_temp_ptr,
        scan_temp_bytes,
        s.tile_counts,
        s.point_offsets,
        P);

    duplicate_points(P, L, W, H, radius, grid_x, grid_y, s);

    cub::DeviceRadixSort::SortPairs(
        cub_temp_ptr,
        sort_temp_bytes,
        s.sort_keys,
        sort_keys_out,
        s.sort_values,
        sort_values_out,
        L);

    identify_ranges(L, sort_keys_out, s.tile_ranges);

    auto res_2d = torch::from_blob(s.points2D, {P, 2}, points3D.options()).clone();
    auto res_counts = torch::from_blob(s.tile_counts, {P}, points3D.options().dtype(torch::kInt)).clone();
    auto res_cov3d = torch::from_blob(s.cov3D, {P, 6}, points3D.options()).clone();
    auto res_conic = torch::from_blob(s.conic, {P, 3}, points3D.options()).clone();
    auto res_ranges = torch::from_blob(s.tile_ranges, {num_tiles, 2}, points3D.options().dtype(torch::kInt)).clone();
    auto res_keys = torch::from_blob(sort_keys_out, {L}, points3D.options().dtype(torch::kInt64)).clone();
    auto res_values = torch::from_blob(sort_values_out, {L}, points3D.options().dtype(torch::kInt)).clone();

    return std::make_tuple(
        res_2d,
        res_counts,
        res_cov3d,
        res_conic,
        res_ranges,
        res_keys,
        res_values);
}
