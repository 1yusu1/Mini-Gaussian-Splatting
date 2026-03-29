#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "ops.h"

namespace {

constexpr int kTileWidth = 16;
constexpr int kTileHeight = 16;
constexpr int kThreadsPerBlock = kTileWidth * kTileHeight;
constexpr float kAlphaThreshold = 1.0f / 255.0f;
constexpr float kTransmittanceEpsilon = 0.0001f;

static inline void checkCuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

[[noreturn]] void throw_not_implemented(const char* fn_name) {
    throw std::runtime_error(std::string(fn_name) + " is not implemented yet");
}

__global__ void renderBackwardColorKernel(
    int W,
    int H,
    const uint2* ranges,
    const uint32_t* point_list,
    const float2* means2D,
    const float4* conic_opacity,
    const float* colors,
    const float* grad_image,
    float* grad_points2D,
    float* grad_conic_opacity,
    float* grad_colors) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t horizontal_blocks = (W + kTileWidth - 1) / kTileWidth;
    const uint2 tile_id = {blockIdx.x, blockIdx.y};
    const uint2 pix = {tile_id.x * kTileWidth + threadIdx.x, tile_id.y * kTileHeight + threadIdx.y};
    if (pix.x >= W || pix.y >= H) return;

    const uint32_t pix_id = W * pix.y + pix.x;
    const uint2 range = ranges[tile_id.y * horizontal_blocks + tile_id.x];
    const int to_do = range.y - range.x;

    __shared__ int collected_id[kThreadsPerBlock];
    __shared__ float2 collected_xy[kThreadsPerBlock];
    __shared__ float4 collected_conic[kThreadsPerBlock];

    float T = 1.0f;
    const float dL_dpixel_r = grad_image[0 * H * W + pix_id];
    const float dL_dpixel_g = grad_image[1 * H * W + pix_id];
    const float dL_dpixel_b = grad_image[2 * H * W + pix_id];

    for (int i = 0; i < to_do; i += kThreadsPerBlock) {
        const int fetch_idx = range.x + i + tid;
        if (fetch_idx < range.y) {
            const int g_id = point_list[fetch_idx];
            collected_id[tid] = g_id;
            collected_xy[tid] = means2D[g_id];
            collected_conic[tid] = conic_opacity[g_id];
        }
        __syncthreads();

        const int batch_size = min(kThreadsPerBlock, to_do - i);
        for (int j = 0; j < batch_size; ++j) {
            const float2 g_xy = collected_xy[j];
            const float4 con_o = collected_conic[j];
            const float dx = g_xy.x - static_cast<float>(pix.x);
            const float dy = g_xy.y - static_cast<float>(pix.y);
            const float power = -0.5f * (dx * dx * con_o.x + dy * dy * con_o.z) - dx * dy * con_o.y;
            if (power > 0.0f) continue;

            const float raw_alpha = con_o.w * expf(power);
            const float alpha = min(0.99f, raw_alpha);
            if (alpha < kAlphaThreshold) continue;

            const float weight = alpha * T;
            const int g_id = collected_id[j];
            const float color_r = colors[g_id * 3 + 0];
            const float color_g = colors[g_id * 3 + 1];
            const float color_b = colors[g_id * 3 + 2];
            const float dL_dalpha =
                (color_r * dL_dpixel_r + color_g * dL_dpixel_g + color_b * dL_dpixel_b) * T;
            atomicAdd(&grad_colors[g_id * 3 + 0], dL_dpixel_r * weight);
            atomicAdd(&grad_colors[g_id * 3 + 1], dL_dpixel_g * weight);
            atomicAdd(&grad_colors[g_id * 3 + 2], dL_dpixel_b * weight);
            if (raw_alpha < 0.99f) {
                const float dL_dpower = raw_alpha * dL_dalpha;
                atomicAdd(&grad_points2D[g_id * 2 + 0], dL_dpower * (-(con_o.x * dx + con_o.y * dy)));
                atomicAdd(&grad_points2D[g_id * 2 + 1], dL_dpower * (-(con_o.z * dy + con_o.y * dx)));
                atomicAdd(&grad_conic_opacity[g_id * 4 + 0], dL_dpower * (-0.5f * dx * dx));
                atomicAdd(&grad_conic_opacity[g_id * 4 + 1], dL_dpower * (-dx * dy));
                atomicAdd(&grad_conic_opacity[g_id * 4 + 2], dL_dpower * (-0.5f * dy * dy));
                atomicAdd(&grad_conic_opacity[g_id * 4 + 3], expf(power) * dL_dalpha);
            }

            T *= (1.0f - alpha);
            if (T < kTransmittanceEpsilon) {
                break;
            }
        }
        __syncthreads();
        if (T < kTransmittanceEpsilon) {
            break;
        }
    }
}

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
render_backward_cuda(
    torch::Tensor points2D,
    torch::Tensor conic_opacity,
    torch::Tensor colors,
    torch::Tensor tile_ranges,
    torch::Tensor point_list,
    torch::Tensor final_T,
    torch::Tensor n_contrib,
    torch::Tensor grad_image,
    torch::Tensor grad_final_T,
    int W,
    int H) {
    auto points2D_c = points2D.contiguous();
    auto conic_c = conic_opacity.contiguous();
    auto colors_c = colors.contiguous();
    auto ranges_c = tile_ranges.contiguous();
    auto point_list_c = point_list.contiguous();
    auto grad_image_c = grad_image.contiguous();

    const auto P = points2D.size(0);
    auto grad_points2D = torch::zeros({P, 2}, points2D.options().dtype(torch::kFloat));
    auto grad_conic_opacity = torch::zeros({P, 4}, conic_opacity.options().dtype(torch::kFloat));
    auto grad_colors = torch::zeros({P, 3}, colors.options().dtype(torch::kFloat));

    (void)final_T;
    (void)n_contrib;
    (void)grad_final_T;

    if (point_list_c.numel() == 0) {
        return std::make_tuple(grad_points2D, grad_conic_opacity, grad_colors);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((W + kTileWidth - 1) / kTileWidth, (H + kTileHeight - 1) / kTileHeight, 1);
    dim3 block(kTileWidth, kTileHeight, 1);

    renderBackwardColorKernel<<<grid, block, 0, stream>>>(
        W,
        H,
        reinterpret_cast<uint2*>(ranges_c.data_ptr<int>()),
        reinterpret_cast<uint32_t*>(point_list_c.data_ptr<int>()),
        reinterpret_cast<float2*>(points2D_c.data_ptr<float>()),
        reinterpret_cast<float4*>(conic_c.data_ptr<float>()),
        colors_c.data_ptr<float>(),
        grad_image_c.data_ptr<float>(),
        grad_points2D.data_ptr<float>(),
        grad_conic_opacity.data_ptr<float>(),
        grad_colors.data_ptr<float>());
    checkCuda(cudaGetLastError(), "render_backward_color launch");
    checkCuda(cudaStreamSynchronize(stream), "render_backward_color sync");

    return std::make_tuple(grad_points2D, grad_conic_opacity, grad_colors);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
preprocess_backward_cuda(
    torch::Tensor points3D,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor grad_points2D,
    torch::Tensor grad_conic_opacity,
    torch::Tensor grad_colors,
    float focal_x,
    float focal_y,
    int W,
    int H,
    float radius) {
    (void)points3D;
    (void)viewmatrix;
    (void)projmatrix;
    (void)scales;
    (void)rotations;
    (void)colors;
    (void)opacities;
    (void)grad_points2D;
    (void)grad_conic_opacity;
    (void)grad_colors;
    (void)focal_x;
    (void)focal_y;
    (void)W;
    (void)H;
    (void)radius;
    throw_not_implemented("preprocess_backward_cuda");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_scene_cuda(
    torch::Tensor points3D,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor grad_image,
    torch::Tensor grad_final_T,
    float focal_x,
    float focal_y,
    int W,
    int H,
    float radius) {
    (void)points3D;
    (void)viewmatrix;
    (void)projmatrix;
    (void)scales;
    (void)rotations;
    (void)colors;
    (void)opacities;
    (void)grad_image;
    (void)grad_final_T;
    (void)focal_x;
    (void)focal_y;
    (void)W;
    (void)H;
    (void)radius;
    throw_not_implemented("backward_scene_cuda");
}
