#include <torch/extension.h>
#include <functional>
#include "state.h"

void preprocess_points(
    int P, const float3* points3D,
    const float* viewmatrix, const float* projmatrix,
    const float focal_x, const float focal_y,
    int W, int H, float r, PointState s);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
run_tiler(
    torch::Tensor points3D,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor scales,
    torch::Tensor rotations,
    float focal_x, float focal_y,
    int W, int H,
    float radius)
{
    int P = points3D.size(0);

    auto pts_c = points3D.contiguous();
    auto view_c = viewmatrix.contiguous();
    auto proj_c = projmatrix.contiguous();
    auto sc_c = scales.contiguous();
    auto rot_c = rotations.contiguous();

    torch::Tensor buffer = torch::empty({0}, points3D.options().dtype(torch::kByte));
    auto resizeFunc = [&](size_t N) {
        buffer.resize_({(long long)N});
        return reinterpret_cast<char*>(buffer.contiguous().data_ptr());
    };

    size_t needed = P * 80 + 1024;
    char* chunkptr = resizeFunc(needed);

    PointState s = PointState::fromChunk(chunkptr, P);
    cudaMemcpyAsync(s.scales, sc_c.data_ptr<float>(), P * sizeof(float3), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(s.quat, rot_c.data_ptr<float>(), P * sizeof(float4), cudaMemcpyDeviceToDevice);

    preprocess_points(P,
        (float3*)pts_c.data_ptr<float>(),
        view_c.data_ptr<float>(),
        proj_c.data_ptr<float>(),
        focal_x, focal_y,
        W, H, radius, s);

    auto res_2d = torch::from_blob(s.points2D, {P, 2}, points3D.options()).clone();
    auto res_counts = torch::from_blob(s.tile_counts, {P}, points3D.options().dtype(torch::kInt)).clone();
    auto res_cov3d = torch::from_blob(s.cov3D, {P, 6}, points3D.options()).clone();
    auto res_conic = torch::from_blob(s.conic, {P, 3}, points3D.options()).clone();


    return std::make_tuple(res_2d, res_counts, res_cov3d, res_conic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_tiler", &run_tiler);
}