#pragma once

#include <torch/extension.h>

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
    float radius);
