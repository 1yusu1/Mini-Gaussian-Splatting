#pragma once

#include <torch/extension.h>

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
preprocess_checked(
    torch::Tensor points3D,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor colors,
    torch::Tensor opacities,
    float focal_x,
    float focal_y,
    int W,
    int H,
    float radius);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
render_forward_checked(
    torch::Tensor points2D,
    torch::Tensor conic_opacity,
    torch::Tensor colors,
    torch::Tensor tile_ranges,
    torch::Tensor point_list,
    int W,
    int H);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
render_scene_checked(
    torch::Tensor points3D,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor colors,
    torch::Tensor opacities,
    float focal_x,
    float focal_y,
    int W,
    int H,
    float radius);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
render_backward_checked(
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
    int H);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
preprocess_backward_checked(
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
    float radius);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_scene_checked(
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
    float radius);
