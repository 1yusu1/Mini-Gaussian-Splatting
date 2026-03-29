#include <torch/extension.h>

#include "api.h"
#include "ops.h"

namespace {

void check_cuda_float_tensor(const torch::Tensor& tensor, const char* name, int dims) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.dim() == dims, name, " must have ", dims, " dimensions");
}

void check_cuda_int_tensor(const torch::Tensor& tensor, const char* name, int dims) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(tensor.dim() == dims, name, " must have ", dims, " dimensions");
}

void check_scene_inputs(
    const torch::Tensor& points3D,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    float focal_x,
    float focal_y,
    int W,
    int H,
    float radius) {
    check_cuda_float_tensor(points3D, "points3D", 2);
    check_cuda_float_tensor(viewmatrix, "viewmatrix", 2);
    check_cuda_float_tensor(projmatrix, "projmatrix", 2);
    check_cuda_float_tensor(scales, "scales", 2);
    check_cuda_float_tensor(rotations, "rotations", 2);
    check_cuda_float_tensor(colors, "colors", 2);
    check_cuda_float_tensor(opacities, "opacities", 1);

    TORCH_CHECK(points3D.size(1) == 3, "points3D must have shape [P, 3]");
    TORCH_CHECK(viewmatrix.sizes() == torch::IntArrayRef({4, 4}), "viewmatrix must have shape [4, 4]");
    TORCH_CHECK(projmatrix.sizes() == torch::IntArrayRef({4, 4}), "projmatrix must have shape [4, 4]");
    TORCH_CHECK(scales.sizes() == torch::IntArrayRef({points3D.size(0), 3}), "scales must have shape [P, 3]");
    TORCH_CHECK(rotations.sizes() == torch::IntArrayRef({points3D.size(0), 4}), "rotations must have shape [P, 4]");
    TORCH_CHECK(colors.sizes() == torch::IntArrayRef({points3D.size(0), 3}), "colors must have shape [P, 3]");
    TORCH_CHECK(opacities.size(0) == points3D.size(0), "opacities must have shape [P]");
    TORCH_CHECK(W > 0 && H > 0, "W and H must be positive");
    TORCH_CHECK(radius >= 0.0f, "radius must be non-negative");
    TORCH_CHECK(focal_x > 0.0f && focal_y > 0.0f, "focal_x and focal_y must be positive");
}

}  // namespace

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
    float radius) {
    check_scene_inputs(
        points3D, viewmatrix, projmatrix, scales, rotations, colors, opacities, focal_x, focal_y, W, H, radius);
    return preprocess_cuda(
        points3D, viewmatrix, projmatrix, scales, rotations, colors, opacities, focal_x, focal_y, W, H, radius);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
render_forward_checked(
    torch::Tensor points2D,
    torch::Tensor conic_opacity,
    torch::Tensor colors,
    torch::Tensor tile_ranges,
    torch::Tensor point_list,
    int W,
    int H) {
    check_cuda_float_tensor(points2D, "points2D", 2);
    check_cuda_float_tensor(conic_opacity, "conic_opacity", 2);
    check_cuda_float_tensor(colors, "colors", 2);
    check_cuda_int_tensor(tile_ranges, "tile_ranges", 2);
    check_cuda_int_tensor(point_list, "point_list", 1);

    const auto P = points2D.size(0);
    const auto num_tiles = ((W + 15) / 16) * ((H + 15) / 16);

    TORCH_CHECK(points2D.size(1) == 2, "points2D must have shape [P, 2]");
    TORCH_CHECK(conic_opacity.sizes() == torch::IntArrayRef({P, 4}), "conic_opacity must have shape [P, 4]");
    TORCH_CHECK(colors.sizes() == torch::IntArrayRef({P, 3}), "colors must have shape [P, 3]");
    TORCH_CHECK(tile_ranges.sizes() == torch::IntArrayRef({num_tiles, 2}), "tile_ranges must have shape [num_tiles, 2]");
    TORCH_CHECK(W > 0 && H > 0, "W and H must be positive");

    return render_forward_cuda(points2D, conic_opacity, colors, tile_ranges, point_list, W, H);
}

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
    float radius) {
    check_scene_inputs(
        points3D, viewmatrix, projmatrix, scales, rotations, colors, opacities, focal_x, focal_y, W, H, radius);
    return render_scene_cuda(
        points3D, viewmatrix, projmatrix, scales, rotations, colors, opacities, focal_x, focal_y, W, H, radius);
}

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
    int H) {
    check_cuda_float_tensor(points2D, "points2D", 2);
    check_cuda_float_tensor(conic_opacity, "conic_opacity", 2);
    check_cuda_float_tensor(colors, "colors", 2);
    check_cuda_int_tensor(tile_ranges, "tile_ranges", 2);
    check_cuda_int_tensor(point_list, "point_list", 1);
    check_cuda_float_tensor(final_T, "final_T", 2);
    check_cuda_int_tensor(n_contrib, "n_contrib", 2);
    check_cuda_float_tensor(grad_image, "grad_image", 3);
    check_cuda_float_tensor(grad_final_T, "grad_final_T", 2);

    const auto P = points2D.size(0);
    const auto num_tiles = ((W + 15) / 16) * ((H + 15) / 16);

    TORCH_CHECK(points2D.size(1) == 2, "points2D must have shape [P, 2]");
    TORCH_CHECK(conic_opacity.sizes() == torch::IntArrayRef({P, 4}), "conic_opacity must have shape [P, 4]");
    TORCH_CHECK(colors.sizes() == torch::IntArrayRef({P, 3}), "colors must have shape [P, 3]");
    TORCH_CHECK(tile_ranges.sizes() == torch::IntArrayRef({num_tiles, 2}), "tile_ranges must have shape [num_tiles, 2]");
    TORCH_CHECK(final_T.sizes() == torch::IntArrayRef({H, W}), "final_T must have shape [H, W]");
    TORCH_CHECK(n_contrib.sizes() == torch::IntArrayRef({H, W}), "n_contrib must have shape [H, W]");
    TORCH_CHECK(grad_image.sizes() == torch::IntArrayRef({3, H, W}), "grad_image must have shape [3, H, W]");
    TORCH_CHECK(grad_final_T.sizes() == torch::IntArrayRef({H, W}), "grad_final_T must have shape [H, W]");
    TORCH_CHECK(W > 0 && H > 0, "W and H must be positive");

    return render_backward_cuda(
        points2D, conic_opacity, colors, tile_ranges, point_list, final_T, n_contrib, grad_image, grad_final_T, W, H);
}

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
    float radius) {
    check_scene_inputs(
        points3D, viewmatrix, projmatrix, scales, rotations, colors, opacities, focal_x, focal_y, W, H, radius);
    check_cuda_float_tensor(grad_points2D, "grad_points2D", 2);
    check_cuda_float_tensor(grad_conic_opacity, "grad_conic_opacity", 2);
    check_cuda_float_tensor(grad_colors, "grad_colors", 2);

    const auto P = points3D.size(0);
    TORCH_CHECK(grad_points2D.sizes() == torch::IntArrayRef({P, 2}), "grad_points2D must have shape [P, 2]");
    TORCH_CHECK(grad_conic_opacity.sizes() == torch::IntArrayRef({P, 4}), "grad_conic_opacity must have shape [P, 4]");
    TORCH_CHECK(grad_colors.sizes() == torch::IntArrayRef({P, 3}), "grad_colors must have shape [P, 3]");

    return preprocess_backward_cuda(
        points3D,
        viewmatrix,
        projmatrix,
        scales,
        rotations,
        colors,
        opacities,
        grad_points2D,
        grad_conic_opacity,
        grad_colors,
        focal_x,
        focal_y,
        W,
        H,
        radius);
}

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
    float radius) {
    check_scene_inputs(
        points3D, viewmatrix, projmatrix, scales, rotations, colors, opacities, focal_x, focal_y, W, H, radius);
    check_cuda_float_tensor(grad_image, "grad_image", 3);
    check_cuda_float_tensor(grad_final_T, "grad_final_T", 2);
    TORCH_CHECK(grad_image.sizes() == torch::IntArrayRef({3, H, W}), "grad_image must have shape [3, H, W]");
    TORCH_CHECK(grad_final_T.sizes() == torch::IntArrayRef({H, W}), "grad_final_T must have shape [H, W]");

    return backward_scene_cuda(
        points3D,
        viewmatrix,
        projmatrix,
        scales,
        rotations,
        colors,
        opacities,
        grad_image,
        grad_final_T,
        focal_x,
        focal_y,
        W,
        H,
        radius);
}
