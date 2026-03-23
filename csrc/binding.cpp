#include <torch/extension.h>

// 声明 CUDA 函数
void quat_to_rot_cuda(const at::Tensor& quats, at::Tensor& rots);

// 接口检查与转发
void quat_to_rot(at::Tensor quats, at::Tensor rots) {
    TORCH_CHECK(quats.is_cuda(), "Input must be CUDA tensor");
    quat_to_rot_cuda(quats, rots);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quat_to_rot", &quat_to_rot, "Quaternion to Rotation Matrix");
}