#include <torch/extension.h>

#include "api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess", &preprocess_checked);
    m.def("render_forward", &render_forward_checked);
    m.def("render_scene", &render_scene_checked);
    m.def("render_backward", &render_backward_checked);
    m.def("preprocess_backward", &preprocess_backward_checked);
    m.def("backward_scene", &backward_scene_checked);
}
