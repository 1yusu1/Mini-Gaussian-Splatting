#include <torch/extension.h>
#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_tiler", &run_tiler_cuda);
}
