import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="my_gs_ops",
    ext_modules=[
        CUDAExtension(
            name="my_gs_ops",
            sources=["csrc/binding.cpp", "csrc/kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)