# Mini-Gaussian-Splatting-CUDA

这是一个基于 PyTorch C++ 扩展实现的 3DGS 核心算子库。本项目旨在通过手写底层 CUDA 代码，深入研究 3DGS 的几何映射、动态内存管理及可微渲染机制。

## 当前进度：几何投影与 EWA Splatting

目前已实现 3DGS 的几何后端逻辑，并与 PyTorch 真值模型实现数值对齐。

### 核心特性
- **各向异性 3D 协方差建模**：实现了基于李群 $SO(3)$ 的四元数旋转与空间缩放复合，构建 3D 协方差矩阵 $\Sigma = RSS^T R^T$。
- **EWA Splatting 投影**：实现了基于雅可比矩阵的投影算子，将 3D 椭球映射至 2D 像平面，包含低通滤波补偿。
- **动态显存调度架构**：模仿官方 3DGS 风格，利用 C++ 回调函数与 `obtain` 线性分配模式，实现了显存池的连续扩容。
- **高性能内存布局**：
  - 采用 **128 字节对齐**，确保 GPU 合并访问。
  - 使用结构化 `PointState` 管理裸指针，优化 L2 缓存命中率。

### 数学实现细节
1. **相机模型**：采用针孔相机模型，支持分拆的 ViewMatrix 与 ProjMatrix 输入。
2. **矩阵主序**：后端算子采用**行主序**寻址，与 PyTorch 默认逻辑一致。

## 项目结构
```text
.
├── csrc/
│   ├── state.h       # 内存布局与动态分配工具
│   ├── kernel.cu     # 3D/2D 协方差计算与投影 CUDA Kernel
│   └── binding.cpp   # PyTorch 接口调度与 C++ Orchestrator
├── setup.py          # 编译脚本
└── .gitignore        # 二进制产物屏蔽清单
```

---

