# Mini-Gaussian-Splatting-CUDA

这是一个基于 PyTorch C++ 扩展实现的 3DGS 核心算子库。本项目旨在通过手写底层 CUDA 代码，深入研究 3DGS 的几何映射、动态内存管理及可微渲染机制。

## 当前进度：任务重组、空间加速与 Tiling 前处理

目前项目已经完成 3DGS 前向流程中的几何预处理与 Tile 任务重组部分，主数据流已经打通，并通过验证：

- 已完成 3D 高斯的 **3D 协方差构建** 与 **2D EWA 投影**。
- 已完成基于 tile 的 **覆盖计数、前缀和扫描、键值复制、排序与区间识别**。
- 已将 **PyTorch 绑定层** 与 **CUDA/CUB 执行层** 分离，避免 `CUB` 头文件进入 `.cpp` 编译单元。

当前代码重点是 3DGS 的 **preprocess + binning / tiling** 阶段，尚未实现完整的光栅化、颜色累积与反向传播。

### 核心特性
- **各向异性 3D 协方差建模**：实现了基于李群 $SO(3)$ 的四元数旋转与空间缩放复合，构建 3D 协方差矩阵 $\Sigma = RSS^T R^T$。
- **EWA Splatting 投影**：实现了基于雅可比矩阵的投影算子，将 3D 椭球映射至 2D 像平面，包含低通滤波补偿。
- **Tile 任务重组**：实现了从 `tile_counts` 到 `point_offsets` 的前缀和扫描、`sort_keys / sort_values` 生成，以及按 tile 的区间识别。
- **动态显存调度架构**：模仿官方 3DGS 风格，利用 `obtain` 线性分配模式和结构化状态块管理显存。
- **高性能内存布局**：
  - 采用 **128 字节对齐**，确保 GPU 合并访问。
  - 使用结构化 `PointState` 管理裸指针，优化 L2 缓存命中率。

### 当前已完成的前向阶段
1. **3D Gaussian 参数展开**：由缩放与四元数构建 3D 协方差。
2. **相机空间变换**：完成世界坐标到 view / clip / NDC 的变换。
3. **2D 椭圆参数计算**：生成屏幕空间的 conic 表示。
4. **Tile 覆盖统计**：为每个高斯估计其覆盖的 tile 数量。
5. **前缀和与复制展开**：使用 CUB 对 tile 计数做 scan，并生成待排序的 key-value 对。
6. **Tile 排序与分段**：按 tile + depth 排序，并识别每个 tile 对应的高斯范围。

### 数学实现细节
1. **相机模型**：采用针孔相机模型，支持分拆的 ViewMatrix 与 ProjMatrix 输入。
2. **矩阵主序**：后端算子采用**行主序**寻址，与 PyTorch 默认逻辑一致。
3. **键值编码方式**：高 32 位编码 tile id，低 32 位编码 depth bit，用于后续排序与 tile range 构建。

### 当前未覆盖的部分
- 完整的光栅化与 alpha blending。
- SH 颜色计算与图像合成。
- backward kernel 与 autograd 支持。
- 数据集、相机参数解析与训练循环。


## 项目结构
```text
.
├── csrc/
│   ├── state.h       # 内存布局与动态分配工具
│   ├── ops.h         # CUDA 主入口声明
│   ├── kernel.cu     # CUDA Kernel、CUB 调度与 run_tiler 主流程
│   └── binding.cpp   # PyTorch 绑定层，仅负责导出 Python 接口
├── setup.py          # 编译脚本
└── .gitignore        # 二进制产物屏蔽清单
```

---

