# Mini-Gaussian-Splatting-CUDA

这是一个面向学习的 `3D Gaussian Splatting` CUDA / PyTorch Extension 原型项目。

这个仓库的目标不是直接复刻完整官方工程，而是把 3DGS 中最核心的几何预处理、tile binning、前向渲染与反向传播拆成几个相对清楚的阶段，方便逐步理解底层数据流、显存布局和链式法则。

## 当前进度

目前已经完成并通过测试的部分：

- `preprocess`
  - 3D covariance 构建
  - 2D EWA / conic 投影
  - tile 覆盖计数
  - prefix sum / duplicate / sort / tile range identify
- `render_forward`
  - 基于 `tile_ranges + point_list` 的前向 alpha compositing
- `render_scene`
  - 串联 `preprocess + render_forward`
- `render_backward`
  - render 阶段的反向传播已经完成 GPU 化
  - 当前可以回传：
    - `grad_points2D`
    - `grad_conic_opacity`
    - `grad_colors`


## 实现思路

### Forward

当前 forward 被拆成两个主要阶段：

1. 几何预处理与 binning
   - 3D covariance
   - 2D conic / opacity
   - tile count
   - scan / duplicate / sort / range identify
2. 基于 tile 的前向渲染
   - 按 tile 读取排序后的 Gaussian
   - 对像素执行 alpha compositing

### Backward

当前 backward 的推进顺序是：

1. 先完成 `render_backward`
2. 再继续做 `preprocess_backward`

## 测试

项目当前使用 [test_build.py](D:\Mini_Gaussian_Splatting\test_build.py) 做最小验证。

已覆盖的测试包括：

- 基础 smoke test
- `render_scene` 与 `preprocess + render_forward` 一致性测试
- 空场景 / `L = 0` 边界测试
- 同 tile、多点、不同深度的排序与遮挡测试
- 多 tile 随机场景稳健性测试
- `render_backward` 单高斯参考梯度测试

## 可视化

可以使用 [visualize_render.py](D:\Mini_Gaussian_Splatting\visualize_render.py) 直接导出 forward 结果：

```bash
python visualize_render.py
```

脚本会在 [debug_outputs](D:\Mini_Gaussian_Splatting\debug_outputs) 中生成：

- `render_debug.png`
- `final_t_debug.png`
- `n_contrib_debug.png`

它们分别对应：

- 最终渲染结果
- 每个像素的最终透射率
- 每个像素的有效 Gaussian 贡献数量

## 项目结构

```text
.
├── csrc/
│   ├── state.h
│   ├── ops.h
│   ├── api.h
│   ├── api.cpp
│   ├── binding.cpp
│   ├── kernel.cu
│   └── backward.cu
├── setup.py
├── test_build.py
├── visualize_render.py
└── README.md
```


