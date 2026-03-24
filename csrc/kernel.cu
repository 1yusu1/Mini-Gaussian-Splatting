#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "state.h"

__device__ void computeCov3D(const float3 scale, const float4 q, float* cov3D)
{
    float w = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    float R[9];
    R[0] = 1.f - 2.f * (y * y + z * z);
    R[1] = 2.f * (x * y - w * z);
    R[2] = 2.f * (x * z + w * y);
    R[3] = 2.f * (x * y + w * z);
    R[4] = 1.f - 2.f * (x * x + z * z);
    R[5] = 2.f * (y * z - w * x);
    R[6] = 2.f * (x * z - w * y);
    R[7] = 2.f * (y * z + w * x);
    R[8] = 1.f - 2.f * (x * x + y * y);

    float L[9];
    L[0] = R[0] * scale.x; L[1] = R[1] * scale.y; L[2] = R[2] * scale.z;
    L[3] = R[3] * scale.x; L[4] = R[4] * scale.y; L[5] = R[5] * scale.z;
    L[6] = R[6] * scale.x; L[7] = R[7] * scale.y; L[8] = R[8] * scale.z;

    cov3D[0] = L[0] * L[0] + L[1] * L[1] + L[2] * L[2];
    cov3D[1] = L[0] * L[3] + L[1] * L[4] + L[2] * L[5];
    cov3D[2] = L[0] * L[6] + L[1] * L[7] + L[2] * L[8];
    cov3D[3] = L[3] * L[3] + L[4] * L[4] + L[5] * L[5];
    cov3D[4] = L[3] * L[6] + L[4] * L[7] + L[5] * L[8];
    cov3D[5] = L[6] * L[6] + L[7] * L[7] + L[8] * L[8];
}

__device__ float3 computeCov2D(
    const float3& mean_view,
    const float focal_x,
    const float focal_y,
    const float* cov3D,
    const float* viewmatrix)
{
    float x = mean_view.x;
    float y = mean_view.y;
    float z = mean_view.z;

    float J[6] = {
        focal_x / z, 0.f, -(focal_x * x) / (z * z),
        0.f, focal_y / z, -(focal_y * y) / (z * z)
    };

    float W[9] = {
        viewmatrix[0], viewmatrix[1], viewmatrix[2],
        viewmatrix[4], viewmatrix[5], viewmatrix[6],
        viewmatrix[8], viewmatrix[9], viewmatrix[10]
    };

    float T[6];
    T[0] = J[0] * W[0] + J[1] * W[3] + J[2] * W[6];
    T[1] = J[0] * W[1] + J[1] * W[4] + J[2] * W[7];
    T[2] = J[0] * W[2] + J[1] * W[5] + J[2] * W[8];
    T[3] = J[3] * W[0] + J[4] * W[3] + J[5] * W[6];
    T[4] = J[3] * W[1] + J[4] * W[4] + J[5] * W[7];
    T[5] = J[3] * W[2] + J[4] * W[5] + J[5] * W[8];

    float m11 = cov3D[0], m12 = cov3D[1], m13 = cov3D[2];
    float m22 = cov3D[3], m23 = cov3D[4], m33 = cov3D[5];

    float TS[6];
    TS[0] = T[0] * m11 + T[1] * m12 + T[2] * m13;
    TS[1] = T[0] * m12 + T[1] * m22 + T[2] * m23;
    TS[2] = T[0] * m13 + T[1] * m23 + T[2] * m33;
    TS[3] = T[3] * m11 + T[4] * m12 + T[5] * m13;
    TS[4] = T[3] * m12 + T[4] * m22 + T[5] * m23;
    TS[5] = T[3] * m13 + T[4] * m23 + T[5] * m33;

    float cov_xx = TS[0] * T[0] + TS[1] * T[1] + TS[2] * T[2];
    float cov_xy = TS[0] * T[3] + TS[1] * T[4] + TS[2] * T[5];
    float cov_yy = TS[3] * T[3] + TS[4] * T[4] + TS[5] * T[5];

    return { cov_xx + 0.3f, cov_xy, cov_yy + 0.3f };
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* m)
{
    float3 res;
    res.x = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
    res.y = m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7];
    res.z = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11];
    return res;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* m)
{
    float4 res;
    res.x = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
    res.y = m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7];
    res.z = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11];
    res.w = m[12] * p.x + m[13] * p.y + m[14] * p.z + m[15];
    return res;
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
    return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

__global__ void preprocess_points_kernel(
    int P,
    const float3* points3D,
    const float* viewmatrix,
    const float* projmatrix,
    const float focal_x,
    const float focal_y,
    int W, int H,
    float radius,
    PointState s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    float3 p_origin = points3D[idx];
    float3 p_view = transformPoint4x3(p_origin, viewmatrix);
    if (p_view.z < 0.2f) return;
    float4 p_hom = transformPoint4x4(p_origin, projmatrix);

    float p_w = 1.0f / (p_hom.w + 1e-7f);
    float3 p_ndc = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
    s.points2D[idx] = { ndc2Pix(p_ndc.x, W), ndc2Pix(p_ndc.y, H) };
    float2 p2d = s.points2D[idx];
    s.depths[idx] = p_view.z;

    int x_min = max(0, (int)floorf((p2d.x - radius) / 16.0f));
    int x_max = min((W + 15) / 16, (int)ceilf((p2d.x + radius) / 16.0f));
    int y_min = max(0, (int)floorf((p2d.y - radius) / 16.0f));
    int y_max = min((H + 15) / 16, (int)ceilf((p2d.y + radius) / 16.0f));
    s.tile_counts[idx] = max(0, x_max - x_min) * max(0, y_max - y_min);

    computeCov3D(s.scales[idx], s.quat[idx], s.cov3D + (idx * 6));
    float3 cov2D = computeCov2D(p_view, focal_x, focal_y, s.cov3D + idx * 6, viewmatrix);
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det <= 0.0f) return;
    float det_inv = 1.f / det;
    s.conic[idx] = { cov2D.z * det_inv, -cov2D.y * det_inv, cov2D.x * det_inv };
}

void preprocess_points(int P, const float3* points3D, const float* viewmatrix, const float* projmatrix,
const float focal_x, const float focal_y, int W, int H, float r, PointState s) {
    preprocess_points_kernel<<<(P+255)/256, 256>>>(P, points3D, viewmatrix, projmatrix, focal_x, focal_y, W, H, r, s);
}
