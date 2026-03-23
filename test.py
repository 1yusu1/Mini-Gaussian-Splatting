import torch
import my_gs_ops

quats = torch.randn(100, 4, device="cuda")
quats = torch.nn.functional.normalize(quats, dim=1)
rots = torch.zeros(100, 9, device="cuda")

my_gs_ops.quat_to_rot(quats, rots)

print("Success! Rotation matrix shape:", rots.shape)