import torch

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
CNS, YNS = 0, 1  # centerness and yawness indices in quality
YAW = 6  # decoded

__all__ = [
    'X', 'Y', 'Z', 'W', 'L', 'H', 'SIN_YAW', 'COS_YAW', 'VX', 'VY', 'VZ',
    'CNS', 'YNS', 'YAW',
    'encode_gt_boxes',
]


def encode_gt_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Encode GT boxes from decoded 9-dim [x,y,z,w,l,h,yaw,vx,vy] to the
    11-dim anchor format [X,Y,Z,log_W,log_L,log_H,SIN_YAW,COS_YAW,VX,VY,VZ]."""
    xyz = boxes[:, :3]
    wlh = boxes[:, 3:6].clamp(min=1e-3).log()
    sin_yaw = torch.sin(boxes[:, YAW:YAW + 1])
    cos_yaw = torch.cos(boxes[:, YAW:YAW + 1])
    vel = (boxes[:, 7:9] if boxes.shape[-1] >= 9
           else boxes.new_zeros(len(boxes), 2))
    vz = boxes.new_zeros(len(boxes), 1)
    return torch.cat([xyz, wlh, sin_yaw, cos_yaw, vel, vz], dim=-1)
