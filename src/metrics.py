import torch
import torch.nn.functional as F


def rmse(prediction, target):
    mask = target > 0
    return torch.sqrt(F.mse_loss(prediction[mask], target[mask])) if mask.any() else prediction.new_zeros(())


def depth_to_normals(depth, eps=1e-6):
    dx = F.pad(depth[:, :, :, 2:] - depth[:, :, :, :-2], [1, 1, 0, 0])
    dy = F.pad(depth[:, :, 2:, :] - depth[:, :, :-2, :], [0, 0, 1, 1])
    normals = torch.cat([-dx, -dy, torch.ones_like(dx)], dim=1)
    return normals / (torch.norm(normals, dim=1, keepdim=True) + eps)


def mean_angular_error(prediction, target):
    mask = target[:, 0] > 0
    cosine = (depth_to_normals(prediction) * depth_to_normals(target)).sum(dim=1).clamp(-1, 1)
    angle = torch.acos(cosine) * (180.0 / torch.pi)
    return angle[mask].mean() if mask.any() else prediction.new_zeros(())
