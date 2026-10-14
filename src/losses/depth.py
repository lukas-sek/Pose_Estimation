import torch
import torch.nn.functional as F

from src.metrics import depth_to_normals


def depth_l1(prediction, target):
    mask = target > 0
    return F.l1_loss(prediction[mask], target[mask]) if mask.any() else prediction.new_zeros(())


def normal_consistency(prediction, target):
    mask = target[:, 0] > 0
    if not mask.any():
        return prediction.new_zeros(())
    pred_normals = depth_to_normals(prediction).permute(0, 2, 3, 1)
    target_normals = depth_to_normals(target).permute(0, 2, 3, 1)
    return F.l1_loss(pred_normals[mask], target_normals[mask])


def scale_invariant(prediction, target, coefficient=0.5):
    mask = target > 0
    if not mask.any():
        return prediction.new_zeros(())
    difference = torch.log(prediction[mask] + 1e-6) - torch.log(target[mask] + 1e-6)
    return difference.square().mean() - coefficient * difference.mean().square()


def ranking(prediction, target, pairs=1024, margin=0.05):
    mask = target[:, 0] > 0
    pred_flat = prediction[:, 0].flatten(1)
    target_flat = target[:, 0].flatten(1)
    mask_flat = mask.flatten(1)
    losses = []
    for batch_index in range(prediction.shape[0]):
        valid = torch.where(mask_flat[batch_index])[0]
        if valid.numel() < 2:
            continue
        choices = valid[torch.randint(valid.numel(), (pairs * 2,), device=prediction.device)]
        first, second = choices[:pairs], choices[pairs:]
        sign = torch.sign(target_flat[batch_index, first] - target_flat[batch_index, second])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        losses.append(F.relu(margin - sign * (pred_flat[batch_index, first] - pred_flat[batch_index, second])).mean())
    return torch.stack(losses).mean() if losses else prediction.new_zeros(())


def soft_argmax_2d(heatmaps, beta=80.0):
    _, _, height, width = heatmaps.shape
    probabilities = F.softmax(heatmaps.flatten(2) * beta, dim=-1)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype),
        torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype), indexing="ij",
    )
    grid = torch.stack((xs.flatten(), ys.flatten())).view(1, 1, 2, -1)
    return (probabilities.unsqueeze(2) * grid).sum(dim=-1)


def joint_l1(predicted_xy, target_xy, visible, img_size=256, normalized=True):
    mask = visible.unsqueeze(-1)
    if mask.sum() < 1:
        return predicted_xy.new_zeros(())
    value = (torch.abs(predicted_xy - target_xy) * mask).sum() / mask.sum()
    return value / img_size if normalized else value
