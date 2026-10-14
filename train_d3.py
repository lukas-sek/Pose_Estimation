from argparse import ArgumentParser

import torch

from src.losses.depth import depth_l1, joint_l1, soft_argmax_2d
from src.models.multitask import MultiTaskDepthModel
from src.training import set_seed
from train_common import add_common_args, device, finish, loaders, resolve_data


def main():
    parser = add_common_args(ArgumentParser(description="Train D3 with depth and projected-joint supervision."), "d3_losses")
    parser.add_argument("--joint-cache", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--joint-weight", type=float, default=1.0)
    args = parser.parse_args()
    set_seed(args.seed)
    _, data = resolve_data(args)
    train_loader, _ = loaders(data, args, imagenet=True, joint_cache=args.joint_cache)
    model = MultiTaskDepthModel(args.patch_size, not args.no_pretrained, args.img_size).to(device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for image, depth, joints, visible in train_loader:
            image, depth = image.to(device()), depth.to(device())
            joints, visible = joints.to(device()), visible.to(device())
            optimizer.zero_grad(set_to_none=True)
            pred_depth, heatmaps = model(image)
            predicted_joints = soft_argmax_2d(heatmaps)
            loss = depth_l1(pred_depth, depth) + args.joint_weight * joint_l1(predicted_joints, joints, visible, args.img_size)
            loss.backward()
            optimizer.step()
            running += loss.item() * image.size(0)
        value = running / max(1, len(train_loader.dataset))
        history.append({"epoch": epoch + 1, "train_loss": value})
        print(f"epoch {epoch + 1:02d}/{args.epochs}: train_loss={value:.5f}")
    finish(args, model, optimizer, history, args.output)


if __name__ == "__main__":
    main()
