"""Final D4 fine-tuning entry point.

D4 reuses the D3 multi-task architecture and accepts a D3 checkpoint so the
final experiment is explicit and reproducible rather than notebook state.
"""

from argparse import ArgumentParser

import torch

from src.losses.depth import depth_l1
from src.models.multitask import MultiTaskDepthModel
from src.training import set_seed
from train_common import add_common_args, device, finish, loaders, resolve_data


def main():
    parser = add_common_args(ArgumentParser(description="Fine-tune the selected D4 depth model."), "d4_final")
    parser.add_argument("--joint-cache", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    _, data = resolve_data(args)
    train_loader, _ = loaders(data, args, imagenet=True, joint_cache=args.joint_cache)
    model = MultiTaskDepthModel(args.patch_size, not args.no_pretrained, args.img_size).to(device())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for image, depth, _, _ in train_loader:
            image, depth = image.to(device()), depth.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(image)
            loss = depth_l1(prediction, depth)
            loss.backward()
            optimizer.step()
            running += loss.item() * image.size(0)
        value = running / max(1, len(train_loader.dataset))
        history.append({"epoch": epoch + 1, "train_loss": value})
        print(f"epoch {epoch + 1:02d}/{args.epochs}: train_l1={value:.5f}")
    finish(args, model, optimizer, history, args.output)


if __name__ == "__main__":
    main()
