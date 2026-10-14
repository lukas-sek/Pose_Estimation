from argparse import ArgumentParser

import torch

from src.losses.depth import depth_l1
from src.models.unet import UNet
from src.training import train_epoch, set_seed
from train_common import add_common_args, device, finish, loaders, resolve_data


def main():
    args = add_common_args(ArgumentParser(description="Train the D1 UNet depth baseline."), "d1_unet").parse_args()
    set_seed(args.seed)
    _, data = resolve_data(args, resized=False)
    train_loader, _ = loaders(data, args, augment=True)
    model = UNet().to(device())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, depth_l1, device())
        history.append({"epoch": epoch + 1, "train_loss": loss})
        print(f"epoch {epoch + 1:02d}/{args.epochs}: train_l1={loss:.5f}")
    finish(args, model, optimizer, history, args.output)


if __name__ == "__main__":
    main()
