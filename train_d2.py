from argparse import ArgumentParser

import torch

from src.losses.depth import depth_l1
from src.models.deit import BilinearDecoder, DepthModel, DeiTEncoder, FPNDecoder
from src.training import train_epoch, set_seed
from train_common import add_common_args, device, finish, loaders, resolve_data


def main():
    parser = add_common_args(ArgumentParser(description="Train the D2 DeiT depth model."), "d2_deit")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--decoder", choices=("bilinear", "fpn"), default="fpn")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    _, data = resolve_data(args)
    train_loader, _ = loaders(data, args, imagenet=True)
    encoder = DeiTEncoder(args.patch_size, not args.no_pretrained, args.img_size)
    decoder_type = BilinearDecoder if args.decoder == "bilinear" else FPNDecoder
    model = DepthModel(encoder, decoder_type(encoder.out_channels, args.img_size)).to(device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, depth_l1, device())
        history.append({"epoch": epoch + 1, "train_loss": loss})
        print(f"epoch {epoch + 1:02d}/{args.epochs}: train_l1={loss:.5f}")
    finish(args, model, optimizer, history, args.output)


if __name__ == "__main__":
    main()
