from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import split_names
from src.paths import ProjectPaths
from src.training import save_checkpoint


def add_common_args(parser: ArgumentParser, task: str):
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(task=task)
    return parser


def resolve_data(args, resized=True):
    paths = ProjectPaths.from_root(args.root)
    data = args.data or (paths.resized(args.img_size) if resized else paths.base_data)
    return paths, Path(data)


def loaders(data, args, imagenet=False, augment=False, joint_cache=None):
    from src.data import Cloth3DDataset, Cloth3DJointDataset
    names = split_names(data)
    dataset_type = Cloth3DJointDataset if joint_cache else Cloth3DDataset
    common = {"root": data, "img_size": args.img_size, "imagenet": imagenet}
    if joint_cache:
        train_set = dataset_type(names["train"], joint_cache=joint_cache, **common)
        valid_set = dataset_type(names["validation"], joint_cache=joint_cache, **common)
    else:
        train_set = dataset_type(names["train"], augment=augment, **common)
        valid_set = dataset_type(names["validation"], augment=False, **common)
    kwargs = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": torch.cuda.is_available()}
    return DataLoader(train_set, shuffle=True, **kwargs), DataLoader(valid_set, shuffle=False, **kwargs)


def finish(args, model, optimizer, history, output):
    output = output or Path(args.root) / "runs" / args.task / "checkpoints" / "last.pt"
    save_checkpoint(output, model, optimizer, args.epochs, history)
    print(f"Saved checkpoint: {output}")


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
