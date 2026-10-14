from pathlib import Path
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * inputs.size(0)
    return total / max(1, len(loader.dataset))


def save_checkpoint(path, model, optimizer=None, epoch=None, history=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if history is not None:
        payload["history"] = history
    torch.save(payload, path)
