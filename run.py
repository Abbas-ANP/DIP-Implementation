import os
import torch
from engine.dehaze import train
from data.uieb import UIEBTrain, UIEBValid
from torch.utils.data import DataLoader
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
from model.base import CLCC
from utils.common_utils import *
from torch.amp import GradScaler


def configuration_dataloader(hparams, stage_index):
    train_dataset = UIEBTrain(
        folder=hparams["data"]["train_path"],
        size=hparams["data"]["train_img_size"][stage_index],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["data"]["train_batch_size"][stage_index],
        shuffle=True,
        num_workers=hparams["data"]["num_workers"],
        pin_memory=hparams["data"]["pin_memory"],
    )

    valid_dataset = UIEBValid(folder=hparams["data"]["valid_path"], size=256)

    valid_loader = DataLoader(valid_dataset, batch_size=1)

    return train_loader, valid_loader


def configuration_optimizer(model, hparams):
    optimizer = AdamW(model.parameters(), lr=hparams["optim"]["lr_init"])

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=sum(hparams["train"]["stage_epochs"]),
        lr_min=hparams["optim"]["lr_min"],
    )

    return optimizer, scheduler


if __name__ == "__main__":
    args = parse_yaml("./config.yaml")

    device = args["train"]["device"]

    model = CLCC(64, 3, 3).to(device)

    scaler = GradScaler(enabled=args["train"]["use_amp"])

    logger = Logger("./log")

    optimizer, scheduler = configuration_optimizer(model, args)

    for i in range(len(args["train"]["stage_epochs"])):
        train_loader, valid_loader = configuration_dataloader(args, i)

        train(
            args,
            model,
            optimizer,
            scaler,
            scheduler,
            logger,
            train_loader,
            valid_loader,
            i,
        )
