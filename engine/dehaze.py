from typing import Dict
import torch
from torch.nn import L1Loss
from torch import amp
from tqdm import tqdm

from kornia.losses import SSIMLoss
from kornia.metrics import ssim, psnr

from loss.perceptual import VGGPerceptualLoss
from utils.common_utils import *


def train_one_epoch(hparams, model, scaler, optimizer, scheduler, train_loader):
    model.train()
    loss_rec = MetricRecorder()

    device = hparams["train"]["device"]

    l1 = L1Loss().to(device)
    ssim_loss = SSIMLoss(5).to(device)
    perceptual = VGGPerceptualLoss().to(device)

    for src, tgt in tqdm(train_loader, ncols=120):
        src, tgt = src.to(device), tgt.to(device)

        with amp.autocast(
            device_type=hparams["train"]["device"], enabled=hparams["train"]["use_amp"]
        ):
            outputs = model(src)
            out = outputs[2]

            loss = l1(out, tgt) + 0.3 * ssim_loss(out, tgt) + 0.2 * perceptual(out, tgt)

        optimizer.zero_grad()

        if hparams["train"]["use_amp"]:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_rec.update(loss.item())

    return {"train_loss": loss_rec.avg, "lr": optimizer.param_groups[0]["lr"]}


def valid(hparams, model, valid_loader):
    model.eval()

    loss_rec = MetricRecorder()
    ssim_rec = MetricRecorder()
    psnr_rec = MetricRecorder()

    device = hparams["train"]["device"]

    l1 = L1Loss().to(device)
    ssim_loss = SSIMLoss(5).to(device)
    perceptual = VGGPerceptualLoss().to(device)

    for i, (src, tgt) in enumerate(valid_loader):
        src, tgt = src.to(device), tgt.to(device)

        with torch.no_grad():
            outputs = model(src)
            out = outputs[2].clamp(0, 1)

        loss = l1(out, tgt) + 0.3 * ssim_loss(out, tgt) + 0.2 * perceptual(out, tgt)

        loss_rec.update(loss.item())
        ssim_rec.update(ssim(out, tgt, 5).mean().item())
        psnr_rec.update(psnr(out, tgt, 1).item())

        if i % 5 == 0:
            save_pics(hparams, src, tgt, out)

    return {
        "valid_loss": loss_rec.avg,
        "ssim": ssim_rec.avg,
        "psnr": psnr_rec.avg,
    }


def train(
    hparams: Dict,
    model,
    optimizer,
    scaler,
    scheduler,
    logger,
    train_loader,
    valid_loader,
    stage_index: int,
):

    set_all_seed(hparams["train"]["seed"])
    make_all_dirs(hparams)

    best_metric = {
        "ssim": {"value": 0, "epoch": 0},
        "psnr": {"value": 0, "epoch": 0},
    }

    start_epoch = 1
    print("==========>Start Training<==========")

    max_epochs = sum(hparams["train"]["stage_epochs"][: stage_index + 1])

    for epoch in range(start_epoch, max_epochs + 1):

        # ---------------- TRAIN ----------------
        train_result = train_one_epoch(
            hparams, model, scaler, optimizer, scheduler, train_loader
        )
        logger.log_multi_scaler(train_result, epoch)

        scheduler.step(epoch)

        # ---------------- VALID ----------------
        valid_result = None  # <-- IMPORTANT FIX

        if epoch % hparams["train"]["valid_frequency"] == 0:
            valid_result = valid(hparams, model, valid_loader)
            logger.log_multi_scaler(valid_result, epoch)

            # -------- SAVE BEST MODELS --------
            if valid_result["ssim"] > best_metric["ssim"]["value"]:
                best_metric["ssim"] = {
                    "value": valid_result["ssim"],
                    "epoch": epoch,
                }
                save_all(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    hparams,
                    best_metric,
                    "best_ssim",
                )

            if valid_result["psnr"] > best_metric["psnr"]["value"]:
                best_metric["psnr"] = {
                    "value": valid_result["psnr"],
                    "epoch": epoch,
                }
                save_all(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    hparams,
                    best_metric,
                    "best_psnr",
                )

        # ---------------- SAVE LAST ----------------
        save_all(
            epoch, model, optimizer, scheduler, scaler, hparams, best_metric, "last"
        )

        # ---------------- PRINT ----------------
        print_epoch_result(train_result, valid_result, epoch)
