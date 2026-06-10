import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PDDCADataset, make_splits
# from unet_basic import UNet3D
from unet_cbam import UNet3D_CBAM


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(pred, target, smooth=1e-5):
    """
    Soft Dice loss averaged over batch and classes.
    pred/target: (B, 2, D, H, W), pred already sigmoid-ed
    """
    pred_flat   = pred.view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.view(target.shape[0], target.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(-1)
    dsc = (2 * intersection + smooth) / (
        pred_flat.sum(-1) + target_flat.sum(-1) + smooth
    )
    return 1 - dsc.mean()


# def combined_loss(pred, target):
#     """Dice + BCE — Dice handles class imbalance, BCE stabilises gradients."""
#     bce  = F.binary_cross_entropy(pred, target)
#     dice = dice_loss(pred, target)
#     return bce + dice

def combined_loss(pred, target):
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return 0.2 * bce + 0.8 * dice  # lean heavily on Dice

# def combined_loss(pred, target):
#     return dice_loss(pred, target) # dice loss only


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dsc(pred, target, threshold=0.5, smooth=1e-5):
    """
    Hard Dice per class, averaged over batch.
    Returns (dsc_left, dsc_right) as Python floats.
    """
    pred_bin    = (pred > threshold).float()
    pred_flat   = pred_bin.view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.view(target.shape[0], target.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(-1)
    dsc = (2 * intersection + smooth) / (
        pred_flat.sum(-1) + target_flat.sum(-1) + smooth
    )
    # dsc shape: (B, 2) — average over batch
    dsc = dsc.mean(0)
    return dsc[0].item(), dsc[1].item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for scans, masks, _ in loader:
        scans = scans.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(scans)
        loss  = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    dsc_left_all  = []
    dsc_right_all = []

    for scans, masks, _ in loader:
        scans = scans.to(device)
        masks = masks.to(device)

        preds = model(scans)
        loss  = combined_loss(preds, masks)
        total_loss += loss.item()

        dsc_l, dsc_r = compute_dsc(preds, masks)
        dsc_left_all.append(dsc_l)
        dsc_right_all.append(dsc_r)

    return (
        total_loss / len(loader),
        np.mean(dsc_left_all),
        np.mean(dsc_right_all),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    data_root:    str,
    output_dir:   str  = "./checkpoints",
    epochs:       int  = 200,
    batch_size:   int  = 2,
    lr:           float = 1e-4,
    base_filters: int  = 32,
    val_size:     int  = 5,
    test_size:    int  = 5,
    seed:         int  = 42,
    num_workers:  int  = 2,
):
    os.makedirs(output_dir, exist_ok=True)

    # --- data ---
    train_ds, val_ds, test_ds = make_splits(
        data_root, val_size=val_size, test_size=test_size, seed=seed
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    import datetime

    # replace the checkpoint naming lines with this
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    n_train = len(train_ds)
    run_name = f"{timestamp}_n{n_train}_lr{lr}_f{base_filters}"
    output_dir = os.path.join(output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model = UNet3D(
    #     n_channels=1,
    #     n_classes=2,
    #     input_shape=(48, 208, 272),
    #     bilinear=False,
    #     base_filters=base_filters,
    # ).to(device)

    model = UNet3D_CBAM(
        n_channels=1,
        n_classes=2,
        input_shape=(48, 208, 272),
        bilinear=False,
        base_filters=base_filters,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- optimizer + scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # --- training ---
    best_dsc  = 0.0
    best_epoch = 0

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} "
          f"{'DSC Left':>10} {'DSC Right':>11} {'Time':>8}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, dsc_l, dsc_r = validate(model, val_loader, device)
        scheduler.step()

        mean_dsc = (dsc_l + dsc_r) / 2
        elapsed  = time.time() - t0

        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
              f"{dsc_l:>10.4f} {dsc_r:>11.4f} {elapsed:>7.1f}s")

        # save best checkpoint
        if mean_dsc > best_dsc:
            best_dsc   = mean_dsc
            best_epoch = epoch
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "dsc_left":    dsc_l,
                "dsc_right":   dsc_r,
                "mean_dsc":    mean_dsc,
            }, os.path.join(output_dir, "best_model.pt"))

        # save latest checkpoint (safe resume)
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, os.path.join(output_dir, "latest.pt"))

    print(f"\nBest mean DSC: {best_dsc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to {output_dir}/best_model.pt")

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=str,   required=True,  help="Path to PDDCA root")
    parser.add_argument("--output",       type=str,   default="./checkpoints")
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=2)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--base_filters", type=int,   default=32)
    parser.add_argument("--val_size",     type=int,   default=5)
    parser.add_argument("--test_size",    type=int,   default=5)
    parser.add_argument("--num_workers",  type=int,   default=2)
    args = parser.parse_args()

    train(
        data_root=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_filters=args.base_filters,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
    )