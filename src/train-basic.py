"""
python src/train-basic.py \
  --data parotid_PDDCA+deepmind \
  --output checkpoints \
  --epochs 200 \
  --batch_size 2 \
  --lr 1e-4 \
  --base_filters 64 \
  --num_workers 4 \
  --model_name cbam \
  2>&1 | tee checkpoints/cbam_train.log

# resume an interrupted run:
python src/train-basic.py \
  --data parotid_PDDCA+deepmind \
  --output checkpoints \
  --epochs 200 \
  --batch_size 2 \
  --lr 1e-4 \
  --base_filters 64 \
  --num_workers 4 \
  --model_name cbam \
  --resume checkpoints/20260612_151650_n52_lr0.0001_f64/latest.pt \
  2>&1 | tee checkpoints/cbam_train.log
"""

import os
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PDDCADataset, make_splits
from unet_basic import UNet3D
from unet_cbam import UNet3D_CBAM


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(pred, target, smooth=1e-5):
    """
    Soft Dice loss averaged over batch and classes.
    pred/target: (B, 1, D, H, W), pred already sigmoid-ed
    """
    pred_flat   = pred.view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.view(target.shape[0], target.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(-1)
    dsc = (2 * intersection + smooth) / (
        pred_flat.sum(-1) + target_flat.sum(-1) + smooth
    )
    return 1 - dsc.mean()


def combined_loss(pred, target):
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return 0.2 * bce + 0.8 * dice


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dsc(pred, target, threshold=0.5, smooth=1e-5):
    """
    Hard Dice averaged over batch.
    Returns dsc as Python float.
    """
    pred_bin    = (pred > threshold).float()
    pred_flat   = pred_bin.view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.view(target.shape[0], target.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(-1)
    dsc = (2 * intersection + smooth) / (
        pred_flat.sum(-1) + target_flat.sum(-1) + smooth
    )
    # dsc shape: (B, 1) — average over batch
    dsc = dsc.mean(0)
    return dsc[0].item()


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
    dsc_all = []

    for scans, masks, _ in loader:
        scans = scans.to(device)
        masks = masks.to(device)

        preds = model(scans)
        loss  = combined_loss(preds, masks)
        total_loss += loss.item()

        dsc_all.append(compute_dsc(preds, masks))

    return total_loss / len(loader), np.mean(dsc_all)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    data_root:    str,
    model_name:   str,
    output_dir:   str   = "./checkpoints",
    epochs:       int   = 200,
    batch_size:   int   = 2,
    lr:           float = 1e-4,
    base_filters: int   = 64,
    val_size:     int   = 5,
    test_size:    int   = 5,
    seed:         int   = 42,
    num_workers:  int   = 2,
    resume:       str   = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # --- data ---
    train_ds, val_ds, _ = make_splits(
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

    # --- output dir and file names ---
    if resume:
        # derive the base name from the latest.pt path
        # e.g. checkpoints/cbam_20260612_151650_n52_lr0.0001_f64_latest.pt
        # → best path: checkpoints/cbam_20260612_151650_n52_lr0.0001_f64_best.pt
        base = os.path.basename(resume).replace('_latest.pt', '')
        run_prefix = os.path.join(output_dir, base)
        print(f"Resuming run: {run_prefix}")
    else:
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        n_train    = len(train_ds)
        run_name   = f"{model_name}_{timestamp}_n{n_train}_lr{lr}_f{base_filters}"
        run_prefix = os.path.join(output_dir, run_name)

    best_path   = f"{run_prefix}_best.pt"
    latest_path = f"{run_prefix}_latest.pt"

    # --- model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_name == "unet":
        model = UNet3D(
            n_channels=1, n_classes=1,
            input_shape=(48, 208, 272),
            bilinear=False, base_filters=base_filters,
        ).to(device)
    else:
        model = UNet3D_CBAM(
            n_channels=1, n_classes=1,
            input_shape=(48, 208, 272),
            bilinear=False, base_filters=base_filters,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- optimizer + scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # --- resume ---
    start_epoch = 1
    best_dsc    = 0.0
    best_epoch  = 0

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

        # recover best_dsc from best.pt if it exists
        if os.path.exists(best_path):
            best_ckpt  = torch.load(best_path, map_location=device, weights_only=False)
            best_dsc   = best_ckpt['mean_dsc']
            best_epoch = best_ckpt['epoch']
            print(f"Best DSC so far: {best_dsc:.4f} at epoch {best_epoch}")

    # --- training ---
    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} "
          f"{'DSC':>10} {'Time':>8}")
    print("-" * 55)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, dsc = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
              f"{dsc:>10.4f} {elapsed:>7.1f}s")

        # save best checkpoint
        if dsc > best_dsc:
            best_dsc   = dsc
            best_epoch = epoch
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "mean_dsc":    dsc,
            }, best_path)

        # save latest checkpoint for resume
        torch.save({
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "optim_state":      optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
        }, latest_path)

    print(f"\nBest DSC: {best_dsc:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {best_path}")

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=str,   required=True,  help="Path to PDDCA root")
    parser.add_argument("--model_name",   type=str,   required=True,  help="unet or cbam")
    parser.add_argument("--output",       type=str,   default="./checkpoints")
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=2)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--base_filters", type=int,   default=64)
    parser.add_argument("--val_size",     type=int,   default=5)
    parser.add_argument("--test_size",    type=int,   default=5)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to latest.pt to resume an interrupted run")
    args = parser.parse_args()

    train(
        data_root=args.data,
        model_name=args.model_name,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_filters=args.base_filters,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
        resume=args.resume,
    )