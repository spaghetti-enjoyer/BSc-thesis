"""
evaluate.py — run both trained models on the held-out test set and
produce a per-patient DSC table plus summary statistics.

Usage:
    python unets/evaluate.py \
        --data  BSc-Thesis-Datasets/parotid_PDDCA+deepmind \
        --basic unet_20260609_150324_n52_lr0.0001_f64.pt \
        --cbam  cbam_20260609_162153_n52_lr0.0001_f64.pt
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import make_splits
from unet_basic import UNet3D
from unet_cbam  import UNet3D_CBAM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_basic(path, device):
    model = UNet3D(
        n_channels=1, n_classes=2,
        input_shape=(48, 208, 272),
        bilinear=False, base_filters=64,
    ).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def load_cbam(path, device):
    model = UNet3D_CBAM(
        n_channels=1, n_classes=2,
        input_shape=(48, 208, 272),
        bilinear=False, base_filters=64,
    ).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


@torch.no_grad()
def dice(pred, target, threshold=0.5, smooth=1e-5):
    """Hard DSC per class. Returns (dsc_left, dsc_right)."""
    pred_bin    = (pred > threshold).float()
    pred_flat   = pred_bin.view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.view(target.shape[0], target.shape[1], -1)
    intersection = (pred_flat * target_flat).sum(-1)
    dsc = (2 * intersection + smooth) / (
        pred_flat.sum(-1) + target_flat.sum(-1) + smooth
    )
    return dsc[0, 0].item(), dsc[0, 1].item()


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Returns list of dicts:
        { patient, dsc_left, dsc_right, mean_dsc }
    """
    results = []
    for scan, mask, pids in loader:
        print("loop")
        scan = scan.to(device)
        mask = mask.to(device)
        print("time to pred")
        pred = model(scan)
        print("mid")
        dsc_l, dsc_r = dice(pred, mask)
        results.append({
            'patient':  pids[0],
            'dsc_left':  round(dsc_l, 4),
            'dsc_right': round(dsc_r, 4),
            'mean_dsc':  round((dsc_l + dsc_r) / 2, 4),
        })
        print(f"results: {results}")
    return results


def print_table(results, model_name):
    print(f"\n{'─'*55}")
    print(f"  {model_name}")
    print(f"{'─'*55}")
    print(f"  {'Patient':<20} {'DSC Left':>10} {'DSC Right':>10} {'Mean':>8}")
    print(f"{'─'*55}")
    for r in results:
        print(f"  {r['patient']:<20} {r['dsc_left']:>10.4f} {r['dsc_right']:>10.4f} {r['mean_dsc']:>8.4f}")
    print(f"{'─'*55}")
    means = [r['mean_dsc']  for r in results]
    lefts = [r['dsc_left']  for r in results]
    rights= [r['dsc_right'] for r in results]
    print(f"  {'Mean':<20} {np.mean(lefts):>10.4f} {np.mean(rights):>10.4f} {np.mean(means):>8.4f}")
    print(f"  {'Std':<20} {np.std(lefts):>10.4f} {np.std(rights):>10.4f} {np.std(means):>8.4f}")
    print(f"  {'Min':<20} {np.min(lefts):>10.4f} {np.min(rights):>10.4f} {np.min(means):>8.4f}")
    print(f"  {'Max':<20} {np.max(lefts):>10.4f} {np.max(rights):>10.4f} {np.max(means):>8.4f}")
    print(f"{'─'*55}")
    return np.mean(means)


def print_comparison(basic_results, cbam_results):
    print(f"\n{'─'*55}")
    print(f"  Head-to-head comparison (mean DSC per patient)")
    print(f"{'─'*55}")
    print(f"  {'Patient':<20} {'Vanilla':>10} {'CBAM':>10} {'Δ':>8}")
    print(f"{'─'*55}")
    deltas = []
    for b, c in zip(basic_results, cbam_results):
        assert b['patient'] == c['patient'], "Patient order mismatch"
        delta = c['mean_dsc'] - b['mean_dsc']
        deltas.append(delta)
        sign = '+' if delta >= 0 else ''
        print(f"  {b['patient']:<20} {b['mean_dsc']:>10.4f} {c['mean_dsc']:>10.4f} {sign}{delta:>7.4f}")
    print(f"{'─'*55}")
    mean_delta = np.mean(deltas)
    sign = '+' if mean_delta >= 0 else ''
    print(f"  {'Mean Δ':<20} {'':>10} {'':>10} {sign}{mean_delta:>7.4f}")
    winner = "CBAM" if mean_delta > 0 else "Vanilla" if mean_delta < 0 else "Tie"
    print(f"\n  Winner: {winner}")
    print(f"{'─'*55}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str, required=True)
    parser.add_argument("--basic",      type=str, required=True, help="Path to vanilla best_model.pt")
    parser.add_argument("--cbam",       type=str, required=True, help="Path to CBAM best_model.pt")
    parser.add_argument("--val_size",   type=int, default=5)
    parser.add_argument("--test_size",  type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    print(f"Device: {device}")

    # --- same split as training ---
    _, _, test_ds = make_splits(
        args.data,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"\nTest patients ({len(test_ds)}):")
    for _, _, pid in test_loader:
        print(f"  {pid[0]}")

    # --- evaluate vanilla first, then free it ---
    print("\nEvaluating vanilla U-Net...")
    basic_model   = load_basic(args.basic, device)
    print("hello")
    basic_results = evaluate(basic_model, test_loader, device)
    del basic_model
    if device.type == 'mps':
        torch.mps.empty_cache()

    # --- now load CBAM ---
    print("Evaluating CBAM U-Net...")
    cbam_model    = load_cbam(args.cbam, device)
    cbam_results  = evaluate(cbam_model, test_loader, device)
    del cbam_model
    if device.type == 'mps':
        torch.mps.empty_cache()

    # --- load and evaluate ---
    # print("\nEvaluating vanilla U-Net...")
    # basic_model   = load_basic(args.basic, device)
    # basic_results = evaluate(basic_model, test_loader, device)

    # print("Evaluating CBAM U-Net...")
    # cbam_model    = load_cbam(args.cbam, device)
    # cbam_results  = evaluate(cbam_model, test_loader, device)

    # --- print results ---
    basic_mean = print_table(basic_results, "Vanilla U-Net")
    cbam_mean  = print_table(cbam_results,  "CBAM U-Net")
    print_comparison(basic_results, cbam_results)