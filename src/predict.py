"""
infer.py — run segmentation and GBP saliency maps for a single patient.

Usage:
    python src/predict.py \
        --patient  0522c0416 \
        --data     parotid_PDDCA+deepmind \
        --basic    models/unet_20260609_150324_n52_lr0.0001_f64.pt \
        --cbam     models/cbam_20260609_162153_n52_lr0.0001_f64.pt \
        --model    both \
        --output   results/

Output per model (e.g. for 'both'):
    results/0522c0001/
        basic_pred_left.nrrd        # predicted mask, left parotid
        basic_pred_right.nrrd       # predicted mask, right parotid
        basic_saliency_left.nrrd    # GBP saliency map, left parotid  (float32, 0-1)
        basic_saliency_right.nrrd   # GBP saliency map, right parotid (float32, 0-1)
        cbam_pred_left.nrrd
        cbam_pred_right.nrrd
        cbam_saliency_left.nrrd
        cbam_saliency_right.nrrd
"""

import argparse
import os
import numpy as np
import nrrd
import torch

from dataset import PDDCADataset, normalize
from unet_basic import UNet3D, GuidedBackprop
from unet_cbam  import UNet3D_CBAM, GuidedBackprop as GuidedBackpropCBAM


# ---------------------------------------------------------------------------
# Model loading
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
    print(f"Loaded vanilla U-Net from {path} (epoch {ckpt['epoch']}, DSC {ckpt['mean_dsc']:.4f})")
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
    print(f"Loaded CBAM U-Net from {path} (epoch {ckpt['epoch']}, DSC {ckpt['mean_dsc']:.4f})")
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_patient(data_root, patient_id):
    """
    Load, transpose and normalise a single patient's scan.
    Returns:
        scan_tensor: (1, 1, D, H, W) float32 torch tensor
        raw_scan:    (D, H, W) float32 numpy array (normalised)
    """
    scan_path  = os.path.join(data_root, patient_id, 'img.nrrd')
    left_path  = os.path.join(data_root, patient_id, 'structures', 'Parotid_L.nrrd')
    right_path = os.path.join(data_root, patient_id, 'structures', 'Parotid_R.nrrd')

    for p in [scan_path, left_path, right_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    scan,  _ = nrrd.read(scan_path)
    left,  _ = nrrd.read(left_path)
    right, _ = nrrd.read(right_path)

    # (W, H, D) -> (D, H, W)
    scan  = scan.transpose(2, 0, 1).astype(np.float32)
    left  = left.transpose(2, 0, 1).astype(np.float32)
    right = right.transpose(2, 0, 1).astype(np.float32)

    scan = normalize(scan)

    scan_tensor = torch.from_numpy(scan[np.newaxis, np.newaxis])  # (1,1,D,H,W)

    gt_mask = np.stack([left, right], axis=0)  # (2, D, H, W)

    return scan_tensor, scan, gt_mask


# ---------------------------------------------------------------------------
# Inference + GBP
# ---------------------------------------------------------------------------

def run_inference(model, scan_tensor, device, model_name, gbp_class):
    """
    Returns:
        pred_masks:   (2, D, H, W) float32 numpy — predicted probabilities
        saliency_l:   (D, H, W) float32 numpy — GBP for left parotid
        saliency_r:   (D, H, W) float32 numpy — GBP for right parotid
    """
    scan_tensor = scan_tensor.to(device)

    # --- segmentation ---
    with torch.no_grad():
        pred = model(scan_tensor)  # (1, 2, D, H, W)

    pred_masks = pred[0].cpu().numpy()  # (2, D, H, W)
    del pred
    torch.cuda.empty_cache()  # free before GBP to avoid running out of memory

    # --- GBP saliency ---
    print(f"  [{model_name}] Running GBP for left parotid...")
    gbp = gbp_class(model)
    saliency_l = gbp.generate(scan_tensor, class_idx=0)
    gbp.remove_hooks()

    print(f"  [{model_name}] Running GBP for right parotid...")
    gbp = gbp_class(model)
    saliency_r = gbp.generate(scan_tensor, class_idx=1)
    gbp.remove_hooks()

    return pred_masks, saliency_l, saliency_r


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_nrrd(array, path):
    """Save a (D, H, W) array back to nrrd, transposing to (W, H, D) convention."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # transpose back to nrrd convention (W, H, D)
    out = array.transpose(2, 1, 0).astype(np.float32)
    nrrd.write(path, out)
    print(f"  Saved: {path}  shape={out.shape}  range=[{out.min():.3f}, {out.max():.3f}]")


def save_results(pred_masks, saliency_l, saliency_r, out_dir, prefix):
    """Save predicted masks and saliency maps for one model."""

    print(f"pred_masks shape: {pred_masks.shape}")
    print(f"pred_masks[0] shape: {pred_masks[0].shape}")
    print(f"pred_masks[1] shape: {pred_masks[1].shape}")

    save_nrrd(pred_masks[0],  os.path.join(out_dir, f"{prefix}_pred_left.nrrd"))
    save_nrrd(pred_masks[1],  os.path.join(out_dir, f"{prefix}_pred_right.nrrd"))
    save_nrrd(saliency_l,     os.path.join(out_dir, f"{prefix}_saliency_left.nrrd"))
    save_nrrd(saliency_r,     os.path.join(out_dir, f"{prefix}_saliency_right.nrrd"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient",  type=str, required=True,  help="Patient folder name e.g. 0522c0001")
    parser.add_argument("--data",     type=str, required=True,  help="Path to dataset root")
    parser.add_argument("--basic",    type=str, default=None,   help="Path to vanilla best_model.pt")
    parser.add_argument("--cbam",     type=str, default=None,   help="Path to CBAM best_model.pt")
    parser.add_argument("--model",    type=str, default="both", choices=["basic", "cbam", "both"])
    parser.add_argument("--output",   type=str, default="results")
    args = parser.parse_args()

    # validate args
    if args.model in ("basic", "both") and args.basic is None:
        raise ValueError("--basic checkpoint path required when --model is basic or both")
    if args.model in ("cbam", "both") and args.cbam is None:
        raise ValueError("--cbam checkpoint path required when --model is cbam or both")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    out_dir = os.path.join(args.output, args.patient)
    os.makedirs(out_dir, exist_ok=True)

    # --- load patient ---
    print(f"\nLoading patient: {args.patient}")
    scan_tensor, scan_np, gt_mask = load_patient(args.data, args.patient)
    print(f"  Scan shape: {scan_np.shape}  range: [{scan_np.min():.2f}, {scan_np.max():.2f}]")
    print(f"  GT left voxels:  {gt_mask[0].sum():.0f}")
    print(f"  GT right voxels: {gt_mask[1].sum():.0f}")

    # also save the ground truth masks for easy comparison in viewer
    save_nrrd(gt_mask[0], os.path.join(out_dir, "gt_left.nrrd"))
    save_nrrd(gt_mask[1], os.path.join(out_dir, "gt_right.nrrd"))
    save_nrrd(scan_np,    os.path.join(out_dir, "scan.nrrd"))

    # --- vanilla ---
    if args.model in ("basic", "both"):
        print(f"\nRunning vanilla U-Net...")
        model = load_basic(args.basic, device)
        pred_masks, sal_l, sal_r = run_inference(
            model, scan_tensor, device, "basic", GuidedBackprop
        )
        save_results(pred_masks, sal_l, sal_r, out_dir, "basic")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # --- cbam ---
    if args.model in ("cbam", "both"):
        print(f"\nRunning CBAM U-Net...")
        model = load_cbam(args.cbam, device)
        pred_masks, sal_l, sal_r = run_inference(
            model, scan_tensor, device, "cbam", GuidedBackpropCBAM
        )
        save_results(pred_masks, sal_l, sal_r, out_dir, "cbam")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"\nDone. Results saved to {out_dir}/")
    print(f"Files:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")