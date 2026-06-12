# import torch
# from unet_basic import UNet3D

# model = UNet3D(n_channels=1, n_classes=2, input_shape=(48, 208, 272), base_filters=64)
# model.eval()

# dummy = torch.randn(1, 1, 48, 208, 272)
# with torch.no_grad():
#     out = model(dummy)
# print(out.shape)

# import numpy as np
# import nrrd

# def dice(pred, gt, smooth=1e-5):
#     pred_bin = (pred > 0.5).astype(np.float32)
#     intersection = (pred_bin * gt).sum()
#     return (2 * intersection + smooth) / (pred_bin.sum() + gt.sum() + smooth)

# def dice_midline(pred, gt):
#     # pred shape is (W, H, D) = (272, 208, 48) after nrrd load
#     # axis 1 is H = 208 = left-right axis

#     mid = pred_l.shape[1] // 2
#     # print(f"pred_l shape: {pred_l.shape}")
#     # print(f"GT left voxels in left half:  {gt_l[:, :mid, :].sum():.0f}")
#     # print(f"GT left voxels in right half: {gt_l[:, mid:, :].sum():.0f}")
#     # print(f"GT right voxels in left half:  {gt_r[:, :mid, :].sum():.0f}")
#     # print(f"GT right voxels in right half: {gt_r[:, mid:, :].sum():.0f}")

#     left_half  = pred[:, :mid, :]   # left side
#     right_half = pred[:, mid:, :]   # right side

#     # left parotid is in right half of axis 1
#     dsc_l_mid = dice(pred_l[:, mid:, :], gt_l[:, mid:, :])
#     # right parotid is in left half of axis 1
#     dsc_r_mid = dice(pred_r[:, :mid, :], gt_r[:, :mid, :])

#     # return dice(left_half, gt[:, :mid, :]), dice(right_half, gt[:, mid:, :])
#     return dsc_l_mid, dsc_r_mid

# import os
# results_dir = "results/0522c0416"

# # pred_l, _ = nrrd.read(os.path.join(results_dir, "basic_pred_left.nrrd"))
# # pred_r, _ = nrrd.read(os.path.join(results_dir, "basic_pred_right.nrrd"))

# pred_l, _ = nrrd.read(os.path.join(results_dir, "cbam_pred_left.nrrd"))
# pred_r, _ = nrrd.read(os.path.join(results_dir, "cbam_pred_right.nrrd"))

# gt_l,   _ = nrrd.read(os.path.join(results_dir, "gt_left.nrrd"))
# gt_r,   _ = nrrd.read(os.path.join(results_dir, "gt_right.nrrd"))

# print(f"Full DSC left:     {dice(pred_l, gt_l):.4f}")
# print(f"Full DSC right:    {dice(pred_r, gt_r):.4f}")

# dsc_l_mid, _ = dice_midline(pred_l, gt_l)
# _, dsc_r_mid = dice_midline(pred_r, gt_r)


# print(f"Midline DSC left:  {dsc_l_mid:.4f}")
# print(f"Midline DSC right: {dsc_r_mid:.4f}")


import numpy as np
import nrrd
import os

def dice(pred, gt, smooth=1e-5):
    pred_bin = (pred > 0.5).astype(np.float32)
    intersection = (pred_bin * gt).sum()
    return (2 * intersection + smooth) / (pred_bin.sum() + gt.sum() + smooth)

results_dir = "results/0522c0416"

gt_l,  _ = nrrd.read(os.path.join(results_dir, "gt_left.nrrd"))
gt_r,  _ = nrrd.read(os.path.join(results_dir, "gt_right.nrrd"))
gt_both  = np.logical_or(gt_l, gt_r).astype(np.float32)

print(f"\n{'─'*60}")
print(f"  {'Model':<10} {'DSC L':>8} {'DSC R':>8} {'Mean L+R':>10} {'Combined L':>10} {'Combined R':>10}")
print(f"{'─'*60}")

for prefix in ["basic", "cbam"]:
    pred_l, _ = nrrd.read(os.path.join(results_dir, f"{prefix}_pred_left.nrrd"))
    pred_r, _ = nrrd.read(os.path.join(results_dir, f"{prefix}_pred_right.nrrd"))
    # pred_both = np.logical_or(pred_l > 0.5, pred_r > 0.5).astype(np.float32)

    dsc_l    = dice(pred_l, gt_l)
    dsc_r    = dice(pred_r, gt_r)
    mean_lr  = (dsc_l + dsc_r) / 2
    combined_l = dice(pred_l, gt_both)
    combined_r = dice(pred_r, gt_both)

    print(f"  {prefix.upper():<10} {dsc_l:>8.4f} {dsc_r:>8.4f} {mean_lr:>10.4f} {combined_l:>10.4f} {combined_r:>10.4f}")

print(f"{'─'*60}")