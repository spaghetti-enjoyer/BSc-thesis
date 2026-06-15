"""
visualize_cbam.py — extract and save attended feature maps from every CBAM block.

For each CBAM block in the network we capture:
  - channel attention weights  (C,)
  - spatial attention map      (D_block, H_block, W_block)
  - raw feature maps           (C, D_block, H_block, W_block)

We then compute attended maps = feature_map[c] * channel_weight[c] * spatial_map
and save the top-k channels (by channel attention weight) as nrrd files,
upsampled to the full scan resolution for overlay in napari.

Usage:
    python src/visualize_cbam.py \
        --patient  0522c0416 \
        --data     parotid_PDDCA+deepmind \
        --cbam     models/cbam_best_model.pt \
        --output   results/ \
        --top_k    10

Output:
    results/0522c0416/cbam_attention/
        inc/
            channel_weights.npy          # (C,) channel attention weights
            spatial_map.nrrd             # spatial attention map at block resolution
            top_00_ch042_w0.91.nrrd      # attended feature map, channel 42, weight 0.91
            top_01_ch017_w0.88.nrrd
            ...
        down1/  down2/  down3/  down4/
        up1/    up2/    up3/    up4/
"""

import argparse
import os
import numpy as np
import nrrd
import torch
import torch.nn as nn
from scipy.ndimage import zoom

from dataset import normalize
from unet_cbam import UNet3D_CBAM


# ---------------------------------------------------------------------------
# Hook manager
# ---------------------------------------------------------------------------

class CBAMHookManager:
    """
    Registers forward hooks on every CBAM block in the model to capture:
      - feature maps before CBAM  (raw conv output)
      - channel attention weights
      - spatial attention map
    """

    BLOCK_NAMES = ['inc', 'down1', 'down2', 'down3', 'down4',
                   'up1',  'up2',  'up3',  'up4']

    def __init__(self, model: UNet3D_CBAM):
        self.model = model
        self.hooks = []
        self.data  = {name: {} for name in self.BLOCK_NAMES}
        self._register()

    def _register(self):
        for block_name in self.BLOCK_NAMES:
            block = getattr(self.model, block_name)

            # get the DoubleConv — inc/down* have it at .conv or directly
            if block_name == 'inc':
                double_conv = block
            elif block_name.startswith('down'):
                double_conv = block.conv
            else:  # up*
                double_conv = block.conv

            cbam = double_conv.cbam
            if cbam is None:
                continue

            # closure to capture block_name
            def make_hooks(name):
                def feat_hook(m, i, o):
                    self.data[name]['feature_maps'] = o.detach().cpu()

                def chan_hook(m, i, o):
                    self.data[name]['channel_weights'] = o.detach().cpu()

                def spat_hook(m, i, o):
                    self.data[name]['spatial_map'] = o.detach().cpu()

                return feat_hook, chan_hook, spat_hook

            fh, ch, sh = make_hooks(block_name)

            # feature maps = output of double_conv (before CBAM multiplies)
            self.hooks.append(
                double_conv.double_conv.register_forward_hook(fh)
            )
            # channel attention = output of channel_att sigmoid
            self.hooks.append(
                cbam.channel_att.sigmoid.register_forward_hook(ch)
            )
            # spatial attention = output of spatial_att sigmoid
            self.hooks.append(
                cbam.spatial_att.sigmoid.register_forward_hook(sh)
            )

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract(model, scan_tensor, device):
    manager = CBAMHookManager(model)
    model.eval()
    with torch.no_grad():
        model(scan_tensor.to(device))
    manager.remove()
    return manager.data


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_nrrd(array, path):
    """Save (D, H, W) array as nrrd, transposed to (W, H, D) convention."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = array.transpose(2, 1, 0).astype(np.float32)
    nrrd.write(path, out)


def upsample_to(volume, target_shape):
    """Upsample (D, H, W) volume to target_shape using trilinear zoom."""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)


def process_block(block_data, block_name, out_dir, scan_shape, top_k):
    """
    Process one CBAM block's captured data and save outputs.

    block_data: dict with 'feature_maps', 'channel_weights', 'spatial_map'
    scan_shape: (D, H, W) of the full resolution scan
    """
    if not block_data:
        print(f"  [{block_name}] no data captured, skipping")
        return

    block_dir = os.path.join(out_dir, block_name)
    os.makedirs(block_dir, exist_ok=True)

    # --- unpack ---
    # feature_maps:    (1, C, D_b, H_b, W_b)
    # channel_weights: (1, C, 1,   1,   1  )
    # spatial_map:     (1, 1, D_b, H_b, W_b)
    fm = block_data['feature_maps'][0].numpy()   # (C, D_b, H_b, W_b)
    cw = block_data['channel_weights'][0, :, 0, 0, 0].numpy()  # (C,)
    sm = block_data['spatial_map'][0, 0].numpy() # (D_b, H_b, W_b)

    print(f"  [{block_name}] feature maps {fm.shape}  "
          f"channel weights {cw.shape}  spatial map {sm.shape}")

    # --- save channel weights ---
    np.save(os.path.join(block_dir, 'channel_weights.npy'), cw)

    # --- save spatial map upsampled to full res ---
    sm_full = upsample_to(sm, scan_shape)
    if sm_full.max() > 0:
        sm_full = sm_full / sm_full.max()
    save_nrrd(sm_full, os.path.join(block_dir, 'spatial_map.nrrd'))

    # --- top-k channels by attention weight ---
    top_k = min(top_k, len(cw))
    top_indices = np.argsort(cw)[::-1][:top_k]

    for rank, ch_idx in enumerate(top_indices):
        weight = cw[ch_idx]

        # attended map = feature_map * channel_weight * spatial_map
        attended = fm[ch_idx] * weight * sm  # (D_b, H_b, W_b)

        # upsample to full scan resolution
        attended_full = upsample_to(attended, scan_shape)

        # normalise to 0-1
        if attended_full.max() > attended_full.min():
            attended_full = (attended_full - attended_full.min()) / \
                            (attended_full.max() - attended_full.min())

        fname = f"top_{rank:02d}_ch{ch_idx:03d}_w{weight:.2f}.nrrd"
        save_nrrd(attended_full, os.path.join(block_dir, fname))

    print(f"  [{block_name}] saved spatial map + top {top_k} channels")
    print(f"  [{block_name}] top channels: " +
          ", ".join(f"ch{i}({cw[i]:.2f})" for i in top_indices))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient",      type=str, required=True)
    parser.add_argument("--data",         type=str, required=True)
    parser.add_argument("--cbam",         type=str, required=True)
    parser.add_argument("--output",       type=str, default="results")
    parser.add_argument("--base_filters", type=int, default=64)
    parser.add_argument("--top_k",        type=int, default=10)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # --- load model ---
    model = UNet3D_CBAM(
        n_channels=1, n_classes=1,
        input_shape=(48, 208, 272),
        bilinear=False,
        base_filters=args.base_filters,
    ).to(device)
    ckpt = torch.load(args.cbam, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded CBAM model (epoch {ckpt['epoch']}, DSC {ckpt['mean_dsc']:.4f})")

    # --- load patient ---
    import nrrd as _nrrd
    scan_path = os.path.join(args.data, args.patient, 'img.nrrd')
    scan_raw, _ = _nrrd.read(scan_path)
    scan = scan_raw.transpose(2, 0, 1).astype(np.float32)  # (D, H, W)
    scan = normalize(scan)
    scan_shape = scan.shape  # (48, 208, 272)

    scan_tensor = torch.from_numpy(scan[np.newaxis, np.newaxis])  # (1,1,D,H,W)

    # --- extract attention ---
    print(f"\nExtracting CBAM attention for patient {args.patient}...")
    block_data = extract(model, scan_tensor, device)

    # --- save per block ---
    out_dir = os.path.join(args.output, args.patient, 'cbam_attention')
    os.makedirs(out_dir, exist_ok=True)

    # also save the scan for reference in viewer
    save_nrrd(scan, os.path.join(out_dir, 'scan.nrrd'))

    for block_name in CBAMHookManager.BLOCK_NAMES:
        process_block(block_data[block_name], block_name, out_dir,
                      scan_shape, args.top_k)

    print(f"\nDone. Results saved to {out_dir}/")
    print("Blocks saved:", sorted(os.listdir(out_dir)))